"""Runs the models on a puzzle, logs every move and saves the results.

Usage from the project folder:
    python -m backend.benchmark primary-1      # one puzzle
    python -m backend.benchmark primary        # every puzzle in a set
"""

import asyncio
import inspect
import json
import re
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from backend import config
from backend.adapters import load_adapters
from backend.adapters.base import AdapterError
from backend.api_guard import CallLimitReached
from backend.cube_engine import apply_move, correct_stickers, is_solved
from backend.puzzles import load_puzzles

SOLVED = "SOLVED"
RAN_OUT_OF_MOVES = "RAN OUT OF MOVES"
OUT_OF_TIME = "OUT OF TIME"
CALL_LIMIT_REACHED = "CALL LIMIT REACHED"


async def emit(on_event, event: dict) -> None:
    # The hook can be a normal function or an async one (like a WebSocket send).
    if on_event:
        returned = on_event(event)
        if inspect.isawaitable(returned):
            await returned


# ---- Event messages, shared by live races and replays so the GUI sees the same thing ----

def race_start_event(race_id, puzzle_id, scramble, state, models, replay=False, one_by_one=False) -> dict:
    return {"type": "race_start", "race_id": race_id, "puzzle_id": puzzle_id, "scramble": scramble,
            "state": state, "models": models, "replay": replay, "one_by_one": one_by_one}


def start_event(model, puzzle_id, state) -> dict:
    return {"type": "start", "model": model, "puzzle_id": puzzle_id, "state": state}


def move_event(model, puzzle_id, entry, state, elapsed) -> dict:
    return {"type": "move", "model": model, "puzzle_id": puzzle_id, "number": entry["number"],
            "move": entry["move"], "valid": entry["valid"], "latency": entry["latency"],
            "retries": entry["retries"], "cache_hit": entry.get("cache_hit", False),
            "failed": bool(entry.get("attempts")) and entry["attempts"][-1].get("call_failed", False),
            "state": state, "elapsed": elapsed}


def finish_event(result) -> dict:
    return {"type": "finish", "model": result["model"], "puzzle_id": result["puzzle_id"],
            "end_reason": result["end_reason"], "solved": result["solved"], "moves": result["move_count"],
            "total_time": result["total_time"], "optimality_gap": result["optimality_gap"]}


def race_end_event(race_id, results) -> dict:
    # api: what this race used (a dry run shows what a real run would have used)
    api = {"calls": sum(r.get("api_calls", 0) for r in results),
           "cache_hits": sum(r.get("cache_hits", 0) for r in results),
           "cost": sum(r.get("total_cost", 0.0) for r in results),
           "dry_run": any(r.get("dry_run") for r in results)}
    return {"type": "race_end", "race_id": race_id, "ranking": rank_results(results), "api": api}


def rank_key(result: dict) -> tuple:
    # Solved first (fewer moves, then less time), then unsolved (more correct stickers, then less time).
    if result["solved"]:
        return (0, result["move_count"], result["total_time"])
    return (1, -correct_stickers(result["final_state"]), result["total_time"])


def rank_results(results: list[dict]) -> list[dict]:
    """Every model gets a rank from 1 to 4. Rows stay in the race (lane) order."""
    ranks = {r["model"]: i + 1 for i, r in enumerate(sorted(results, key=rank_key))}
    return [{"model": r["model"], "rank": ranks[r["model"]], "end_reason": r["end_reason"],
             "moves": r["move_count"], "total_time": r["total_time"], "optimality_gap": r["optimality_gap"],
             "correct_stickers": correct_stickers(r["final_state"])}
            for r in results]


LATE_GRACE = 0.5  # seconds; an answer that arrives later than timeout + this is not accepted

# Per request logging fields from the api_guard (OpenRouter only; None for the other modes)
API_FIELDS = {"cache_hit": None, "api_latency": None, "cache_latency": None,
              "request_number": None, "cost": None, "dry_run": None}


async def ask_once(adapter, state: str, recent_moves: list[str], timeout: float) -> dict:
    """One model call, timed. Returns a log entry for this attempt.

    CallLimitReached is not caught here: it must stop the model at once, not count as a failed call.
    """
    start_time = time.perf_counter()
    result, failure, api_info = None, None, None
    try:
        result = await asyncio.wait_for(adapter.get_move(state, recent_moves), timeout)
        api_info = result.api_info
    except CallLimitReached:
        raise
    except TimeoutError:
        failure = f"timed out after {timeout:.1f}s"
    except AdapterError as e:
        failure, api_info = str(e), e.info
    except Exception as e:  # a bug in one adapter must not stop the whole race
        failure = f"unexpected error: {e!r}"
    latency = time.perf_counter() - start_time

    # The timeout is a hard rule: if the computer was paused (for example by sleep) and the
    # answer only came in after the limit, it is not accepted.
    if result is not None and latency > timeout + LATE_GRACE:
        result, failure = None, f"answer came after the {timeout:.0f}s limit ({latency:.0f}s)"

    entry = {"latency": latency, **API_FIELDS, **(api_info or {})}
    if result is None:
        return {**entry, "output": None, "move": None, "confidence": None,
                "call_failed": True, "error": failure, "raw_response": None}
    return {**entry, "output": result.output, "move": result.move, "confidence": result.confidence,
            "call_failed": False, "error": result.error, "raw_response": result.raw_response}


async def ask_for_move(adapter, state: str, recent_moves: list[str], run_start: float) -> dict:
    """Ask for one move, with at most adapter.retries() retries. Returns the move's log entry."""
    attempts = []
    move_start = time.perf_counter()
    for _ in range(1 + adapter.retries()):  # fixed maximum, never an open ended loop
        now = time.perf_counter()
        time_left = min(adapter.timeout(), config.RUN_TIME_LIMIT - (now - run_start))
        if adapter.timeout_covers_retries:
            time_left = min(time_left, adapter.timeout() - (now - move_start))
        if time_left <= 0:
            break
        attempt = await ask_once(adapter, state, recent_moves, time_left)
        attempts.append(attempt)
        if attempt["move"]:
            break

    final = attempts[-1] if attempts else {}
    return {
        "move": final.get("move"),
        "valid": bool(final.get("move")),
        "latency": sum(a["latency"] for a in attempts),  # includes time spent on retries
        "retries": max(len(attempts) - 1, 0),
        "confidence": final.get("confidence"),
        "cache_hit": bool(attempts) and all(a["cache_hit"] for a in attempts),
        "attempts": attempts,
    }


async def run_model(adapter, puzzle: dict, race_id: str = "", on_event=None) -> dict:
    """Play one model on one puzzle until solved, out of moves or out of time."""
    state = puzzle["state"]
    applied = []   # moves that actually changed the cube
    move_log = []
    end_reason = None
    started_at = datetime.now().isoformat(timespec="seconds")
    run_start = time.perf_counter()

    limit_message = None
    await emit(on_event, start_event(adapter.name, puzzle["id"], state))

    # Every round either makes a move or ends the run, so MAX_MOVES + 1 rounds always reach an end reason.
    for _ in range(config.MAX_MOVES + 1):
        if is_solved(state):
            end_reason = SOLVED
        elif len(move_log) >= config.MAX_MOVES:
            end_reason = RAN_OUT_OF_MOVES
        elif time.perf_counter() - run_start >= config.RUN_TIME_LIMIT:
            end_reason = OUT_OF_TIME
        if end_reason:
            break
        try:
            entry = await ask_for_move(adapter, state, applied, run_start)
        except CallLimitReached as e:  # stop this model right away
            end_reason, limit_message = CALL_LIMIT_REACHED, str(e)
            await emit(on_event, {"type": "warning", "message": f"{adapter.name}: call limit reached ({e})"})
            break
        # Hard limit: a move still running at the limit is cancelled and not counted.
        if not entry["attempts"] or time.perf_counter() - run_start > config.RUN_TIME_LIMIT:
            end_reason = OUT_OF_TIME
            break
        # An invalid move still uses up one of the moves, but the cube stays as it is.
        if entry["valid"]:
            state = apply_move(state, entry["move"])
            applied.append(entry["move"])
        entry["number"] = len(move_log) + 1
        move_log.append(entry)
        elapsed = time.perf_counter() - run_start
        await emit(on_event, move_event(adapter.name, puzzle["id"], entry, state, elapsed))

    total_time = min(time.perf_counter() - run_start, config.RUN_TIME_LIMIT)  # never shows more than the limit
    result = build_result(adapter, puzzle, race_id, started_at, move_log, state, total_time, end_reason)
    result["limit_message"] = limit_message
    await emit(on_event, finish_event(result))
    return result


def build_result(adapter, puzzle, race_id, started_at, move_log, final_state, total_time, end_reason) -> dict:
    attempts = [a for entry in move_log for a in entry["attempts"]]
    solved = end_reason == SOLVED
    optimal = puzzle.get("optimal_distance")
    return {
        "race_id": race_id,
        "model": adapter.name,
        "model_id": adapter.model_id,
        "kind": adapter.kind,
        "mode": adapter.mode,
        "puzzle_id": puzzle["id"],
        "puzzle_set": puzzle["set"],
        "scramble": puzzle["scramble"],
        "start_state": puzzle["state"],
        "final_state": final_state,
        "started_at": started_at,
        "moves": [entry["move"] for entry in move_log],  # None where the output was invalid
        "move_count": len(move_log),
        "latencies": [entry["latency"] for entry in move_log],
        "retries": sum(entry["retries"] for entry in move_log),
        "invalid_outputs": sum(1 for a in attempts if not a["move"] and not a["call_failed"]),
        "failed_calls": sum(1 for a in attempts if a["call_failed"]),
        "confidences": [entry["confidence"] for entry in move_log],
        # OpenRouter numbers (0 in the other modes). Cache hits are not counted as API calls.
        "api_calls": sum(1 for a in attempts if a["request_number"] is not None),
        "cache_hits": sum(1 for a in attempts if a["cache_hit"]),
        "total_cost": sum(a["cost"] or 0.0 for a in attempts),
        "api_latencies": [a["api_latency"] for a in attempts if a["api_latency"] is not None],
        "dry_run": any(a["dry_run"] for a in attempts),
        "total_time": total_time,
        "solved": solved,
        "end_reason": end_reason,
        "optimal_distance": optimal,
        "optimality_gap": len(move_log) - optimal if solved and optimal is not None else None,
        "move_log": move_log,
    }


def make_race_id(puzzle_id: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{puzzle_id}"


async def run_turn(adapter, puzzle: dict, race_id: str, on_event) -> dict:
    """One model's turn when models race one after another (Ollama mode)."""
    # Loading happens before the run clock starts, so it does not count as move latency.
    try:
        await adapter.prepare()
    except AdapterError as e:
        await emit(on_event, {"type": "warning", "message": f"Could not load {adapter.name}: {e}"})
    try:
        return await run_model(adapter, puzzle, race_id, on_event)
    finally:
        try:
            await adapter.release()
        except AdapterError:
            pass  # if unloading fails, Ollama frees the memory itself later


@contextmanager
def keep_awake():
    """Stop Windows from going to sleep during a race. A sleeping laptop breaks every timing."""
    if sys.platform != "win32":
        yield
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


async def run_race(puzzle: dict, adapters=None, on_event=None, save=True) -> list[dict]:
    """All models on the same puzzle, each with its own cube state.

    Normally they run at the same time. Ollama models take turns, because a laptop
    can only hold one local model in memory at once. Raises AdapterError with a clear
    message if a model is not ready (for example Ollama is not running).
    """
    adapters = adapters or load_adapters()
    for adapter in adapters:
        await adapter.check_ready()
    race_id = make_race_id(puzzle["id"])
    models = [{"name": a.name, "kind": a.kind} for a in adapters]
    one_by_one = any(a.runs_alone for a in adapters)
    await emit(on_event, race_start_event(race_id, puzzle["id"], puzzle["scramble"], puzzle["state"], models,
                                          one_by_one=one_by_one))

    with keep_awake():
        if one_by_one:
            # Turns follow OLLAMA_RUN_ORDER; results (and panels) stay in lane order.
            turn = {name: i for i, name in enumerate(config.OLLAMA_RUN_ORDER)}
            done = {}
            for a in sorted(adapters, key=lambda a: turn.get(a.name, len(turn))):
                done[a.name] = await run_turn(a, puzzle, race_id, on_event)
            results = [done[a.name] for a in adapters]
        else:
            results = list(await asyncio.gather(*(run_model(a, puzzle, race_id, on_event) for a in adapters)))
    for lane, result in enumerate(results):
        result["lane"] = lane  # position in the race, so saved races load back in the same order
        if save:
            save_result(result)
    await emit(on_event, race_end_event(race_id, results))
    return results


# ---- Replay of a saved race ----

def replay_timeline(results: list[dict]) -> list[tuple[float, dict]]:
    """Rebuild every event of a saved race with the time it happened, using the recorded latencies."""
    timeline = []
    for r in results:
        state, clock = r["start_state"], 0.0
        timeline.append((0.0, start_event(r["model"], r["puzzle_id"], state)))
        for entry in r["move_log"]:
            clock += entry["latency"]
            if entry["valid"]:
                state = apply_move(state, entry["move"])
            timeline.append((clock, move_event(r["model"], r["puzzle_id"], entry, state, clock)))
        timeline.append((max(clock, r["total_time"]), finish_event(r)))
    timeline.sort(key=lambda item: item[0])  # stable sort keeps each model's events in order
    return timeline


async def replay_race(results: list[dict], speed: float, on_event) -> None:
    first = results[0]
    models = [{"name": r["model"], "kind": r.get("kind", "")} for r in results]
    await emit(on_event, race_start_event(first["race_id"], first["puzzle_id"], first["scramble"],
                                          first["start_state"], models, replay=True))
    # Wait until each event's time on one clock, so small sleep delays do not add up.
    replay_start = time.perf_counter()
    for at, event in replay_timeline(results):
        wait = replay_start + at / speed - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        await emit(on_event, event)
    await emit(on_event, race_end_event(first["race_id"], results))


# ---- Saving and loading ----

def save_result(result: dict, folder: Path | None = None) -> Path:
    folder = folder or config.RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    model_slug = re.sub(r"[^a-z0-9]+", "-", result["model"].lower()).strip("-")
    path = folder / f"{result['race_id']}_{model_slug}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def load_results(folder: Path | None = None) -> list[dict]:
    folder = folder or config.RESULTS_DIR
    results = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
    return sorted(results, key=lambda r: (r["race_id"], r.get("lane", 0)))


# ---- Dashboard stats ----

def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(results: list[dict]) -> list[dict]:
    """One row of dashboard stats per model, in the order the models first appear."""
    by_model = {}
    for result in results:
        by_model.setdefault(result["model"], []).append(result)

    summary = []
    for model, runs in by_model.items():
        solved = [r for r in runs if r["solved"]]
        summary.append({
            "model": model,
            "runs": len(runs),
            "solved": len(solved),
            "success_rate": 100 * len(solved) / len(runs),
            "avg_latency": average([lat for r in runs for lat in r["latencies"]]),
            "avg_total_time": average([r["total_time"] for r in runs]),
            "avg_moves": average([r["move_count"] for r in runs]),
            "avg_optimality_gap": average([r["optimality_gap"] for r in solved]),
            "invalid_outputs": sum(r["invalid_outputs"] for r in runs),
            "retries": sum(r["retries"] for r in runs),
            "api_calls": sum(r.get("api_calls", 0) for r in runs),
            "cache_hits": sum(r.get("cache_hits", 0) for r in runs),
            "total_cost": sum(r.get("total_cost", 0.0) for r in runs),
        })
    return summary


def puzzle_table(results: list[dict], puzzles: list[dict]) -> list[dict]:
    """One row per puzzle with each model's latest run on it (moves used and solved or not)."""
    latest = {}
    for r in results:  # results are sorted by race_id, which starts with the date, so later runs win
        latest[(r["puzzle_id"], r["model"])] = r

    rows = []
    for p in puzzles:
        cells = {model: {"moves": r["move_count"], "solved": r["solved"], "end_reason": r["end_reason"]}
                 for (puzzle_id, model), r in latest.items() if puzzle_id == p["id"]}
        rows.append({"puzzle_id": p["id"], "scramble": p["scramble"],
                     "optimal_distance": p["optimal_distance"], "cells": cells})
    return rows


def best_models(summary: list[dict], key: str, highest: bool):
    """Models with the best value for one stat (more than one if they tie), and that value."""
    rows = [r for r in summary if r[key] is not None]
    if not rows:
        return [], None
    value = (max if highest else min)(r[key] for r in rows)
    return [r["model"] for r in rows if r[key] == value], value


def join_names(names: list[str]) -> str:
    # ["A"] -> "A", ["A", "B"] -> "A and B", ["A", "B", "C"] -> "A, B and C"
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def overall_key(row: dict) -> tuple:
    # "Best" overall = highest solve rate, then smallest optimality gap.
    gap = row["avg_optimality_gap"]
    return (-row["success_rate"], gap if gap is not None else float("inf"))


def key_findings(summary: list[dict]) -> list[dict]:
    fastest, latency = best_models(summary, "avg_latency", highest=False)
    if not fastest:  # no results, or no model made a single move
        return []
    solvers, rate = best_models(summary, "success_rate", highest=True)
    closest, gap = best_models(summary, "avg_optimality_gap", highest=False)

    findings = [{"label": "Fastest", "text": f"{join_names(fastest)} ({latency:.2f}s per move on average)"}]
    if rate == 0:
        findings.append({"label": "Highest solve rate", "text": "No model solved any puzzle (0% for all)"})
    else:
        findings.append({"label": "Highest solve rate",
                         "text": f"{join_names(solvers)} ({rate:.0f}% of puzzles solved)"})
    if closest:
        findings.append({"label": "Closest to optimal",
                         "text": f"{join_names(closest)} (average gap {gap:.1f} moves on solved puzzles)"})
    else:
        findings.append({"label": "Closest to optimal", "text": "No model solved a puzzle, so there is no gap yet"})

    if rate == 0:  # "best" only means something once a model has solved a puzzle
        findings.append({"label": "Fastest also the best",
                         "text": "Can't tell yet, because no model solved a puzzle"})
        return findings

    best = min(summary, key=overall_key)
    fast = next(r for r in summary if r["model"] == fastest[0])
    if overall_key(fast) == overall_key(best):  # a tie for best counts as best
        text = f"Yes. {fast['model']} was the fastest and also the best (highest solve rate, then closest to optimal)"
    elif fast["success_rate"] < best["success_rate"]:
        text = (f"No. {fast['model']} was the fastest, but {best['model']} was the best: "
                f"it solved {best['success_rate']:.0f}% against {fast['success_rate']:.0f}%")
    else:
        text = (f"No. {fast['model']} was the fastest, but {best['model']} solved just as many "
                f"and was closer to optimal")
    findings.append({"label": "Fastest also the best", "text": text})
    return findings


def print_summary(summary: list[dict]) -> None:
    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print(f"{'model':<22}{'solved':>8}{'success':>9}{'latency':>10}{'time':>9}{'moves':>7}{'gap':>6}{'invalid':>9}")
    for row in summary:
        print(f"{row['model']:<22}{row['solved']:>4}/{row['runs']:<3}{fmt(row['success_rate'], '.0f'):>8}%"
              f"{fmt(row['avg_latency'], '.3f'):>9}s{fmt(row['avg_total_time'], '.2f'):>8}s"
              f"{fmt(row['avg_moves'], '.1f'):>7}{fmt(row['avg_optimality_gap'], '.1f'):>6}{row['invalid_outputs']:>9}")


async def main(target: str) -> None:
    # Teacher's rule: the real OpenRouter run is only ever started from the GUI, never from here.
    if config.active_mode() == "openrouter":
        sys.exit("MODE=openrouter: the real run can only be started from the race screen in the GUI.")
    puzzles = [p for p in load_puzzles() if target in (p["id"], p["set"])]
    if not puzzles:
        sys.exit(f"No puzzle or set called {target!r}")

    all_results = []
    for puzzle in puzzles:
        try:
            results = await run_race(puzzle)
        except AdapterError as e:  # for example Ollama is not running
            sys.exit(str(e))
        for r in results:
            print(f"{puzzle['id']:<12} {r['model']:<22} {r['end_reason']:<17} moves {r['move_count']}")
        all_results.extend(results)
    print()
    print_summary(summarize(all_results))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "primary-1"))
