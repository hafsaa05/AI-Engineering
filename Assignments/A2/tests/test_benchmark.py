import asyncio
import time

import pytest

from backend import config
from backend.adapters.base import Adapter, AdapterError, make_result
from backend.adapters.mock_adapter import BrokenMock, NoisyMock, OracleMock, RandomMock
from backend.benchmark import (
    key_findings, load_results, puzzle_table, rank_results, run_model, run_race, save_result, summarize,
)
from backend.cube_engine import get_state
from backend.puzzles import get_puzzle

PUZZLE = get_puzzle("primary-3")  # optimal distance 5


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch):
    # Never write test runs into the real results/ folder, and never use a mode set in .env.
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODE", "mock")


class ScriptedAdapter(Adapter):
    """Test adapter that plays back a fixed list of outputs. "FAIL" raises, "SLOW" hangs."""

    def __init__(self, outputs):
        super().__init__("scripted", "scripted")
        self.outputs = list(outputs)

    async def get_move(self, state, recent_moves):
        output = self.outputs.pop(0) if self.outputs else "U"
        if output == "FAIL":
            raise AdapterError("HTTP 500: server error")
        if output == "SLOW":
            await asyncio.sleep(10)
        return make_result(output, raw_response=output)


def play(adapter, puzzle=PUZZLE, on_event=None):
    return asyncio.run(run_model(adapter, puzzle, "test-race", on_event))


def test_oracle_solves_with_zero_gap():
    result = play(OracleMock("mock-oracle", delay=0))
    assert result["solved"] and result["end_reason"] == "SOLVED"
    assert result["move_count"] == result["optimal_distance"] == 5
    assert result["optimality_gap"] == 0
    assert result["retries"] == result["invalid_outputs"] == 0


def test_random_runs_out_of_moves(monkeypatch):
    monkeypatch.setattr(config, "MAX_MOVES", 6)
    result = play(RandomMock("mock-random", seed=1))
    assert result["end_reason"] == "RAN OUT OF MOVES"
    assert result["move_count"] == 6 and not result["solved"]
    assert result["optimality_gap"] is None


def test_invalid_output_is_retried_then_wastes_a_move(monkeypatch):
    monkeypatch.setattr(config, "MAX_MOVES", 2)
    result = play(ScriptedAdapter(["R3", "x", None, "R3", "R", "U"]))
    first, second = result["move_log"]

    # Move 1: three bad answers, so the move is used up and the cube does not change.
    assert not first["valid"] and first["retries"] == 2 and len(first["attempts"]) == 3
    assert [a["output"] for a in first["attempts"]] == ["R3", "x", None]
    # Move 2: one bad answer, then "R" on the retry.
    assert second["valid"] and second["move"] == "R" and second["retries"] == 1

    assert result["moves"] == [None, "R"]
    assert result["invalid_outputs"] == 4 and result["retries"] == 3
    assert result["final_state"] != PUZZLE["state"]


def test_failed_calls_are_retried_and_counted(monkeypatch):
    monkeypatch.setattr(config, "MAX_MOVES", 1)
    result = play(ScriptedAdapter(["FAIL", "U"]))
    attempt = result["move_log"][0]["attempts"][0]
    assert attempt["call_failed"] and "HTTP 500" in attempt["error"]
    assert result["failed_calls"] == 1 and result["invalid_outputs"] == 0
    assert result["moves"] == ["U"]


def test_slow_call_times_out_and_is_retried(monkeypatch):
    monkeypatch.setattr(config, "MOVE_TIMEOUT", 0.1)
    monkeypatch.setattr(config, "MAX_MOVES", 1)
    result = play(ScriptedAdapter(["SLOW", "U"]))
    attempt = result["move_log"][0]["attempts"][0]
    assert attempt["call_failed"] and "timed out" in attempt["error"]
    assert result["moves"] == ["U"]


def test_one_timeout_for_the_whole_move_when_retries_share_it(monkeypatch):
    # Like Ollama: three slow attempts must still fit in one move timeout, not three.
    monkeypatch.setattr(config, "MOVE_TIMEOUT", 0.2)
    monkeypatch.setattr(config, "MAX_MOVES", 1)
    adapter = ScriptedAdapter(["SLOW"] * 3)
    adapter.timeout_covers_retries = True
    result = play(adapter)
    assert result["move_log"][0]["latency"] < 0.2 + 0.1
    assert not result["move_log"][0]["valid"]


def test_out_of_time(monkeypatch):
    monkeypatch.setattr(config, "RUN_TIME_LIMIT", 0.3)
    start_time = time.perf_counter()
    result = play(ScriptedAdapter(["SLOW"] * 10))
    assert result["end_reason"] == "OUT OF TIME" and not result["solved"]
    assert time.perf_counter() - start_time < 1


def test_broken_mock_still_logs_everything(monkeypatch):
    monkeypatch.setattr(config, "MAX_MOVES", 30)
    result = play(BrokenMock("mock-broken", delay=0, seed=4))
    assert result["invalid_outputs"] > 0
    assert len(result["latencies"]) == result["move_count"]
    assert result["end_reason"] in ("SOLVED", "RAN OUT OF MOVES")


def test_events_after_every_move():
    events = []

    async def on_event(event):  # async hook, like a WebSocket send
        events.append(event)

    result = play(OracleMock("mock-oracle", delay=0), on_event=on_event)
    assert [e["type"] for e in events] == ["start"] + ["move"] * 5 + ["finish"]
    assert [e["number"] for e in events[1:-1]] == [1, 2, 3, 4, 5]
    assert events[-2]["state"] == result["final_state"]
    assert events[-1]["end_reason"] == "SOLVED"


def test_race_runs_in_parallel_and_saves(tmp_path):
    adapters = [OracleMock(f"oracle-{i}", delay=0.1) for i in range(4)]
    start_time = time.perf_counter()
    results = asyncio.run(run_race(PUZZLE, adapters))
    # 4 models x 5 moves x 0.1s = 2s if done one after another.
    assert time.perf_counter() - start_time < 1.2

    assert len({r["race_id"] for r in results}) == 1
    assert all(r["solved"] for r in results)
    saved = load_results(tmp_path)
    assert sorted(r["model"] for r in saved) == [f"oracle-{i}" for i in range(4)]


def test_four_mocks_on_same_puzzle():
    adapters = [RandomMock("mock-random", seed=1), OracleMock("mock-oracle", delay=0),
                NoisyMock("mock-noisy", delay=0, seed=2), BrokenMock("mock-broken", delay=0, seed=3)]
    results = asyncio.run(run_race(PUZZLE, adapters, save=False))
    assert {r["puzzle_id"] for r in results} == {"primary-3"}
    assert {r["start_state"] for r in results} == {PUZZLE["state"]}
    assert results[1]["optimality_gap"] == 0


def test_saved_result_has_all_fields(tmp_path):
    result = play(OracleMock("Claude Opus 5.5", delay=0))
    path = save_result(result, tmp_path)
    assert path.name == "test-race_claude-opus-5-5.json"
    saved = load_results(tmp_path)[0]
    for field in ["model", "kind", "puzzle_id", "scramble", "moves", "latencies", "retries", "invalid_outputs",
                  "total_time", "solved", "end_reason", "optimal_distance", "optimality_gap", "confidences"]:
        assert field in saved


def ranked(model, solved, moves, total_time, final_moves=()):
    return {"model": model, "solved": solved, "move_count": moves, "total_time": total_time,
            "final_state": get_state(list(final_moves)), "end_reason": "SOLVED" if solved else "RAN OUT OF MOVES",
            "optimality_gap": None}


def test_ranking_gives_everyone_a_place():
    results = [
        ranked("slow-solver", True, 5, 9.0),
        ranked("far-from-solved", False, 30, 1.0, ["R", "U", "F"]),
        ranked("long-solver", True, 7, 1.0),
        ranked("almost-solved", False, 30, 8.0, ["R"]),
        ranked("almost-solved-faster", False, 30, 2.0, ["U"]),
    ]
    ranks = {row["model"]: row["rank"] for row in rank_results(results)}
    assert ranks == {
        "slow-solver": 1,           # fewer moves beats being faster
        "long-solver": 2,
        "almost-solved-faster": 3,  # same sticker count as below, so time decides
        "almost-solved": 4,
        "far-from-solved": 5,       # fewest correct stickers
    }
    assert [row["model"] for row in rank_results(results)] == [r["model"] for r in results]


def test_ranking_tie_on_moves_goes_to_faster():
    ranks = {row["model"]: row["rank"] for row in rank_results([
        ranked("a", True, 5, 2.0), ranked("b", True, 5, 1.0)])}
    assert ranks == {"b": 1, "a": 2}


def test_puzzle_table_uses_latest_run():
    puzzles = [get_puzzle("warmup-1"), get_puzzle("warmup-2")]
    results = [
        {"race_id": "20260101-000000_warmup-1", "puzzle_id": "warmup-1", "model": "A",
         "move_count": 30, "solved": False, "end_reason": "RAN OUT OF MOVES"},
        {"race_id": "20260102-000000_warmup-1", "puzzle_id": "warmup-1", "model": "A",
         "move_count": 4, "solved": True, "end_reason": "SOLVED"},
    ]
    first, second = puzzle_table(results, puzzles)
    assert first["puzzle_id"] == "warmup-1" and first["optimal_distance"] == 4
    assert first["cells"]["A"] == {"moves": 4, "solved": True, "end_reason": "SOLVED"}
    assert second["cells"] == {}  # no runs on this puzzle yet


def summary_row(model, latency, rate, gap):
    return {"model": model, "avg_latency": latency, "success_rate": rate, "avg_optimality_gap": gap}


def findings_text(summary):
    return {f["label"]: f["text"] for f in key_findings(summary)}


def test_findings_when_fastest_is_not_best():
    found = findings_text([summary_row("Quick", 0.1, 20, 3.0), summary_row("Careful", 2.0, 90, 0.5)])
    assert found["Fastest"].startswith("Quick (0.10s")
    assert found["Highest solve rate"].startswith("Careful (90%")
    assert found["Closest to optimal"].startswith("Careful (average gap 0.5")
    assert found["Fastest also the best"].startswith("No. Quick was the fastest, but Careful")


def test_findings_when_fastest_is_best():
    found = findings_text([summary_row("Star", 0.1, 100, 0.0), summary_row("Other", 2.0, 50, 1.0)])
    assert found["Fastest also the best"].startswith("Yes. Star")


def test_findings_with_ties_and_no_solves():
    found = findings_text([summary_row("A", 0.5, 0, None), summary_row("B", 0.5, 0, None)])
    assert found["Fastest"].startswith("A and B")
    assert "No model solved" in found["Closest to optimal"]
    assert key_findings([]) == []


def test_findings_when_nobody_solved():
    found = findings_text([summary_row("A", 0.2, 0, None), summary_row("B", 0.9, 0, None),
                           summary_row("C", 1.5, 0, None)])
    assert found["Highest solve rate"] == "No model solved any puzzle (0% for all)"
    assert found["Fastest also the best"] == "Can't tell yet, because no model solved a puzzle"


def test_findings_with_a_tie_above_zero():
    found = findings_text([summary_row("A", 0.2, 60, 1.0), summary_row("B", 0.9, 60, 0.5),
                           summary_row("C", 1.5, 60, 2.0), summary_row("D", 0.4, 20, 0.0)])
    assert found["Highest solve rate"] == "A, B and C (60% of puzzles solved)"
    assert found["Fastest also the best"].startswith("No. A was the fastest, but B solved just as many")


def fake_run(model, solved, moves, gap, latencies, total_time, invalid=0):
    return {"model": model, "solved": solved, "move_count": moves, "optimality_gap": gap,
            "latencies": latencies, "total_time": total_time, "invalid_outputs": invalid, "retries": invalid}


def test_summary_stats():
    results = [
        fake_run("A", True, 5, 0, [0.1, 0.3], 1.0),
        fake_run("A", True, 7, 2, [0.2], 2.0),
        fake_run("A", False, 30, None, [0.2], 6.0, invalid=3),
        fake_run("B", False, 30, None, [0.0], 0.5),
    ]
    a, b = summarize(results)
    assert a["model"] == "A" and a["runs"] == 3 and a["solved"] == 2
    assert a["success_rate"] == pytest.approx(66.67, abs=0.01)
    assert a["avg_latency"] == pytest.approx(0.2)
    assert a["avg_total_time"] == pytest.approx(3.0)
    assert a["avg_moves"] == pytest.approx(14.0)
    assert a["avg_optimality_gap"] == pytest.approx(1.0)  # solved runs only
    assert a["invalid_outputs"] == 3
    assert b["success_rate"] == 0 and b["avg_optimality_gap"] is None


# ---- Ollama mode: models take turns ----

class TrackedOracle(OracleMock):
    """Oracle that takes turns like an Ollama model, and records when it is loaded and unloaded."""

    runs_alone = True

    def __init__(self, name, log, fail_to_load=False):
        super().__init__(name, delay=0.02)
        self.log, self.fail_to_load = log, fail_to_load

    async def prepare(self):
        self.log.append(("load", self.name))
        if self.fail_to_load:
            raise AdapterError("model not found")

    async def release(self):
        self.log.append(("unload", self.name))


def test_ollama_models_run_one_by_one():
    log, events = [], []
    adapters = [TrackedOracle(f"m{i}", log) for i in range(3)]
    results = asyncio.run(run_race(PUZZLE, adapters, on_event=events.append, save=False))

    assert events[0]["type"] == "race_start" and events[0]["one_by_one"] is True
    turns = [(e["type"], e["model"]) for e in events if e["type"] in ("start", "finish")]
    assert turns == [("start", "m0"), ("finish", "m0"), ("start", "m1"), ("finish", "m1"),
                     ("start", "m2"), ("finish", "m2")]
    assert log == [("load", "m0"), ("unload", "m0"), ("load", "m1"), ("unload", "m1"),
                   ("load", "m2"), ("unload", "m2")]
    assert all(r["solved"] for r in results) and [r["lane"] for r in results] == [0, 1, 2]


def test_failed_load_warns_and_race_goes_on():
    log, events = [], []
    adapters = [TrackedOracle("broken", log, fail_to_load=True), TrackedOracle("fine", log)]
    results = asyncio.run(run_race(PUZZLE, adapters, on_event=events.append, save=False))
    warnings = [e for e in events if e["type"] == "warning"]
    assert len(warnings) == 1 and "broken" in warnings[0]["message"]
    assert len(results) == 2 and ("unload", "broken") in log


def test_mock_mode_still_runs_in_parallel(monkeypatch):
    monkeypatch.setattr(config, "MODE", "mock")
    events = []
    asyncio.run(run_race(PUZZLE, [OracleMock(f"m{i}", delay=0.02) for i in range(2)],
                         on_event=events.append, save=False))
    starts = [i for i, e in enumerate(events) if e["type"] == "start"]
    first_finish = next(i for i, e in enumerate(events) if e["type"] == "finish")
    assert events[0]["one_by_one"] is False and max(starts) < first_finish
