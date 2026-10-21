"""FastAPI app: API routes, live race WebSocket, replays, and the frontend files.

Run from the project folder:
    uvicorn backend.main:app --reload
then open http://127.0.0.1:8000
"""

import asyncio

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.staticfiles import StaticFiles

from backend import api_guard, benchmark, config
from backend.adapters import load_adapters
from backend.adapters.base import AdapterError
from backend.puzzles import load_puzzles

app = FastAPI(title="Rubik's Cube Race")

# Only one race or benchmark may call the models at a time.
# `busy` says what is running, or is None when free.
busy = None
benchmark_task = None
benchmark_progress = {"running": False, "set": None, "done": 0, "total": 0, "current": None, "error": None}


def busy_message() -> str:
    return f"Busy: {busy} is already running. Please wait for it to finish."


def find_puzzle(puzzle_id: str) -> dict | None:
    return next((p for p in load_puzzles() if p["id"] == puzzle_id), None)


def strip_logs(result: dict) -> dict:
    # The full move_log (with raw API responses) is large, so the list view leaves it out.
    return {k: v for k, v in result.items() if k != "move_log"}


# ---- REST routes ----

@app.get("/api/puzzles")
def get_puzzles():
    return load_puzzles()


def current_lineup() -> list[dict]:
    if config.active_mode() == "openrouter":  # from config, so showing the page never touches the counter
        return [{"name": m["name"], "kind": "LLM" if m["api"] == "chat" else "DECISION MODEL"} for m in config.MODELS]
    return [{"name": a.name, "kind": a.kind} for a in load_adapters()]


@app.get("/api/status")
def get_status():
    lineup = current_lineup()
    try:
        usage = api_guard.usage_summary()  # api_calls_used, api_call_limit, cache_hits, total_cost, ...
    except api_guard.CallLimitReached as e:  # unreadable counter file
        usage = {"api_call_limit": config.API_TOTAL_LIMIT, "usage_error": str(e)}
    return {"busy": busy, "mode": config.active_mode(), "models": lineup,
            "max_moves": config.MAX_MOVES, "run_time_limit": config.RUN_TIME_LIMIT,
            "replay_speeds": config.REPLAY_SPEEDS,
            "openrouter_set": config.OPENROUTER_SET, "benchmark": benchmark_progress, **usage}


@app.post("/api/mode")
def change_mode(mode: str = Body(..., embed=True)):
    """Switch the mode from the GUI. Kept in memory only; .env is not touched."""
    mode = mode.strip().lower()
    if mode not in config.MODES:
        raise HTTPException(400, f"Unknown mode {mode!r}")
    if busy:
        raise HTTPException(409, f"The mode cannot change now. {busy_message()}")
    if mode == "openrouter" and not config.OPENROUTER_API_KEY:
        raise HTTPException(400, "OpenRouter mode needs OPENROUTER_API_KEY in .env (then restart the server).")
    config.MODE = mode
    return get_status()


def result_mode(result: dict) -> str:
    # Results saved before modes existed have no "mode" field.
    return result.get("mode") or ("mock" if result.get("kind") == "MOCK" else "openrouter")


def load_results_for_set(set_name: str | None, mode: str | None = None) -> list[dict]:
    """Saved runs, optionally only one set and only one mode (mock, ollama or openrouter)."""
    results = benchmark.load_results()
    if set_name:
        results = [r for r in results if r["puzzle_set"] == set_name]
    if mode:
        results = [r for r in results if result_mode(r) == mode]
    return results


@app.get("/api/results")
def get_results(set_name: str | None = Query(None, alias="set"), race_id: str | None = None):
    results = load_results_for_set(set_name)
    if race_id:
        results = [r for r in results if r["race_id"] == race_id]
    return [{**strip_logs(r), "mode": result_mode(r)} for r in results]


@app.get("/api/summary")
def get_summary(set_name: str | None = Query(None, alias="set"), mode: str | None = None):
    return benchmark.summarize(load_results_for_set(set_name, mode))


@app.get("/api/dashboard")
def get_dashboard(set_name: str = Query("primary", alias="set"), mode: str | None = None):
    """Everything the dashboard page needs in one call. Shows the current mode's results by default."""
    mode = mode or config.active_mode()
    results = load_results_for_set(set_name, mode)
    summary = benchmark.summarize(results)
    puzzles = [p for p in load_puzzles() if p["set"] == set_name]
    return {
        "set": set_name,
        "mode": mode,
        "models": [row["model"] for row in summary],
        "summary": summary,
        "puzzles": benchmark.puzzle_table(results, puzzles),
        "findings": benchmark.key_findings(summary),
    }


async def run_benchmark_set(set_name: str, puzzles: list[dict]) -> None:
    global busy
    benchmark_progress.update(running=True, set=set_name, done=0, total=len(puzzles), current=None, error=None)
    try:
        for puzzle in puzzles:
            benchmark_progress["current"] = puzzle["id"]
            await benchmark.run_race(puzzle)
            benchmark_progress["done"] += 1
    except AdapterError as e:  # for example Ollama is not running
        benchmark_progress["error"] = str(e)
    except Exception as e:
        benchmark_progress["error"] = repr(e)
    finally:
        benchmark_progress.update(running=False, current=None)
        busy = None


@app.post("/api/benchmark/{set_name}", status_code=202)
async def start_benchmark(set_name: str):
    global busy, benchmark_task
    puzzles = [p for p in load_puzzles() if p["set"] == set_name]
    if not puzzles:
        raise HTTPException(404, f"No puzzle set called {set_name!r}")
    if busy:
        raise HTTPException(409, busy_message())
    if config.active_mode() == "openrouter":
        # Teacher's rule: the real run is one race on the openrouter puzzle, started from the race screen.
        raise HTTPException(409, "In OpenRouter mode, whole-set benchmarks are switched off. "
                                 "Use the race screen to run the openrouter puzzle once.")

    # A full set can take a long time with real models, so it runs in the background.
    # The page can follow it with GET /api/status.
    busy = f"benchmark on the {set_name} set"
    benchmark_task = asyncio.create_task(run_benchmark_set(set_name, puzzles))
    return {"message": f"Started benchmark on {len(puzzles)} {set_name} puzzles", "total": len(puzzles)}


# ---- WebSockets ----

class LiveSender:
    """Sends events to the browser. If the browser goes away, the race still finishes and is saved."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connected = True

    async def __call__(self, event: dict) -> None:
        if not self.connected:
            return
        try:
            await self.websocket.send_json(event)
        except Exception:
            self.connected = False


async def send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})
    await websocket.close()


def openrouter_refusal(puzzle: dict, confirm: str) -> str | None:
    """Why a real OpenRouter race may not start, or None if it may. The GUI asks the user first."""
    if confirm != "yes":
        return "A real OpenRouter race needs confirmation. Start it with the Run Live Race button."
    if puzzle["set"] != config.OPENROUTER_SET:
        return f"In OpenRouter mode only the {config.OPENROUTER_SET} puzzle may be raced."
    return None


@app.websocket("/ws/race/{puzzle_id}")
async def race_socket(websocket: WebSocket, puzzle_id: str, dry_run: bool = False, confirm: str = ""):
    """Live race. dry_run=true runs the OpenRouter lineup without any HTTP call (any mode)."""
    global busy
    await websocket.accept()
    puzzle = find_puzzle(puzzle_id)
    if puzzle is None:
        return await send_error(websocket, f"No puzzle called {puzzle_id!r}")
    if busy:
        return await send_error(websocket, busy_message())
    if not dry_run and config.active_mode() == "openrouter":
        refusal = openrouter_refusal(puzzle, confirm)
        if refusal:
            return await send_error(websocket, refusal)

    busy = f"{'dry run' if dry_run else 'race'} on {puzzle_id}"
    sender = LiveSender(websocket)
    try:
        # A dry run is never saved, so it can never be mistaken for real results.
        await benchmark.run_race(puzzle, load_adapters(dry_run=dry_run), on_event=sender, save=not dry_run)
    except AdapterError as e:  # for example Ollama is not running
        await sender({"type": "error", "message": str(e)})
    except api_guard.CallLimitReached as e:
        await sender({"type": "error", "message": f"Call limit reached: {e}"})
    except Exception as e:
        await sender({"type": "error", "message": f"Race failed: {e!r}"})
    finally:
        busy = None
    if sender.connected:
        await websocket.close()


@app.websocket("/ws/replay/{race_id}")
async def replay_socket(websocket: WebSocket, race_id: str, speed: float = 1):
    # Replays only read saved files, so they do not need the busy lock.
    await websocket.accept()
    if speed not in config.REPLAY_SPEEDS:
        return await send_error(websocket, f"Speed must be one of {config.REPLAY_SPEEDS}")
    results = [r for r in benchmark.load_results() if r["race_id"] == race_id]
    if not results:
        return await send_error(websocket, f"No saved race called {race_id!r}")

    sender = LiveSender(websocket)
    await benchmark.replay_race(results, speed, sender)
    if sender.connected:
        await websocket.close()


# Mounted last so the /api and /ws routes above take priority.
app.mount("/", StaticFiles(directory=config.PROJECT_ROOT / "frontend", html=True), name="frontend")
