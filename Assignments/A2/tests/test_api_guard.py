"""Tests for the teacher's OpenRouter rules. The HTTP call is always a fake, nothing leaves the machine."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from backend import api_guard, benchmark, config, main
from backend.adapters import load_adapters
from backend.adapters.base import AdapterError, MoveResult, post_json
from backend.api_guard import Budget, CallLimitReached
from backend.puzzles import get_puzzle

CLAUDE, JEV = "anthropic/claude-opus-5.5", "typesafe/jev-1.13"
URL = "https://example.invalid/never-used"


def fake_api(monkeypatch, response=None, fail=False):
    """Replace the real HTTP call. Returns a list that records every (pretend) call."""
    calls = []

    async def fake_post_json(url, payload, headers=None, timeout=None):
        calls.append(payload)
        if fail:
            raise AdapterError("HTTP 500: server error")
        return response or {"answers": {"next_move": {"choice": "U"}}, "usage": {"cost": 0.002}}

    monkeypatch.setattr(api_guard, "post_json", fake_post_json)
    return calls


def ask(budget, model_id, payload, fake=lambda: {"fake": True}):
    return asyncio.run(api_guard.request(budget, model_id, URL, payload, fake))


def saved_usage():
    return json.loads(config.API_USAGE_FILE.read_text(encoding="utf-8"))


# ---- No network ----

def test_tests_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        asyncio.run(post_json("https://openrouter.ai/api/v1/chat/completions", {}))


# ---- Persistent counter and limits ----

def test_requests_are_counted_and_saved_before_sending(monkeypatch):
    seen_counts = []

    async def fake_post_json(url, payload, headers=None, timeout=None):
        seen_counts.append(saved_usage()["total"])  # what is on disk while the request is "in flight"
        return {"usage": {"cost": 0.001}}

    monkeypatch.setattr(api_guard, "post_json", fake_post_json)
    _, info = ask(Budget(), JEV, {"n": 1})
    assert seen_counts == [1] and info["request_number"] == 1
    assert saved_usage()["per_model"][JEV] == 1


def test_counter_survives_a_restart(monkeypatch):
    fake_api(monkeypatch)
    ask(Budget(), JEV, {"n": 1})
    ask(Budget(), JEV, {"n": 2})  # a new Budget is like restarting the app
    assert Budget().usage["total"] == 2


def test_claude_stops_after_5_requests(monkeypatch):
    calls = fake_api(monkeypatch)
    budget = Budget()
    for n in range(5):
        ask(budget, CLAUDE, {"n": n})
    with pytest.raises(CallLimitReached, match="5 / 5"):
        ask(budget, CLAUDE, {"n": 99})
    assert len(calls) == 5


def test_decision_model_stops_after_20_requests(monkeypatch):
    calls = fake_api(monkeypatch)
    budget = Budget()
    for n in range(20):
        ask(budget, JEV, {"n": n})
    with pytest.raises(CallLimitReached, match="20 / 20"):
        ask(budget, JEV, {"n": 99})
    assert len(calls) == 20


def test_overall_limit_of_65_is_never_passed(monkeypatch):
    calls = fake_api(monkeypatch)
    config.API_USAGE_FILE.write_text(json.dumps({"total": 65, "per_model": {}}), encoding="utf-8")
    with pytest.raises(CallLimitReached, match="65 / 65"):
        ask(Budget(), JEV, {"n": 1})
    assert calls == []


def test_unknown_model_can_never_call(monkeypatch):
    calls = fake_api(monkeypatch)
    with pytest.raises(CallLimitReached):
        ask(Budget(), "someone/else", {"n": 1})
    assert calls == []


def test_broken_counter_file_blocks_everything(monkeypatch):
    calls = fake_api(monkeypatch)
    config.API_USAGE_FILE.write_text("not json", encoding="utf-8")
    with pytest.raises(CallLimitReached, match="unreadable"):
        ask(Budget(), JEV, {"n": 1})
    assert calls == []


def test_failed_requests_count_too(monkeypatch):
    fake_api(monkeypatch, fail=True)
    with pytest.raises(AdapterError) as error:
        ask(Budget(), JEV, {"n": 1})
    assert error.value.info["request_number"] == 1
    assert saved_usage()["total"] == 1


def test_cost_comes_from_the_usage_field(monkeypatch):
    fake_api(monkeypatch, response={"usage": {"cost": 0.0125}})
    _, info = ask(Budget(), JEV, {"n": 1})
    assert info["cost"] == 0.0125 and saved_usage()["total_cost"] == 0.0125


# ---- Cache ----

def test_cache_hit_reuses_the_answer_without_calling_or_counting(monkeypatch):
    calls = fake_api(monkeypatch)
    first, info1 = ask(Budget(), JEV, {"state": "same"})
    second, info2 = ask(Budget(), JEV, {"state": "same"})
    assert len(calls) == 1 and first == second
    assert info1["cache_hit"] is False and info1["api_latency"] is not None and info1["cache_latency"] is None
    assert info2["cache_hit"] is True and info2["cache_latency"] is not None and info2["api_latency"] is None
    assert info2["request_number"] is None and info2["cost"] == 0.0
    assert saved_usage()["total"] == 1 and saved_usage()["cache_hits"] == 1


def test_cache_key_depends_on_model_and_exact_body():
    assert api_guard.cache_key(JEV, {"a": 1, "b": 2}) == api_guard.cache_key(JEV, {"b": 2, "a": 1})
    assert api_guard.cache_key(JEV, {"a": 1}) != api_guard.cache_key(CLAUDE, {"a": 1})
    assert api_guard.cache_key(JEV, {"a": 1}) != api_guard.cache_key(JEV, {"a": 2})


def test_cache_hit_works_even_at_the_limit(monkeypatch):
    fake_api(monkeypatch)
    budget = Budget()
    for n in range(5):
        ask(budget, CLAUDE, {"n": n})
    _, info = ask(budget, CLAUDE, {"n": 0})  # already cached, so allowed and free
    assert info["cache_hit"] is True


# ---- Dry run ----

def test_dry_run_never_calls_and_never_saves(monkeypatch):
    calls = fake_api(monkeypatch)
    budget = Budget(dry_run=True)
    data, info = ask(budget, JEV, {"n": 1}, fake=lambda: {"pretend": "answer"})
    assert calls == [] and data == {"pretend": "answer"}
    assert info["dry_run"] is True and info["request_number"] == 1 and budget.usage["total"] == 1
    assert not config.API_USAGE_FILE.exists() and not config.API_CACHE_FILE.exists()


def test_dry_run_follows_the_same_limits():
    budget = Budget(dry_run=True)
    for n in range(5):
        ask(budget, CLAUDE, {"n": n})
    with pytest.raises(CallLimitReached):
        ask(budget, CLAUDE, {"n": 99})


def test_max_calls_next_race():
    assert api_guard.usage_summary()["max_calls_next_race"] == 65  # 20 + 20 + 20 + 5
    config.API_USAGE_FILE.write_text(json.dumps({"total": 5, "per_model": {CLAUDE: 5}}), encoding="utf-8")
    assert api_guard.usage_summary()["max_calls_next_race"] == 60


# ---- Whole races with fake OpenRouter answers ----

def openrouter_mode(monkeypatch):
    monkeypatch.setattr(config, "MODE", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")


def answer_with(pick_move):
    """Fake OpenRouter that answers in the right shape. pick_move(call_number) chooses the move."""
    calls = []

    async def fake_post_json(url, payload, headers=None, timeout=None):
        calls.append(payload)
        move = pick_move(len(calls))
        if "questions" in payload:
            return {"answers": {"next_move": {"choice": move, "confidence": 0.5}}, "usage": {"cost": 0.001}}
        return {"choices": [{"message": {"content": json.dumps({"move": move})}}], "usage": {"cost": 0.01}}
    return fake_post_json


def answer_always(move):
    return answer_with(lambda n: move)


# Never repeats a position, so every request is new (no cache hits) and R F U is never solved.
CYCLE = ["U", "D", "L", "B", "U2", "D2", "L2", "B2"]


def test_full_race_respects_every_limit(monkeypatch):
    openrouter_mode(monkeypatch)
    monkeypatch.setattr(api_guard, "post_json", answer_with(lambda n: CYCLE[n % len(CYCLE)]))
    puzzle = get_puzzle("openrouter-1")
    results = {r["model"]: r for r in asyncio.run(benchmark.run_race(puzzle, load_adapters(), save=False))}

    claude = results["Claude Opus 5.5"]
    assert claude["end_reason"] == "CALL LIMIT REACHED" and not claude["solved"]
    assert claude["move_count"] == 5 and claude["api_calls"] == 5
    decision_models = ("Jev 1.13", "GPT-6 Luna Decisions", "Cloudflare Clef")
    for name in decision_models:
        assert results[name]["end_reason"] == "RAN OUT OF MOVES" and results[name]["move_count"] == 20
    assert saved_usage()["total"] == sum(r["api_calls"] for r in results.values()) <= 65


def test_repeated_requests_come_from_the_cache(monkeypatch):
    openrouter_mode(monkeypatch)
    monkeypatch.setattr(api_guard, "post_json", answer_always("U"))  # U four times is back to the start
    puzzle = get_puzzle("openrouter-1")
    first = {r["model"]: r for r in asyncio.run(benchmark.run_race(puzzle, load_adapters(), save=False))}
    jev = first["Jev 1.13"]
    assert jev["move_count"] == 20 and jev["cache_hits"] > 0 and jev["api_calls"] + jev["cache_hits"] == 20

    # Racing the same puzzle again: every request is already cached, so nothing new is counted.
    used = saved_usage()["total"]
    again = {r["model"]: r for r in asyncio.run(benchmark.run_race(puzzle, load_adapters(), save=False))}
    assert saved_usage()["total"] == used
    assert again["Jev 1.13"]["api_calls"] == 0 and again["Jev 1.13"]["cache_hits"] == 20


def test_never_more_than_65_even_after_many_races(monkeypatch):
    openrouter_mode(monkeypatch)
    calls = []
    fake = answer_with(lambda n: CYCLE[n % len(CYCLE)])

    async def counting(url, payload, headers=None, timeout=None):
        calls.append(1)
        payload = {**payload, "race": len(calls)}  # make every request look new, so the cache never helps
        return await fake(url, payload, headers, timeout)

    monkeypatch.setattr(api_guard, "post_json", counting)
    for puzzle_id in ("openrouter-1", "ollama-1", "ollama-2", "ollama-3"):  # run on extra puzzles too
        asyncio.run(benchmark.run_race(get_puzzle(puzzle_id), load_adapters(), save=False))
    assert len(calls) == saved_usage()["total"] <= 65
    for m in config.MODELS:
        assert saved_usage()["per_model"].get(m["model_id"], 0) <= m["max_requests"]


def test_openrouter_invalid_answer_is_not_retried(monkeypatch):
    openrouter_mode(monkeypatch)
    monkeypatch.setattr(api_guard, "post_json", answer_always("R3"))  # never a valid move
    puzzle = get_puzzle("openrouter-1")
    results = asyncio.run(benchmark.run_race(puzzle, load_adapters(), save=False))
    for r in results:
        assert r["retries"] == 0 and r["moves"] == [None] * r["move_count"]
        assert all(len(entry["attempts"]) == 1 for entry in r["move_log"])


def test_move_log_has_the_api_fields(monkeypatch):
    openrouter_mode(monkeypatch)
    monkeypatch.setattr(api_guard, "post_json", answer_always("U"))
    result = asyncio.run(benchmark.run_race(get_puzzle("openrouter-1"), load_adapters(), save=False))[0]
    attempt = result["move_log"][0]["attempts"][0]
    for field in ("cache_hit", "api_latency", "cache_latency", "request_number", "cost"):
        assert field in attempt
    assert result["total_cost"] == pytest.approx(result["api_calls"] * 0.001)  # cache hits cost nothing


# ---- Hard per move timeout ----

class InstantAdapter:
    name = model_id = "instant"

    async def get_move(self, state, recent_moves):
        return MoveResult("U", "U", "U")


def test_answer_after_the_timeout_is_not_accepted(monkeypatch):
    clock = iter([0.0, 500.0])  # as if the computer slept for 500 seconds during the call
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    attempt = asyncio.run(benchmark.ask_once(InstantAdapter(), get_puzzle("ollama-1")["state"], [], timeout=120))
    assert attempt["move"] is None and attempt["call_failed"] and "after the 120s limit" in attempt["error"]


def test_keep_awake_is_safe_to_use():
    with benchmark.keep_awake():
        pass


# ---- Routes and the GUI flow ----

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "busy", None)
    with TestClient(main.app) as test_client:
        yield test_client


def first_event(client, path):
    with client.websocket_connect(path) as ws:
        return ws.receive_json()


def test_status_reports_usage(client):
    status = client.get("/api/status").json()
    assert status["api_calls_used"] == 0 and status["api_call_limit"] == 65
    assert status["cache_hits"] == 0 and status["total_cost"] == 0.0
    assert status["max_calls_next_race"] == 65 and status["openrouter_set"] == "openrouter"


def test_real_race_needs_confirmation(client, monkeypatch):
    openrouter_mode(monkeypatch)
    calls = fake_api(monkeypatch)
    event = first_event(client, "/ws/race/openrouter-1")
    assert event["type"] == "error" and "needs confirmation" in event["message"]
    event = first_event(client, "/ws/race/primary-1?confirm=yes")
    assert event["type"] == "error" and "only the openrouter puzzle" in event["message"]
    assert calls == []


def test_dry_run_from_the_gui_saves_nothing(client):
    events = []
    with client.websocket_connect("/ws/race/openrouter-1?dry_run=true") as ws:
        while True:
            events.append(ws.receive_json())
            if events[-1]["type"] in ("race_end", "error"):
                break
    assert events[-1]["type"] == "race_end" and events[-1]["api"]["dry_run"] is True
    assert events[-1]["api"]["calls"] > 0
    assert client.get("/api/results").json() == []          # not saved as a result
    assert not config.API_USAGE_FILE.exists()                 # the real counter is untouched
    assert client.get("/api/status").json()["api_calls_used"] == 0


def test_no_set_benchmarks_in_openrouter_mode(client, monkeypatch):
    openrouter_mode(monkeypatch)
    response = client.post("/api/benchmark/openrouter")
    assert response.status_code == 409 and "switched off" in response.json()["detail"]


def test_cli_refuses_openrouter_mode(monkeypatch):
    openrouter_mode(monkeypatch)
    with pytest.raises(SystemExit, match="only be started from the race screen"):
        asyncio.run(benchmark.main("openrouter-1"))
