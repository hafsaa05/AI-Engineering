import time

import pytest
from fastapi.testclient import TestClient

from backend import config, main
from backend.adapters.mock_adapter import OracleMock
from backend.benchmark import save_result
from backend.cube_engine import SOLVED_STATE, get_state, is_solved


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODE", "mock")  # always mock models in tests
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "MAX_MOVES", 10)
    monkeypatch.setattr(OracleMock, "delay", 0.01)
    monkeypatch.setattr(main, "busy", None)
    with TestClient(main.app) as test_client:
        yield test_client


def receive_all(websocket) -> list[dict]:
    events = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event["type"] in ("race_end", "error"):
            return events


def wait_for_benchmark(client, limit=30):
    deadline = time.time() + limit
    while client.get("/api/status").json()["benchmark"]["running"]:
        assert time.time() < deadline, "benchmark did not finish"
        time.sleep(0.05)


def test_list_puzzles(client):
    puzzles = client.get("/api/puzzles").json()
    assert len(puzzles) == 22
    assert puzzles[0]["id"] == "warmup-1"


def test_status_shows_mock_mode(client):
    status = client.get("/api/status").json()
    assert status["mode"] == "mock" and status["busy"] is None
    assert status["replay_speeds"] == [1, 2.5, 5]
    assert [m["name"] for m in status["models"]] == config.MOCK_MODELS
    assert {m["kind"] for m in status["models"]} == {"MOCK"}


def test_change_mode_switches_the_lineup(client):
    status = client.post("/api/mode", json={"mode": "ollama"}).json()
    assert status["mode"] == "ollama"
    assert [m["name"] for m in status["models"]] == [m["name"] for m in config.OLLAMA_MODELS]


def test_openrouter_mode_needs_the_key(client, monkeypatch):
    response = client.post("/api/mode", json={"mode": "openrouter"})
    assert response.status_code == 400 and "OPENROUTER_API_KEY" in response.json()["detail"]
    assert client.get("/api/status").json()["mode"] == "mock"

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "fake-key-for-tests")
    status = client.post("/api/mode", json={"mode": "openrouter"}).json()
    assert status["mode"] == "openrouter"
    assert [m["name"] for m in status["models"]] == [m["name"] for m in config.MODELS]


def test_mode_cannot_change_while_busy(client, monkeypatch):
    monkeypatch.setattr(main, "busy", "race on primary-1")
    assert client.post("/api/mode", json={"mode": "ollama"}).status_code == 409
    assert client.post("/api/mode", json={"mode": "banana"}).status_code == 400


def test_one_page_has_race_and_results(client):
    page = client.get("/").text
    assert "Hafsa Rashid · 23K-0064" in page
    assert 'id="race-grid"' in page and 'id="summary-body"' in page  # results are on the race page
    assert client.get("/dashboard.html").status_code == 404


def test_live_race_sends_events_and_saves(client):
    with client.websocket_connect("/ws/race/primary-3") as ws:
        events = receive_all(ws)

    assert events[0]["type"] == "race_start" and events[-1]["type"] == "race_end"
    assert [m["name"] for m in events[0]["models"]] == config.MOCK_MODELS
    for model in config.MOCK_MODELS:
        own = [e["type"] for e in events if e.get("model") == model]
        assert own[0] == "start" and own[-1] == "finish"
        assert set(own[1:-1]) == {"move"}

    ranking = events[-1]["ranking"]
    assert sorted(row["rank"] for row in ranking) == [1, 2, 3, 4]
    solved = [row["rank"] for row in ranking if row["end_reason"] == "SOLVED"]
    unsolved = [row["rank"] for row in ranking if row["end_reason"] != "SOLVED"]
    assert max(solved) < min(unsolved)  # random cannot solve in 10 moves, so it is behind the solvers

    saved = client.get("/api/results", params={"race_id": events[0]["race_id"]}).json()
    assert len(saved) == 4 and "move_log" not in saved[0]
    assert client.get("/api/status").json()["busy"] is None


def test_unknown_puzzle(client):
    with client.websocket_connect("/ws/race/nope") as ws:
        event = ws.receive_json()
    assert event["type"] == "error" and "nope" in event["message"]


def test_only_one_race_at_a_time(client, monkeypatch):
    monkeypatch.setattr(main, "busy", "race on primary-1")
    with client.websocket_connect("/ws/race/primary-2") as ws:
        event = ws.receive_json()
    assert event["type"] == "error" and "already running" in event["message"]

    response = client.post("/api/benchmark/warmup")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_benchmark_set_runs_in_background(client):
    response = client.post("/api/benchmark/warmup")
    assert response.status_code == 202 and response.json()["total"] == 3
    assert client.post("/api/benchmark/warmup").status_code == 409  # still running

    wait_for_benchmark(client)
    progress = client.get("/api/status").json()["benchmark"]
    assert progress["done"] == 3 and progress["error"] is None

    results = client.get("/api/results", params={"set": "warmup"}).json()
    assert len(results) == 12
    summary = client.get("/api/summary", params={"set": "warmup"}).json()
    assert [row["model"] for row in summary] == config.MOCK_MODELS
    assert client.get("/api/summary", params={"set": "primary"}).json() == []


def test_dashboard_data(client):
    with client.websocket_connect("/ws/race/warmup-1") as ws:
        receive_all(ws)

    data = client.get("/api/dashboard", params={"set": "warmup"}).json()
    assert data["mode"] == "mock"  # the current mode is shown by default
    assert data["models"] == config.MOCK_MODELS
    assert [row["puzzle_id"] for row in data["puzzles"]] == ["warmup-1", "warmup-2", "warmup-3"]
    assert set(data["puzzles"][0]["cells"]) == set(config.MOCK_MODELS)
    assert data["puzzles"][1]["cells"] == {}
    assert [f["label"] for f in data["findings"]][0] == "Fastest"

    for other in ("ollama", "openrouter"):  # mock runs never show up under the other modes
        data = client.get("/api/dashboard", params={"set": "warmup", "mode": other}).json()
        assert data["summary"] == [] and data["findings"] == []


def test_unknown_benchmark_set(client):
    assert client.post("/api/benchmark/nope").status_code == 404


# ---- Replay ----

def fake_saved_race(model, latencies, moves):
    log = [{"number": i + 1, "move": m, "valid": m is not None, "latency": lat, "retries": 0}
           for i, (m, lat) in enumerate(zip(moves, latencies))]
    return {
        "race_id": "old-race", "model": model, "model_id": model, "kind": "MOCK", "puzzle_id": "warmup-1",
        "puzzle_set": "warmup", "scramble": ["R", "U"], "start_state": get_state(["R", "U"]),
        "move_log": log, "move_count": len(log), "latencies": latencies, "total_time": sum(latencies),
        "final_state": SOLVED_STATE,
        "solved": True, "end_reason": "SOLVED", "optimality_gap": len(log) - 2,
        "retries": 0, "invalid_outputs": 0,
    }


def save_fake_race():
    save_result(fake_saved_race("fast", [0.1, 0.1], ["U'", "R'"]))
    save_result(fake_saved_race("slow", [0.3, 0.2, 0.25], ["U'", None, "R'"]))  # one wasted move


def replay(client, speed):
    start_time = time.perf_counter()
    with client.websocket_connect(f"/ws/replay/old-race?speed={speed}") as ws:
        events = receive_all(ws)
    return events, time.perf_counter() - start_time


def test_replay_uses_recorded_timing(client):
    save_fake_race()
    events, took = replay(client, 1)
    assert 0.7 <= took < 1.5  # the slow model took 0.75s in total

    assert events[0]["type"] == "race_start" and events[0]["replay"] is True
    moves = [e for e in events if e["type"] == "move"]
    assert [e["model"] for e in moves] == ["fast", "fast", "slow", "slow", "slow"]  # in time order
    slow_states = [e["state"] for e in moves if e["model"] == "slow"]
    assert slow_states[0] == slow_states[1]  # invalid move leaves the cube alone
    assert is_solved(slow_states[-1])

    ranking = {row["model"]: row["rank"] for row in events[-1]["ranking"]}
    assert ranking == {"fast": 1, "slow": 2}


def test_replay_faster_speed(client):
    save_fake_race()
    events, took = replay(client, 5)
    assert took < 0.5  # 0.75s / 5 = 0.15s
    assert events[-1]["type"] == "race_end"


def test_replay_errors(client):
    save_fake_race()
    with client.websocket_connect("/ws/replay/old-race?speed=3") as ws:
        assert "Speed must be one of" in ws.receive_json()["message"]
    with client.websocket_connect("/ws/replay/missing?speed=1") as ws:
        assert "No saved race" in ws.receive_json()["message"]
