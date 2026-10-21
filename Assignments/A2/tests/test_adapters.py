import asyncio
import json

import pytest

from backend import api_guard, config
from backend.adapters import chat_adapter, decisions_adapter, load_adapters, ollama_adapter
from backend.adapters.base import AdapterError
from backend.adapters.chat_adapter import ChatAdapter
from backend.adapters.decisions_adapter import DecisionsAdapter
from backend.adapters.mock_adapter import BrokenMock, NoisyMock, OracleMock, RandomMock
from backend.adapters.ollama_adapter import OllamaChatAdapter, OllamaDecisionAdapter
from backend.cube_engine import MOVES, apply_move, get_state, is_solved
from backend.solver_bfs import optimal_distance

STATE = get_state(["F", "D", "R'", "B2", "L'"])
HISTORY = ["U", "R", "F", "D", "L", "B", "U2"]


def ask(adapter, state=STATE, recent=()):
    return asyncio.run(adapter.get_move(state, list(recent)))


# ---- Mock models ----

def test_random_mock_gives_valid_moves():
    adapter = RandomMock("mock-random", seed=1)
    results = [ask(adapter) for _ in range(30)]
    assert all(r.valid and r.move in MOVES for r in results)
    assert len({r.move for r in results}) > 5


def test_oracle_mock_solves_optimally():
    adapter = OracleMock("mock-oracle", delay=0)
    state, moves = STATE, 0
    while not is_solved(state):
        state = apply_move(state, ask(adapter, state).move)
        moves += 1
    assert moves == optimal_distance(STATE)


def test_noisy_mock_is_valid_but_not_perfect():
    adapter = NoisyMock("mock-noisy", delay=0, seed=2)
    oracle_move = ask(OracleMock("mock-oracle", delay=0)).move
    results = [ask(adapter) for _ in range(40)]
    assert all(r.valid for r in results)
    assert any(r.move != oracle_move for r in results)


def test_broken_mock_output_is_flagged_not_fixed():
    adapter = BrokenMock("mock-broken", delay=0, seed=3)
    results = [ask(adapter) for _ in range(40)]
    bad = [r for r in results if not r.valid]
    assert bad and len(bad) < len(results)
    for r in bad:
        assert r.move is None and r.error and r.output not in MOVES


# ---- Switching real models on and off ----

def set_mode(monkeypatch, mode, key=""):
    monkeypatch.setattr(config, "MODE", mode)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", key)


def openrouter_is_blocked():
    for adapter_class in (DecisionsAdapter, ChatAdapter):
        with pytest.raises(RuntimeError):
            adapter_class("test", "test/model", api_guard.Budget())
    return True


def test_mock_mode_by_default(monkeypatch):
    set_mode(monkeypatch, "mock")
    assert [a.name for a in load_adapters()] == config.MOCK_MODELS
    assert openrouter_is_blocked()


def test_a_key_alone_does_not_turn_on_openrouter(monkeypatch):
    set_mode(monkeypatch, "mock", key="test-key")
    assert config.active_mode() == "mock"
    assert openrouter_is_blocked()


def test_ollama_mode_never_uses_openrouter(monkeypatch):
    set_mode(monkeypatch, "ollama", key="test-key")  # even with a key in .env
    adapters = load_adapters()
    assert [type(a) for a in adapters] == [OllamaDecisionAdapter] + [OllamaChatAdapter] * 3
    assert [a.model_id for a in adapters] == ["clef-flash:latest", "qwen3.5:0.8b", "deepseek-r1:1.5b", "gemma3:1b"]
    assert [a.kind for a in adapters] == ["DECISION MODEL", "LLM", "LLM", "LLM"]
    assert "openrouter" not in config.OLLAMA_URL
    assert openrouter_is_blocked()
    assert all(a.timeout() == 120 and a.runs_alone and a.retries() == 2 for a in adapters)


def test_openrouter_needs_mode_and_key(monkeypatch):
    set_mode(monkeypatch, "openrouter", key="")
    assert config.active_mode() == "mock"  # no key, so it falls back to mock

    set_mode(monkeypatch, "openrouter", key="test-key")
    adapters = load_adapters()
    assert [a.model_id for a in adapters] == [m["model_id"] for m in config.MODELS]
    assert [type(a) for a in adapters] == [DecisionsAdapter] * 3 + [ChatAdapter]
    assert all(a.timeout() == 60 and not a.runs_alone and a.retries() == 0 for a in adapters)
    assert len({id(a.budget) for a in adapters}) == 1  # one shared budget for the race


def test_dry_run_works_in_any_mode_without_a_key(monkeypatch):
    set_mode(monkeypatch, "mock")
    adapters = load_adapters(dry_run=True)
    assert [type(a) for a in adapters] == [DecisionsAdapter] * 3 + [ChatAdapter]
    assert all(a.budget.dry_run for a in adapters)


# ---- Requests are the same for every model ----

def test_decisions_requests_differ_only_by_model():
    requests = [decisions_adapter.build_request(m["model_id"], STATE, HISTORY)
                for m in config.MODELS if m["api"] == "decisions"]
    for request in requests:
        request.pop("model")
    assert requests[0] == requests[1] == requests[2]

    question = requests[0]["questions"]["next_move"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == MOVES


def test_requests_show_last_five_moves():
    decisions = decisions_adapter.build_request("x", STATE, HISTORY)
    chat = chat_adapter.build_request("x", STATE, HISTORY)
    chat_question = json.loads(chat["messages"][1]["content"])
    assert decisions["state"]["recent_moves"] == HISTORY[-5:]
    assert chat_question["state"] == decisions["state"]
    assert chat_question["moves"] == decisions["questions"]["next_move"]["criteria"]


# ---- Parsing fake API responses ----

def fake_decisions_answer(choice, confidence=0.84):
    return {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"next_move": {"type": "choice", "choice": choice, "confidence": confidence,
                                  "probabilities": {choice: confidence}}},
        "usage": {"input_tokens": 500, "output_tokens": 5},
    }


def fake_chat_answer(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


MODEL_IDS = {DecisionsAdapter: "typesafe/jev-1.13", ChatAdapter: "anthropic/claude-opus-5.5"}


def run_with_fake_response(monkeypatch, module, adapter_class, response):
    # The fake replaces the HTTP call, so nothing leaves the machine.
    async def fake_post_json(url, payload, headers=None, timeout=None):
        return response

    set_mode(monkeypatch, "openrouter", key="test-key")
    monkeypatch.setattr(api_guard, "post_json", fake_post_json)
    return ask(adapter_class("test", MODEL_IDS[adapter_class], api_guard.Budget()))


def test_decisions_good_response(monkeypatch):
    result = run_with_fake_response(monkeypatch, decisions_adapter, DecisionsAdapter,
                                    fake_decisions_answer("R'"))
    assert result.valid and result.move == "R'" and result.confidence == 0.84
    assert result.raw_response["model"] == "typesafe/jev-1.13-20260917"
    assert result.api_info["request_number"] == 1 and result.api_info["cache_hit"] is False


@pytest.mark.parametrize("response", [
    fake_decisions_answer("R3"),                     # not a real move
    fake_decisions_answer("r"),                      # wrong case is not fixed
    {"answers": {}, "usage": {}},                    # question missing
    {"error": {"code": 400, "message": "bad"}},      # no answers at all
])
def test_decisions_bad_responses(monkeypatch, response):
    result = run_with_fake_response(monkeypatch, decisions_adapter, DecisionsAdapter, response)
    assert not result.valid and result.error
    assert result.raw_response == response


def test_chat_good_response(monkeypatch):
    result = run_with_fake_response(monkeypatch, chat_adapter, ChatAdapter,
                                    fake_chat_answer('{"move": "U2"}'))
    assert result.valid and result.move == "U2" and result.confidence is None


@pytest.mark.parametrize("content", [
    "U2",                                  # not JSON
    '```json\n{"move": "U2"}\n```',        # wrapped in a code block
    '{"move": "U2", "reason": "because"}', # extra keys
    '{"move": " U2"}',                     # extra space is not stripped
    '["U2"]',                              # wrong JSON type
    None,                                  # empty reply
])
def test_chat_bad_responses(monkeypatch, content):
    result = run_with_fake_response(monkeypatch, chat_adapter, ChatAdapter, fake_chat_answer(content))
    assert not result.valid and result.error


def test_chat_request_uses_structured_output():
    request = chat_adapter.build_request("anthropic/claude-opus-5.5", STATE, [])
    schema = request["response_format"]["json_schema"]["schema"]
    assert request["response_format"]["type"] == "json_schema"
    assert schema["properties"]["move"]["enum"] == MOVES
    assert request["max_tokens"] == 1024  # never more than the key's credit allows


# ---- Ollama (local models) ----

def ollama_answer(content, thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    return {"model": "deepseek-r1:1.5b", "message": message, "done": True}


def fake_ollama(monkeypatch, response):
    calls = []

    async def fake_post_json(url, payload, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return response

    set_mode(monkeypatch, "ollama")
    monkeypatch.setattr(ollama_adapter, "post_json", fake_post_json)
    return calls


def ask_ollama(monkeypatch, response):
    calls = fake_ollama(monkeypatch, response)
    return ask(OllamaChatAdapter("DeepSeek-R1 1.5B", "deepseek-r1:1.5b", "LLM")), calls


def test_ollama_request_matches_chat_wording():
    request = ollama_adapter.build_chat_request("gemma3:1b", STATE, HISTORY)
    assert request["model"] == "gemma3:1b" and request["stream"] is False
    assert request["messages"] == chat_adapter.build_request("x", STATE, HISTORY)["messages"]
    assert request["format"] == chat_adapter.MOVE_SCHEMA
    assert request["options"] == {"num_predict": 4096, "num_ctx": config.OLLAMA_CONTEXT}


def test_ollama_thinking_switch_and_time_limits(monkeypatch):
    assert "think" not in ollama_adapter.build_chat_request("gemma3:1b", STATE, HISTORY)
    assert ollama_adapter.build_chat_request("gemma3:1b", STATE, HISTORY, think=False)["think"] is False

    set_mode(monkeypatch, "ollama")
    adapters = {a.name: a for a in load_adapters()}
    assert adapters["Qwen3.5 0.8B"].think is False and adapters["Gemma3 1B"].think is False
    assert adapters["DeepSeek-R1 1.5B"].think is True
    clef = adapters["Clef Flash"]
    assert clef.timeout() == 120 and clef.timeout_covers_retries


def test_ollama_good_reply_goes_to_local_server(monkeypatch):
    result, calls = ask_ollama(monkeypatch, ollama_answer('{"move": "F2"}'))
    assert result.valid and result.move == "F2"
    assert calls[0]["url"] == f"{config.OLLAMA_URL}/api/chat"
    assert calls[0]["headers"] is None and calls[0]["timeout"] == 120  # no API key is sent


@pytest.mark.parametrize("content, thinking", [
    ('<think>\nThe front face looks wrong, so F2 might help.\n</think>\n\n{"move": "F2"}', None),  # old style
    ('<think></think>{"move": "F2"}', None),                                                      # empty thinking
    ('{"move": "F2"}', "Let me think about the front face..."),                                   # new style
])
def test_thinking_block_is_skipped(monkeypatch, content, thinking):
    result, _ = ask_ollama(monkeypatch, ollama_answer(content, thinking))
    assert result.valid and result.move == "F2"


@pytest.mark.parametrize("content", [
    '<think>I will play F2</think>\nF2',              # answer after thinking is not JSON
    '<think>still thinking about {"move": "F2"}',     # thinking never finished
    '{"move": "F2"} <think>extra</think>',            # thinking is only skipped at the start
    '<think>ok</think>{"move": "f2"}',                # wrong case is still not fixed
])
def test_bad_ollama_replies_are_invalid(monkeypatch, content):
    result, _ = ask_ollama(monkeypatch, ollama_answer(content))
    assert not result.valid and result.error


def fake_decision(choice):
    return {"model": "clef-flash:latest", "usage": {"input_tokens": 900, "output_tokens": 0},
            "answers": {"next_move": {"type": "choice", "choice": choice, "confidence": 0.3}}}


def test_ollama_decision_model_gets_the_openrouter_question(monkeypatch):
    calls = fake_ollama(monkeypatch, fake_decision("R2"))
    result = ask(OllamaDecisionAdapter("Clef Flash", "clef-flash:latest", "DECISION MODEL"), recent=HISTORY)
    assert result.valid and result.move == "R2" and result.confidence == 0.3
    assert calls[0]["url"] == f"{config.OLLAMA_URL}/v1/systemone"
    # same request as the OpenRouter decision models, only the model name differs
    assert calls[0]["payload"] == decisions_adapter.build_request("clef-flash:latest", STATE, HISTORY)


def test_ollama_decision_model_bad_choice_is_invalid(monkeypatch):
    fake_ollama(monkeypatch, fake_decision("r2"))
    result = ask(OllamaDecisionAdapter("Clef Flash", "clef-flash:latest", "DECISION MODEL"))
    assert not result.valid and result.output == "r2"


def check_ready_with(monkeypatch, tags=None):
    async def fake_get_json(url, timeout=5):
        if tags is None:
            raise AdapterError("could not connect")
        return {"models": [{"name": t} for t in tags]}

    monkeypatch.setattr(ollama_adapter, "get_json", fake_get_json)
    adapter = OllamaChatAdapter("Gemma3 1B", "gemma3:1b", "LLM")
    return asyncio.run(adapter.check_ready())


def test_ollama_not_running_gives_a_clear_message(monkeypatch):
    with pytest.raises(AdapterError, match="Ollama is not running"):
        check_ready_with(monkeypatch, tags=None)


def test_missing_ollama_model_says_how_to_pull_it(monkeypatch):
    with pytest.raises(AdapterError, match="ollama pull gemma3:1b"):
        check_ready_with(monkeypatch, tags=["qwen3.5:0.8b"])
    check_ready_with(monkeypatch, tags=["gemma3:1b"])  # installed: no error
