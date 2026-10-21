"""OpenRouter chat completions adapter (Claude Opus 5.5). The reply must be exactly {"move": "..."}."""

import json
import random
import re

from backend import api_guard, config
from backend.adapters.base import (
    INSTRUCTIONS, MOVE_DESCRIPTIONS, Adapter, MoveResult, describe_state,
    invalid_result, make_result, require_openrouter,
)
from backend.cube_engine import MOVES

REPLY_FORMAT = 'Reply with JSON only, exactly in the form {"move": "<one of the moves>"}.'

MOVE_SCHEMA = {
    "type": "object",
    "properties": {"move": {"type": "string", "enum": MOVES}},
    "required": ["move"],
    "additionalProperties": False,
}

# Thinking models (like DeepSeek-R1) put their reasoning in <think>...</think> first.
THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def build_messages(state: str, recent_moves: list[str]) -> list[dict]:
    question = {"state": describe_state(state, recent_moves), "moves": MOVE_DESCRIPTIONS}
    return [
        {"role": "system", "content": f"{INSTRUCTIONS} {REPLY_FORMAT}"},
        {"role": "user", "content": json.dumps(question, indent=2)},
    ]


def build_request(model_id: str, state: str, recent_moves: list[str]) -> dict:
    return {
        "model": model_id,
        "messages": build_messages(state, recent_moves),
        "max_tokens": config.CHAT_MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "next_move", "strict": True, "schema": MOVE_SCHEMA},
        },
    }


def parse_reply(content: str):
    """Read {"move": "..."} from a reply. A leading <think> block is skipped, the rest is checked strictly."""
    answer = THINK_BLOCK.sub("", content, count=1).strip()
    reply = json.loads(answer)
    if not isinstance(reply, dict) or set(reply) != {"move"}:
        raise ValueError(f"expected exactly {{\"move\": ...}}, got {answer!r}")
    return reply["move"]


def parse_response(data: dict):
    """(move, confidence) from a chat completion. Chat gives no confidence, so it is None."""
    return parse_reply(data["choices"][0]["message"]["content"]), None


def result_from_reply(data: dict, parse) -> MoveResult:
    # Raw response -> checked MoveResult (also used by the Ollama adapter).
    try:
        move, confidence = parse(data)
    except (KeyError, IndexError, TypeError) as e:
        return invalid_result(data, f"unexpected response format: {e!r}")
    except ValueError as e:  # also catches bad JSON
        return invalid_result(data, f"bad reply: {e}")
    return make_result(move, data, confidence)


def dry_run_response() -> dict:
    # Same shape as a real answer, so the dry run goes through the normal parser.
    content = json.dumps({"move": random.choice(MOVES)})
    return {"model": "dry-run", "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}}


class ChatAdapter(Adapter):
    kind = "LLM"
    mode = "openrouter"

    def __init__(self, name: str, model_id: str, budget):
        if not budget.dry_run:
            require_openrouter()
        super().__init__(name, model_id)
        self.budget = budget

    def retries(self) -> int:
        return config.OPENROUTER_RETRIES

    async def get_move(self, state: str, recent_moves: list[str]) -> MoveResult:
        request = build_request(self.model_id, state, recent_moves)
        data, info = await api_guard.request(self.budget, self.model_id, config.CHAT_URL, request, dry_run_response)
        result = result_from_reply(data, parse_response)
        result.api_info = info
        return result
