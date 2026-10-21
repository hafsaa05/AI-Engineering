"""OpenRouter Decisions API adapter (Jev, Luna, Clef): one choice question, the 18 moves are the options."""

import random

from backend import api_guard, config
from backend.adapters.base import (
    INSTRUCTIONS, MOVE_DESCRIPTIONS, Adapter, MoveResult, describe_state,
    invalid_result, make_result, require_openrouter,
)
from backend.cube_engine import MOVES

QUESTION_KEY = "next_move"


def build_request(model_id: str, state: str, recent_moves: list[str]) -> dict:
    return {
        "model": model_id,
        "state": describe_state(state, recent_moves),
        "questions": {
            QUESTION_KEY: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": MOVE_DESCRIPTIONS,
            }
        },
    }


def parse_response(data: dict):
    """(move, confidence) from a Decisions answer, e.g. {"answers": {"next_move": {"choice": "R"}}}."""
    answer = data["answers"][QUESTION_KEY]
    return answer["choice"], answer.get("confidence")


def result_from_response(data: dict) -> MoveResult:
    try:
        move, confidence = parse_response(data)
    except (KeyError, TypeError) as e:
        return invalid_result(data, f"unexpected response format: {e!r}")
    return make_result(move, data, confidence)


def dry_run_response() -> dict:
    # Same shape as a real answer, so the dry run goes through the normal parser.
    move = random.choice(MOVES)
    return {"model": "dry-run", "answers": {QUESTION_KEY: {"type": "choice", "choice": move, "confidence": 0.5}},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}}


class DecisionsAdapter(Adapter):
    kind = "DECISION MODEL"
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
        data, info = await api_guard.request(self.budget, self.model_id, config.DECISIONS_URL, request,
                                             dry_run_response)
        result = result_from_response(data)
        result.api_info = info
        return result
