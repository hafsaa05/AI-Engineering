"""Shared adapter pieces: the result type, the prompt wording and the HTTP call."""

from dataclasses import dataclass
from typing import Any

import httpx

from backend import config
from backend.cube_engine import MOVES, SOLVED_STATE, is_valid_move


@dataclass
class MoveResult:
    output: Any               # exactly what the model answered, before validation
    move: str | None          # the move if the output was valid, otherwise None
    raw_response: Any         # full response, kept for the logs
    confidence: float | None = None
    error: str | None = None  # why the output was rejected
    api_info: dict | None = None  # OpenRouter only: cache hit, latencies, request number, cost

    @property
    def valid(self) -> bool:
        return self.move is not None


class AdapterError(Exception):
    """The call itself failed (network, timeout, HTTP error), so there is no answer at all."""

    info = None  # OpenRouter only: the logging fields of the request that failed


class Adapter:
    kind = ""           # badge shown in the GUI: "DECISION MODEL", "LLM" or "MOCK"
    mode = ""           # "mock", "ollama" or "openrouter", saved with each result
    runs_alone = False  # True = models take turns instead of racing at the same time (Ollama)

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    async def get_move(self, state: str, recent_moves: list[str]) -> MoveResult:
        raise NotImplementedError

    # Read from config each time, so a test can change them.
    def timeout(self) -> float:
        return config.MOVE_TIMEOUT

    def retries(self) -> int:
        return config.MAX_RETRIES

    # True (Ollama): one timeout covers the whole move, retries included.
    timeout_covers_retries = False

    # Hooks around a run, outside the timed part (only Ollama uses them).
    async def check_ready(self) -> None:
        pass

    async def prepare(self) -> None:
        pass

    async def release(self) -> None:
        pass


def make_result(output, raw_response, confidence=None) -> MoveResult:
    # Only an exact match counts, we never fix the output.
    if is_valid_move(output):
        return MoveResult(output, output, raw_response, confidence)
    return MoveResult(output, None, raw_response, confidence, f"invalid move {output!r}")


def invalid_result(raw_response, error: str) -> MoveResult:
    return MoveResult(None, None, raw_response, None, error)


# ---- Prompt wording, shared by every real model so the test is fair ----

INSTRUCTIONS = (
    "You are solving a 3x3 Rubik's Cube one move at a time. "
    "Look at the current cube state and the recent moves, then choose the single next move "
    "that brings the cube closest to solved. A cube is solved when every face is one color."
)

CUBE_FORMAT = (
    "54 letters: 9 stickers for each face in the order U, R, F, D, L, B. "
    "Each letter names the face whose center has that sticker's color. "
    "Each face is read row by row, left to right, top to bottom, looking straight at that face: "
    "U is seen with B at the top, D with F at the top, and R, F, L, B with U at the top."
)

FACE_NAMES = {"U": "Up (top)", "R": "Right", "F": "Front", "D": "Down (bottom)", "L": "Left", "B": "Back"}
TURN_NAMES = {
    "": "a quarter turn (90 degrees) clockwise",
    "'": "a quarter turn (90 degrees) counterclockwise",
    "2": "a half turn (180 degrees)",
}

# One plain sentence per move, e.g. "R'": "Turn the Right face a quarter turn ... counterclockwise ..."
MOVE_DESCRIPTIONS = {
    move: f"Turn the {FACE_NAMES[move[0]]} face {TURN_NAMES[move[1:]]}, as seen looking straight at that face."
    for move in MOVES
}


def describe_state(state: str, recent_moves: list[str]) -> dict:
    return {
        "cube": state,
        "cube_format": CUBE_FORMAT,
        "solved_cube": SOLVED_STATE,
        "recent_moves": list(recent_moves[-config.HISTORY_LENGTH:]),  # oldest first
    }


# ---- HTTP, shared by the OpenRouter and Ollama adapters ----

def require_openrouter() -> None:
    # Safety switch: no OpenRouter adapter unless MODE=openrouter and a key is set.
    if config.active_mode() != "openrouter":
        raise RuntimeError("OpenRouter is switched off (needs MODE=openrouter and OPENROUTER_API_KEY in .env).")


def openrouter_headers() -> dict:
    return {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}


async def post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 60) -> dict:
    return await send("POST", url, payload, headers, timeout)


async def get_json(url: str, timeout: float = 5) -> dict:
    return await send("GET", url, None, None, timeout)


async def send(method: str, url: str, payload, headers, timeout: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, json=payload, headers=headers)
    except httpx.TimeoutException:
        raise AdapterError(f"timed out after {timeout}s")
    except httpx.ConnectError:
        raise AdapterError(f"could not connect to {url}")
    except httpx.HTTPError as e:
        raise AdapterError(f"request failed: {e!r}")

    if response.status_code != 200:
        raise AdapterError(f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError:
        raise AdapterError(f"response was not JSON: {response.text[:300]}")
