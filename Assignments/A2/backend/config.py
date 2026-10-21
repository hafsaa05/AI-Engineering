"""All settings in one place. Secrets and the mode are read from the .env file in the project folder."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
load_dotenv(PROJECT_ROOT / ".env")

# Which models race: "mock" (fake models), "ollama" (local models) or "openrouter" (real API models).
# MODE in .env is only the starting value; the GUI can switch it while the server runs (POST /api/mode).
# A key alone never turns on OpenRouter: it needs MODE=openrouter AND the key (see active_mode).
MODES = ["mock", "ollama", "openrouter"]
MODE = os.getenv("MODE", "ollama").strip().lower()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Game rules (same for every model)
MAX_MOVES = 20
MOVE_TIMEOUT = 60        # seconds per model call
MAX_RETRIES = 2          # extra attempts per move after a failed call or invalid output
RUN_TIME_LIMIT = 300     # hard limit per model run (every mode), then "OUT OF TIME"
HISTORY_LENGTH = 5       # how many recent moves the model is shown

REPLAY_SPEEDS = [1, 2.5, 5]

# ---- OpenRouter (teacher's rules: hard limits, persistent counter, cache) ----

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_RETRIES = 0           # an invalid answer just uses up a move
CHAT_MAX_TOKENS = 1024           # output cap for Opus; a {"move": ...} reply is tiny, and a big cap needs more credit
API_TOTAL_LIMIT = 65             # all models together, for the whole experiment
API_USAGE_FILE = PROJECT_ROOT / "data" / "api_usage.json"
API_CACHE_FILE = PROJECT_ROOT / "data" / "api_cache.json"
OPENROUTER_SET = "openrouter"    # real races are only allowed on this puzzle set

# max_requests = total requests this model may ever make (failed ones and retries count too).
# Luna and Clef IDs still need checking once we have a key.
MODELS = [
    {"name": "Jev 1.13", "model_id": "typesafe/jev-1.13", "api": "decisions", "max_requests": 20},
    {"name": "GPT-6 Luna Decisions", "model_id": "openai/gpt-6-luna-decisions", "api": "decisions", "max_requests": 20},
    {"name": "Cloudflare Clef", "model_id": "cloudflare/clef", "api": "decisions", "max_requests": 20},
    {"name": "Claude Opus 5.5", "model_id": "anthropic/claude-opus-5.5", "api": "chat", "max_requests": 5},
]

# ---- Local Ollama models (no API limits). A laptop holds one at a time, so they take turns. ----

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MOVE_TIMEOUT = 120   # seconds for a whole move, retries included (small local models are slower)
OLLAMA_LOAD_TIMEOUT = 600   # loading a big model into memory (not counted in the race)
OLLAMA_MAX_TOKENS = 4096    # same output cap for every model, enough for thinking models to finish
OLLAMA_CONTEXT = 8192       # room for the prompt (about 800 tokens) plus the 4096 output tokens
# api "decision": asked the same choice question as the OpenRouter decision models (Ollama /v1/systemone)
# api "chat": asked the same messages as Claude on OpenRouter (Ollama /api/chat)
# think: thinking off for the small fast models, on for deepseek-r1, our "slow thinker"
OLLAMA_MODELS = [
    {"name": "Clef Flash", "model_id": "clef-flash:latest", "kind": "DECISION MODEL", "api": "decision"},
    {"name": "Qwen3.5 0.8B", "model_id": "qwen3.5:0.8b", "kind": "LLM", "api": "chat", "think": False},
    {"name": "DeepSeek-R1 1.5B", "model_id": "deepseek-r1:1.5b", "kind": "LLM", "api": "chat", "think": True},
    {"name": "Gemma3 1B", "model_id": "gemma3:1b", "kind": "LLM", "api": "chat", "think": False},
]
# The order the local models take their turns in. Their panels stay in the order above.
OLLAMA_RUN_ORDER = ["Gemma3 1B", "Qwen3.5 0.8B", "DeepSeek-R1 1.5B", "Clef Flash"]

# Stand-ins used in mock mode
MOCK_MODELS = ["mock-random", "mock-oracle", "mock-noisy", "mock-broken"]


def active_mode() -> str:
    # OpenRouter is only used when asked for by name AND a key is set. Anything else falls back to mock.
    if MODE == "ollama":
        return "ollama"
    if MODE == "openrouter" and OPENROUTER_API_KEY:
        return "openrouter"
    return "mock"


def request_limit(model_id: str) -> int:
    # Unknown models get 0, so they can never make a request by accident.
    return next((m["max_requests"] for m in MODELS if m["model_id"] == model_id), 0)
