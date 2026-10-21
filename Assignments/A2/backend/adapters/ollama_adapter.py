"""Local Ollama adapters. Nothing here talks to OpenRouter: every request goes to the local Ollama server.

To keep the comparison fair, the local models get exactly the same requests as the OpenRouter ones:
- the LLMs get the same messages as Claude on OpenRouter (Ollama's /api/chat)
- the decision model gets the same choice question as the OpenRouter decision models
  (Ollama's decision endpoint /v1/systemone, which uses the same request and answer format)
"""

from backend import config
from backend.adapters import decisions_adapter
from backend.adapters.base import Adapter, AdapterError, MoveResult, get_json, post_json
from backend.adapters.chat_adapter import MOVE_SCHEMA, build_messages, parse_reply, result_from_reply

DECISION_PATH = "/v1/systemone"


def build_chat_request(model_id: str, state: str, recent_moves: list[str], think: bool | None = None) -> dict:
    request = {
        "model": model_id,
        "messages": build_messages(state, recent_moves),
        "format": MOVE_SCHEMA,  # Ollama only lets the model write {"move": one of the 18 moves}
        "stream": False,
        "options": {"num_predict": config.OLLAMA_MAX_TOKENS, "num_ctx": config.OLLAMA_CONTEXT},
    }
    if think is not None:
        request["think"] = think
    return request


def parse_chat_response(data: dict):
    """Pull (move, confidence) out of an Ollama /api/chat reply. Update here if the API changes.

    Expected shape: {"message": {"role": "assistant", "content": "{\\"move\\": \\"R\\"}", "thinking": "..."}}
    Thinking models put their reasoning in "thinking" (newer Ollama) or in a <think> block at the
    start of "content" (older Ollama). Only the final answer is read. No confidence is given.
    """
    return parse_reply(data["message"]["content"]), None


class OllamaAdapter(Adapter):
    """What both local adapters share: taking turns, longer time limits, and the readiness check."""

    mode = "ollama"
    runs_alone = True
    timeout_covers_retries = True  # 120s per move in total, so retries can not stretch one move to 6 minutes

    def __init__(self, name: str, model_id: str, kind: str):
        super().__init__(name, model_id)
        self.kind = kind

    def timeout(self) -> float:
        return config.OLLAMA_MOVE_TIMEOUT


    async def check_ready(self) -> None:
        # A clear message instead of a crash when Ollama is not running or the model is missing.
        try:
            tags = await get_json(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        except AdapterError:
            raise AdapterError(f"Ollama is not running at {config.OLLAMA_URL}. "
                               "Start the Ollama app (or run 'ollama serve') and try again.")
        installed = {m.get("name") for m in tags.get("models", [])}
        if self.model_id not in installed:
            raise AdapterError(f"The Ollama model {self.model_id} is not installed. Run: ollama pull {self.model_id}")

    async def post(self, path: str, payload: dict, timeout: float) -> dict:
        return await post_json(f"{config.OLLAMA_URL}{path}", payload, timeout=timeout)


class OllamaChatAdapter(OllamaAdapter):
    """The local LLMs (qwen3.5, deepseek-r1, gemma3)."""

    def __init__(self, name: str, model_id: str, kind: str, think: bool | None = None):
        super().__init__(name, model_id, kind)
        self.think = think

    async def get_move(self, state: str, recent_moves: list[str]) -> MoveResult:
        request = build_chat_request(self.model_id, state, recent_moves, self.think)
        data = await self.post("/api/chat", request, self.timeout())
        return result_from_reply(data, parse_chat_response)

    async def prepare(self) -> None:
        # Load the model before its run starts, so loading time is not counted as move latency.
        await self.post("/api/generate", {"model": self.model_id, "keep_alive": "30m"}, config.OLLAMA_LOAD_TIMEOUT)

    async def release(self) -> None:
        # Unload it again so the next model has the memory to itself.
        await self.post("/api/generate", {"model": self.model_id, "keep_alive": 0}, config.OLLAMA_LOAD_TIMEOUT)


class OllamaDecisionAdapter(OllamaAdapter):
    """The local decision model (clef-flash). It cannot chat, it only answers choice questions."""

    async def get_move(self, state: str, recent_moves: list[str]) -> MoveResult:
        request = decisions_adapter.build_request(self.model_id, state, recent_moves)
        data = await self.post(DECISION_PATH, request, self.timeout())
        return decisions_adapter.result_from_response(data)

    async def warm_question(self, keep_alive) -> None:
        # A tiny question, only used to load or unload the model (decision models have no /api/generate).
        payload = {"model": self.model_id, "keep_alive": keep_alive, "state": "warm up",
                   "questions": {"q": {"type": "choice", "instructions": "Pick one", "criteria": {"a": "A", "b": "B"}}}}
        await self.post(DECISION_PATH, payload, config.OLLAMA_LOAD_TIMEOUT)

    async def prepare(self) -> None:
        await self.warm_question("30m")

    async def release(self) -> None:
        await self.warm_question(0)
