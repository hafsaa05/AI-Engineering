from backend import api_guard, config
from backend.adapters.chat_adapter import ChatAdapter
from backend.adapters.decisions_adapter import DecisionsAdapter
from backend.adapters.mock_adapter import MOCK_ADAPTERS
from backend.adapters.ollama_adapter import OllamaChatAdapter, OllamaDecisionAdapter


def openrouter_adapters(dry_run: bool) -> list:
    # One shared budget for the race, so the four models count against the same 65.
    budget = api_guard.Budget(dry_run=dry_run)
    return [(DecisionsAdapter if m["api"] == "decisions" else ChatAdapter)(m["name"], m["model_id"], budget)
            for m in config.MODELS]


def load_adapters(dry_run: bool = False) -> list:
    """The four racers for the current mode (see MODE in config.py).

    dry_run=True gives the OpenRouter lineup in any mode, but it never makes an HTTP call.
    """
    if dry_run:
        return openrouter_adapters(dry_run=True)
    mode = config.active_mode()
    if mode == "ollama":
        return [OllamaDecisionAdapter(m["name"], m["model_id"], m["kind"]) if m["api"] == "decision"
                else OllamaChatAdapter(m["name"], m["model_id"], m["kind"], m.get("think"))
                for m in config.OLLAMA_MODELS]
    if mode == "openrouter":
        return openrouter_adapters(dry_run=False)
    return [MOCK_ADAPTERS[name](name) for name in config.MOCK_MODELS]
