"""Safety settings for every test (teacher's rule: tests never touch the network)."""

import httpx
import pytest

from backend import config


@pytest.fixture(autouse=True)
def safe_test_setup(tmp_path, monkeypatch):
    # 1. No network: any real HTTP request fails the test (TestClient works in memory, so it still works).
    async def no_network(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)

    # 2. Mock mode, whatever .env says.
    monkeypatch.setattr(config, "MODE", "mock")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")

    # 3. Counter, cache and results go to a temporary folder, never the real data/ and results/.
    monkeypatch.setattr(config, "API_USAGE_FILE", tmp_path / "api_usage.json")
    monkeypatch.setattr(config, "API_CACHE_FILE", tmp_path / "api_cache.json")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
