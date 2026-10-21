"""Protects the real OpenRouter API: request limits, a saved counter and a response cache.

Every request goes through request(): cache -> limits -> count (saved before sending) -> send.
A dry run does the same on a copy of the counts and never sends anything.
"""

import hashlib
import json
import time

from backend import config
from backend.adapters.base import AdapterError, openrouter_headers, post_json


class CallLimitReached(Exception):
    """A limit is used up. Not an AdapterError, so it is never retried."""


# ---- Small JSON file helpers ----

def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    # Write a temp file, then swap it in, so the file is never half written.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


# ---- Persistent request counter ----

def empty_usage() -> dict:
    return {"total": 0, "per_model": {}, "cache_hits": 0, "total_cost": 0.0}


def load_usage() -> dict:
    try:
        return {**empty_usage(), **read_json(config.API_USAGE_FILE, empty_usage())}
    except ValueError:
        # A broken counter file must never mean "nothing used", so refuse everything instead.
        raise CallLimitReached(f"{config.API_USAGE_FILE.name} is unreadable, so no requests are allowed")


def usage_summary(usage: dict | None = None) -> dict:
    """Numbers for /api/status and the confirm dialog."""
    usage = usage or load_usage()
    per_model = {m["model_id"]: {"used": usage["per_model"].get(m["model_id"], 0), "limit": m["max_requests"]}
                 for m in config.MODELS}
    return {
        "api_calls_used": usage["total"],
        "api_call_limit": config.API_TOTAL_LIMIT,
        "cache_hits": usage["cache_hits"],
        "total_cost": usage["total_cost"],
        "per_model": per_model,
        "max_calls_next_race": max_calls_next_race(usage),
    }


def max_calls_next_race(usage: dict) -> int:
    # With 0 retries a model makes at most one request per move, and never more than it has left.
    per_model = sum(min(config.MAX_MOVES, max(0, m["max_requests"] - usage["per_model"].get(m["model_id"], 0)))
                    for m in config.MODELS)
    return min(per_model, max(0, config.API_TOTAL_LIMIT - usage["total"]))


class Budget:
    """Request counts for one race. The real one always re-reads and saves api_usage.json."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.usage = load_usage()  # a dry run starts from the real numbers but never saves them

    def refresh(self):
        if not self.dry_run:
            self.usage = load_usage()  # the file is the source of truth, even across restarts

    def save(self):
        if not self.dry_run:
            write_json(config.API_USAGE_FILE, self.usage)

    def count_request(self, model_id: str) -> int:
        """Check the limits, then count one request. Returns its number overall."""
        self.refresh()
        used = self.usage["per_model"].get(model_id, 0)
        limit = config.request_limit(model_id)
        if used >= limit:
            raise CallLimitReached(f"{model_id} has used {used} / {limit} requests")
        if self.usage["total"] >= config.API_TOTAL_LIMIT:
            raise CallLimitReached(f"all models together have used {self.usage['total']} / {config.API_TOTAL_LIMIT} requests")
        self.usage["per_model"][model_id] = used + 1
        self.usage["total"] += 1
        self.save()  # saved before the request is sent
        return self.usage["total"]

    def record_cache_hit(self):
        self.refresh()
        self.usage["cache_hits"] += 1
        self.save()

    def record_cost(self, cost: float):
        self.refresh()
        self.usage["total_cost"] += cost
        self.save()


# ---- Response cache ----

def cache_key(model_id: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{model_id}:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"


def load_cache() -> dict:
    try:
        return read_json(config.API_CACHE_FILE, {})
    except ValueError:
        return {}  # an unreadable cache only means more misses; it is never deleted


def save_to_cache(key: str, response: dict):
    cache = load_cache()
    cache[key] = {"response": response, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(config.API_CACHE_FILE, cache)


def response_cost(data: dict) -> float:
    try:
        return float(data.get("usage", {}).get("cost") or 0.0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


# ---- The one way to call OpenRouter ----

async def request(budget: Budget, model_id: str, url: str, payload: dict, fake_response):
    """Returns (response, info). info has the logging fields for this request."""
    info = {"cache_hit": False, "api_latency": None, "cache_latency": None,
            "request_number": None, "cost": 0.0, "dry_run": budget.dry_run}

    start_time = time.perf_counter()
    cached = load_cache().get(cache_key(model_id, payload))
    if cached is not None:
        budget.record_cache_hit()
        info.update(cache_hit=True, cache_latency=time.perf_counter() - start_time)
        return cached["response"], info

    info["request_number"] = budget.count_request(model_id)  # raises CallLimitReached if used up

    if budget.dry_run:
        info["api_latency"] = 0.0
        return fake_response(), info

    start_time = time.perf_counter()
    try:
        data = await post_json(url, payload, headers=openrouter_headers(), timeout=config.MOVE_TIMEOUT)
    except AdapterError as e:
        e.info = {**info, "api_latency": time.perf_counter() - start_time}
        raise
    info["api_latency"] = time.perf_counter() - start_time
    info["cost"] = response_cost(data)
    budget.record_cost(info["cost"])
    save_to_cache(cache_key(model_id, payload), data)
    return data, info
