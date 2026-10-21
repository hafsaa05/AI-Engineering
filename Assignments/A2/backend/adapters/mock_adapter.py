"""Fake models for testing the whole app before we have an API key."""

import asyncio
import random

from backend.adapters.base import Adapter, MoveResult, make_result
from backend.cube_engine import MOVES
from backend.solver_bfs import solve

ORACLE_DELAY = 0.3       # seconds, so the oracle looks like a slow but perfect model
ORACLE_MAX_DEPTH = 6     # deeper cubes (extension set) are too far, the oracle then guesses
NOISY_ORACLE_RATE = 0.7  # how often mock-noisy plays the oracle move
BROKEN_RATE = 0.3        # how often mock-broken returns junk
BAD_OUTPUTS = ["R3", "r", "", None, "U2'", "Turn the right face", "X", 42]


class MockAdapter(Adapter):
    kind = "MOCK"
    mode = "mock"
    delay = 0.0

    def __init__(self, name: str, delay: float | None = None, seed: int | None = None):
        super().__init__(name, model_id=name)
        if delay is not None:
            self.delay = delay
        self.rng = random.Random(seed)

    async def get_move(self, state: str, recent_moves: list[str]) -> MoveResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        output = await self.pick(state)
        return make_result(output, raw_response=output)

    async def pick(self, state: str):
        raise NotImplementedError


class RandomMock(MockAdapter):
    async def pick(self, state):
        return self.rng.choice(MOVES)


class OracleMock(MockAdapter):
    delay = ORACLE_DELAY

    async def pick(self, state):
        # BFS can take a moment, so run it off the event loop.
        solution = await asyncio.to_thread(solve, state, ORACLE_MAX_DEPTH)
        if solution:
            return solution[0]
        return self.rng.choice(MOVES)


class NoisyMock(OracleMock):
    async def pick(self, state):
        if self.rng.random() < NOISY_ORACLE_RATE:
            return await super().pick(state)
        return self.rng.choice(MOVES)


class BrokenMock(OracleMock):
    async def pick(self, state):
        if self.rng.random() < BROKEN_RATE:
            return self.rng.choice(BAD_OUTPUTS)
        return await super().pick(state)


MOCK_ADAPTERS = {
    "mock-random": RandomMock,
    "mock-oracle": OracleMock,
    "mock-noisy": NoisyMock,
    "mock-broken": BrokenMock,
}
