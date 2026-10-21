"""Generates the fixed test sets and saves / loads them from data/puzzles.json.

Run `python -m backend.puzzles` from the project folder to regenerate the file.
"""

import json
import random
from pathlib import Path

from backend.cube_engine import get_state, is_solved, scramble
from backend.solver_bfs import optimal_distance

SEED = 64
PUZZLES_FILE = Path(__file__).resolve().parent.parent / "data" / "puzzles.json"

# set name -> (number of puzzles, min scramble length, max scramble length, store BFS distance?,
#              required BFS distance or None)
# New sets are added at the end, so the random numbers (and puzzles) of the older sets never change.
TEST_SETS = {
    "warmup": (3, 3, 4, True, None),
    "primary": (10, 4, 5, True, None),
    "extension": (3, 15, 20, False, None),
    "openrouter": (1, 3, 3, True, 3),   # the one puzzle for the real API run
    "ollama": (5, 3, 3, True, 3),       # local models
}
MAX_TRIES = 10_000  # per set, so the generator can never loop forever


def generate_puzzles(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    puzzles = []
    seen_states = set()

    for set_name, (count, min_len, max_len, use_bfs, exact_distance) in TEST_SETS.items():
        found = []
        for _ in range(MAX_TRIES):
            if len(found) == count:
                break
            moves = scramble(rng.randint(min_len, max_len), rng)
            state = get_state(moves)
            # Moves like R L R' L' cancel out, so skip solved or repeated cubes.
            if is_solved(state) or state in seen_states:
                continue
            distance = optimal_distance(state) if use_bfs else None
            if exact_distance is not None and distance != exact_distance:
                continue  # for example R L R' is really just one move away
            seen_states.add(state)
            found.append({
                "id": f"{set_name}-{len(found) + 1}",
                "set": set_name,
                "scramble": moves,
                "scramble_length": len(moves),
                "state": state,
                "optimal_distance": distance,
            })
        if len(found) < count:
            raise RuntimeError(f"could only make {len(found)} of {count} puzzles for the {set_name} set")
        puzzles.extend(found)

    return puzzles


def save_puzzles(puzzles: list[dict], path: Path = PUZZLES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"seed": SEED, "puzzles": puzzles}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_puzzles(path: Path = PUZZLES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["puzzles"]


def get_puzzle(puzzle_id: str, path: Path = PUZZLES_FILE) -> dict:
    for puzzle in load_puzzles(path):
        if puzzle["id"] == puzzle_id:
            return puzzle
    raise KeyError(f"No puzzle with id {puzzle_id!r}")


if __name__ == "__main__":
    puzzles = generate_puzzles()
    save_puzzles(puzzles)
    print(f"Saved {len(puzzles)} puzzles to {PUZZLES_FILE}")
    for p in puzzles:
        distance = p["optimal_distance"] if p["optimal_distance"] is not None else "-"
        print(f"  {p['id']:<12} len {p['scramble_length']:<3} optimal {distance:<3} {' '.join(p['scramble'])}")
