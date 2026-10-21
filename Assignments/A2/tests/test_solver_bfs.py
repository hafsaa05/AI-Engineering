import random
import time

from backend.cube_engine import MOVES, SOLVED_STATE, apply_move, apply_moves, get_state, is_solved, scramble
from backend.solver_bfs import optimal_distance, solve


def plain_bfs_distances(max_depth: int) -> dict:
    """Simple one sided BFS from solved, used to check the fast solver."""
    distances = {SOLVED_STATE: 0}
    layer = [SOLVED_STATE]
    for depth in range(1, max_depth + 1):
        next_layer = []
        for state in layer:
            for move in MOVES:
                nxt = apply_move(state, move)
                if nxt not in distances:
                    distances[nxt] = depth
                    next_layer.append(nxt)
        layer = next_layer
    return distances


def test_solved_cube_needs_no_moves():
    assert solve(SOLVED_STATE) == []


def test_single_moves():
    for move in MOVES:
        assert len(solve(get_state([move]))) == 1


def test_known_short_cases():
    assert optimal_distance(get_state(["R", "L"])) == 2
    assert optimal_distance(get_state(["R", "U", "R'", "U'"])) == 4
    # R L' R' is really just L', so the true distance is shorter than the scramble.
    assert optimal_distance(get_state(["R", "L'", "R'"])) == 1


def test_solution_actually_solves():
    rng = random.Random(5)
    for _ in range(10):
        state = get_state(scramble(5, rng))
        assert is_solved(apply_moves(state, solve(state)))


def test_matches_plain_bfs():
    distances = plain_bfs_distances(3)
    rng = random.Random(11)
    for _ in range(40):
        state = get_state(scramble(rng.randint(1, 3), rng))
        assert optimal_distance(state) == distances[state]


def test_never_longer_than_scramble():
    rng = random.Random(9)
    for _ in range(10):
        moves = scramble(5, rng)
        assert optimal_distance(get_state(moves)) <= 5


def test_returns_none_past_max_depth():
    state = get_state(["R", "U", "F", "D"])
    assert solve(state, max_depth=3) is None
    assert len(solve(state, max_depth=4)) == 4


def test_depth_six_is_fast():
    state = get_state(["R", "U", "F", "L", "D", "B"])
    start_time = time.perf_counter()
    solution = solve(state)
    assert len(solution) == 6
    assert time.perf_counter() - start_time < 5
