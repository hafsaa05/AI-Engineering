"""Exact shortest solution with bidirectional BFS over all 18 moves.

One search goes forward from the scramble and one backward from solved, a layer at a time.
Where they meet is a shortest solution, and each side only goes about half as deep.
"""

from backend.cube_engine import MOVES, SOLVED_STATE, apply_move, face_of, inverse_move

DEFAULT_MAX_DEPTH = 8


def expand_layer(frontier: dict, visited: dict) -> dict:
    """Take every state in the frontier one move further. Returns the new layer."""
    new_layer = {}
    for state, path in frontier.items():
        for move in MOVES:
            # Turning the same face twice in a row is never needed.
            if path and face_of(move) == face_of(path[-1]):
                continue
            nxt = apply_move(state, move)
            if nxt not in visited and nxt not in new_layer:
                new_layer[nxt] = path + [move]
    return new_layer


def join_paths(forward_path: list[str], backward_path: list[str]) -> list[str]:
    # The backward path goes solved -> meeting state, so undo it in reverse.
    return forward_path + [inverse_move(m) for m in reversed(backward_path)]


def solve(state: str, max_depth: int = DEFAULT_MAX_DEPTH) -> list[str] | None:
    """Shortest move list that solves the state, or None if longer than max_depth."""
    if state == SOLVED_STATE:
        return []

    forward_seen = {state: []}
    backward_seen = {SOLVED_STATE: []}
    forward_frontier = dict(forward_seen)
    backward_frontier = dict(backward_seen)
    depth = 0

    while depth < max_depth:
        # Grow the smaller side, it is cheaper.
        if len(forward_frontier) <= len(backward_frontier):
            forward_frontier = expand_layer(forward_frontier, forward_seen)
            forward_seen.update(forward_frontier)
            new_layer = forward_frontier
        else:
            backward_frontier = expand_layer(backward_frontier, backward_seen)
            backward_seen.update(backward_frontier)
            new_layer = backward_frontier
        depth += 1

        # The first layer where the two sides meet gives a shortest path.
        for s in new_layer:
            if s in forward_seen and s in backward_seen:
                return join_paths(forward_seen[s], backward_seen[s])

    return None


def optimal_distance(state: str, max_depth: int = DEFAULT_MAX_DEPTH) -> int | None:
    solution = solve(state, max_depth)
    return None if solution is None else len(solution)
