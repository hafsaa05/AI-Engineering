"""Cube engine: scrambles, moves and state checks for a 3x3 Rubik's Cube.

State is a 54 character facelet string in URFDLB face order (kociemba style),
for example the solved cube is "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB".
"""

import random

import magiccube

FACES = "URFDLB"
MOVES = [face + suffix for face in FACES for suffix in ("", "'", "2")]
SOLVED_STATE = "".join(face * 9 for face in FACES)

# magiccube works with colors and stores faces in U L F R B D order,
# so we need these to convert our facelet strings into its format.
FACE_TO_COLOR = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
MAGICCUBE_FACE_ORDER = "ULFRBD"


def to_magiccube_image(state: str) -> str:
    faces = {face: state[i * 9:(i + 1) * 9] for i, face in enumerate(FACES)}
    return "".join(FACE_TO_COLOR[ch] for face in MAGICCUBE_FACE_ORDER for ch in faces[face])


def apply_move_with_magiccube(state: str, move: str) -> str:
    """Slow but direct: rebuild a magiccube cube, turn it, read it back."""
    cube = magiccube.Cube(3, hist=False)
    cube.set(to_magiccube_image(state))
    cube.rotate(move)
    return cube.get_kociemba_facelet_positions()


def learn_permutation(move: str) -> list[int]:
    # A move only moves stickers around, so it is a fixed reordering of the
    # 54 positions. We find it by marking one sticker at a time and seeing
    # where magiccube puts it. Result: new_state[j] = old_state[perm[j]].
    perm = [0] * 54
    for i in range(54):
        marked = "U" * i + "R" + "U" * (53 - i)
        j = apply_move_with_magiccube(marked, move).index("R")
        perm[j] = i
    return perm


def build_move_tables() -> dict[str, list[int]]:
    tables = {}
    for face in FACES:
        quarter = learn_permutation(face)
        double = [quarter[i] for i in quarter]   # quarter turn done twice
        prime = [double[i] for i in quarter]     # quarter turn done three times
        tables[face] = quarter
        tables[face + "2"] = double
        tables[face + "'"] = prime
    return tables


# Built once at import, then every move is just a fast string reorder.
MOVE_TABLES = build_move_tables()


def is_valid_move(move) -> bool:
    return isinstance(move, str) and move in MOVE_TABLES


def face_of(move: str) -> str:
    return move[0]


def inverse_move(move: str) -> str:
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return move + "'"


def apply_move(state: str, move: str) -> str:
    if not is_valid_move(move):
        raise ValueError(f"Invalid move: {move!r}")
    return "".join([state[i] for i in MOVE_TABLES[move]])


def apply_moves(state: str, moves: list[str]) -> str:
    for move in moves:
        state = apply_move(state, move)
    return state


def get_state(moves: list[str]) -> str:
    """State reached by applying the given moves to a solved cube."""
    return apply_moves(SOLVED_STATE, moves)


def is_solved(state: str) -> bool:
    return state == SOLVED_STATE


def correct_stickers(state: str) -> int:
    """How many of the 54 stickers are where they are on a solved cube (the 6 centers always are)."""
    return sum(a == b for a, b in zip(state, SOLVED_STATE))


def scramble(length: int, rng: random.Random | None = None) -> list[str]:
    """Random move sequence that never turns the same face twice in a row."""
    rng = rng or random.Random()
    moves = []
    while len(moves) < length:
        move = rng.choice(MOVES)
        if moves and face_of(move) == face_of(moves[-1]):
            continue
        moves.append(move)
    return moves
