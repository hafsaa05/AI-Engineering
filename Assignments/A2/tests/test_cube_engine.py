import random

import pytest

from backend.cube_engine import (
    MOVES, SOLVED_STATE, apply_move, apply_move_with_magiccube, apply_moves,
    correct_stickers, face_of, get_state, inverse_move, is_solved, is_valid_move, scramble,
)


def test_eighteen_moves():
    assert len(MOVES) == 18
    assert len(set(MOVES)) == 18


def test_solved_state():
    assert len(SOLVED_STATE) == 54
    assert is_solved(SOLVED_STATE)
    assert not is_solved(apply_move(SOLVED_STATE, "R"))


def test_correct_stickers():
    assert correct_stickers(SOLVED_STATE) == 54
    assert correct_stickers(get_state(["U"])) == 42  # U moves 12 side stickers out of place
    assert correct_stickers(get_state(["R", "U", "F"])) < 42


def test_known_r_move():
    # Hand checked: R lifts the front right column onto U, and so on round the cube.
    expected = ("UUFUUFUUF" "RRRRRRRRR" "FFDFFDFFD" "DDBDDBDDB" "LLLLLLLLL" "UBBUBBUBB")
    assert apply_move(SOLVED_STATE, "R") == expected


@pytest.mark.parametrize("move", MOVES)
def test_tables_match_magiccube(move):
    state = get_state(scramble(20, random.Random(1)))
    assert apply_move(state, move) == apply_move_with_magiccube(state, move)


@pytest.mark.parametrize("move", MOVES)
def test_move_then_inverse_is_identity(move):
    state = get_state(["R", "U", "F'"])
    assert apply_move(apply_move(state, move), inverse_move(move)) == state


def test_four_quarter_turns_is_identity():
    for face in "URFDLB":
        assert apply_moves(SOLVED_STATE, [face] * 4) == SOLVED_STATE


def test_sexy_move_has_order_six():
    # (R U R' U') repeated 6 times returns to solved.
    assert apply_moves(SOLVED_STATE, ["R", "U", "R'", "U'"] * 6) == SOLVED_STATE


def test_undoing_a_scramble_solves_it():
    moves = scramble(25, random.Random(7))
    undo = [inverse_move(m) for m in reversed(moves)]
    assert is_solved(apply_moves(get_state(moves), undo))


def test_invalid_moves_are_rejected():
    for bad in ["X", "r", "R3", "", None, "R '", "U2'"]:
        assert not is_valid_move(bad)
        with pytest.raises(ValueError):
            apply_move(SOLVED_STATE, bad)


def test_scramble_never_repeats_a_face():
    moves = scramble(200, random.Random(3))
    assert len(moves) == 200
    assert all(face_of(a) != face_of(b) for a, b in zip(moves, moves[1:]))


def test_scramble_is_reproducible_with_seed():
    assert scramble(10, random.Random(42)) == scramble(10, random.Random(42))
