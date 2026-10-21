from backend.cube_engine import get_state, is_solved
from backend.puzzles import TEST_SETS, generate_puzzles, get_puzzle, load_puzzles, save_puzzles


def test_set_sizes_and_lengths():
    puzzles = generate_puzzles()
    for set_name, (count, min_len, max_len, _, _) in TEST_SETS.items():
        group = [p for p in puzzles if p["set"] == set_name]
        assert len(group) == count
        assert all(min_len <= p["scramble_length"] <= max_len for p in group)


def test_new_sets_are_exactly_three_moves_away():
    puzzles = generate_puzzles()
    openrouter = [p for p in puzzles if p["set"] == "openrouter"]
    ollama = [p for p in puzzles if p["set"] == "ollama"]
    assert len(openrouter) == 1 and len(ollama) == 5
    assert all(p["optimal_distance"] == 3 for p in openrouter + ollama)


def test_same_seed_gives_same_puzzles():
    assert generate_puzzles(1) == generate_puzzles(1)
    assert generate_puzzles(1) != generate_puzzles(2)


def test_puzzles_are_valid():
    puzzles = generate_puzzles()
    assert len({p["state"] for p in puzzles}) == len(puzzles)
    for p in puzzles:
        assert p["state"] == get_state(p["scramble"])
        assert not is_solved(p["state"])
        if p["set"] == "extension":
            assert p["optimal_distance"] is None
        else:
            assert 1 <= p["optimal_distance"] <= p["scramble_length"]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "puzzles.json"
    puzzles = generate_puzzles()
    save_puzzles(puzzles, path)
    assert load_puzzles(path) == puzzles
    assert get_puzzle("primary-3", path) == puzzles[5]


def test_saved_file_matches_seed():
    # data/puzzles.json must be exactly what the fixed seed produces.
    assert load_puzzles() == generate_puzzles()
