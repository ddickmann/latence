"""The suggester learns combinations the per-field weights cannot, and says how far to trust it."""

from __future__ import annotations

import numpy as np
import pytest

from suggest import MIN_LEAF, suggest, wording

COMPARISONS = ["name", "city", "industry"]
LABELS = {"name": "Name", "city": "City", "industry": "Industry"}
LEVELS = {
    "name": {-1: "missing", 0: "else", 1: "close", 2: "closer", 3: "exact"},
    "city": {-1: "missing", 0: "else", 1: "exact"},
    "industry": {-1: "missing", 0: "else", 1: "exact"},
}


def _planted(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same city always. The pair is the same entity unless the industry differs while the name
    is only close: a combination, which no single field decides."""
    rng = np.random.default_rng(seed)
    name = rng.integers(0, 4, size=n)
    industry = rng.integers(-1, 2, size=n)
    same = ~((industry == 0) & (name < 3))
    return np.column_stack([name, np.ones(n, dtype=int), industry]), same.astype(int)


def test_it_finds_a_planted_combination_and_reports_it_out_of_sample() -> None:
    gammas, same = _planted(600, seed=3)
    suggestions, summary = suggest(gammas, same, COMPARISONS, LABELS, LEVELS, seed=1)
    assert summary["available"] is True and summary["labelled_pairs"] == 600
    right = sum((s["verdict"] == "same") == bool(y) for s, y in zip(suggestions, same.tolist()) if s is not None)
    assert summary["out_of_sample_right"] == right
    assert summary["out_of_sample_share"] == round(right / 600, 4)
    # The planted rule is exact, so a tree that found it is right nearly everywhere; calling
    # everything "same" is right only as often as pairs are the same.
    assert summary["out_of_sample_share"] > 0.97 > summary["always_same_share"]


def test_a_suggestion_carries_its_conditions_and_its_record() -> None:
    gammas, same = _planted(600, seed=3)
    suggestions, _ = suggest(gammas, same, COMPARISONS, LABELS, LEVELS, seed=1)
    i = int(np.where((gammas[:, 2] == 0) & (gammas[:, 0] == 1))[0][0])  # industry differs, name only close
    s = suggestions[i]
    assert s is not None and s["verdict"] == "different"
    assert s["like_this"] >= MIN_LEAF and 0 < s["like_this_agree"] <= s["like_this"]
    said = {c["label"]: c["text"] for c in s["conditions"]}
    assert said["Industry"] == "differs"
    # The tree splits on "level <= 2", a range that also covers a missing name.
    assert said["Name"] == "not the same, or missing"
    assert "City" not in said  # a field the tree never tested is not a reason
    for c in s["conditions"]:
        assert c["levels"] == sorted(c["levels"]) and set(c["levels"]) <= set(LEVELS[c["comparison"]])


def test_it_declines_when_there_is_too_little_to_learn_from() -> None:
    gammas, same = _planted(40, seed=3)
    suggestions, summary = suggest(gammas, same, COMPARISONS, LABELS, LEVELS, seed=1)
    assert suggestions == [None] * 40 and summary["available"] is False
    gammas, _ = _planted(400, seed=3)
    suggestions, summary = suggest(gammas, np.ones(400, dtype=int), COMPARISONS, LABELS, LEVELS, seed=1)
    assert suggestions == [None] * 400 and summary["available"] is False


@pytest.mark.parametrize(
    ("levels", "top", "text"),
    [
        ([-1], 3, "missing on one side"),
        ([-1, 0], 3, "differs or is missing"),
        ([-1, 0, 1, 2], 3, "not the same, or missing"),
        ([-1, 0, 1, 2, 3], 3, "any"),
        ([0], 3, "differs"),
        ([0, 1, 2], 3, "not the same"),
        ([1, 2], 3, "close, not the same"),
        ([1, 2, 3], 3, "close or the same"),
        ([0, 1, 2, 3], 3, "present on both sides"),
        ([3], 3, "agrees"),
        ([0, 1], 1, "present on both sides"),
        ([1], 1, "agrees"),
        ([], 1, "any"),
    ],
)
def test_wording_of_a_level_range(levels: list[int], top: int, text: str) -> None:
    assert wording(levels, top) == text
