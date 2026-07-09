"""Unit tests for held-out MRR/Hits@k + the genuine leak-detection guard."""

from __future__ import annotations

import pytest
from latence_linkpred_ultra.candidates import ObservedEdge
from latence_linkpred_ultra.evaluation import (
    LeakDetectionError,
    assert_no_leak,
    metrics_from_ranks,
    rank_of,
    split_edges,
)


def _chain() -> list[ObservedEdge]:
    return [
        ObservedEdge("a", "r", "b"),
        ObservedEdge("b", "r", "c"),
        ObservedEdge("c", "r", "d"),
        ObservedEdge("d", "r", "e"),
    ]


def test_split_is_deterministic_and_removes_heldout() -> None:
    s1 = split_edges(_chain(), holdout_fraction=0.5, seed=7)
    s2 = split_edges(_chain(), holdout_fraction=0.5, seed=7)
    assert s1 == s2
    # Held-out edges are not in the observed set.
    obs = {(e.head, e.relation, e.tail) for e in s1.observed}
    for e in s1.held_out:
        assert (e.head, e.relation, e.tail) not in obs
    assert s1.held_out, "a non-trivial graph must hold out at least one edge"


def test_assert_no_leak_passes_on_clean_split() -> None:
    observed = [ObservedEdge("a", "r", "b")]
    held_out = [ObservedEdge("c", "r", "d")]
    assert_no_leak(observed, held_out)  # no raise


def test_assert_no_leak_catches_direct_leak() -> None:
    observed = [ObservedEdge("a", "r", "b")]
    held_out = [ObservedEdge("a", "r", "b")]  # still in observed → leak
    with pytest.raises(LeakDetectionError, match="leaks into the observed graph"):
        assert_no_leak(observed, held_out)


def test_assert_no_leak_catches_inverse_leak() -> None:
    """The inverse of a held-out edge surviving in the observed graph is ALSO a leak (ULTRA doubles)."""
    observed = [ObservedEdge("b", "r", "a")]  # the inverse of the held-out (a, r, b)
    held_out = [ObservedEdge("a", "r", "b")]
    with pytest.raises(LeakDetectionError, match="inverse"):
        assert_no_leak(observed, held_out)


def test_rank_of_filtered() -> None:
    # scores over 4 entities; target index 2 has the 2nd-highest score.
    scores = [0.1, 0.9, 0.5, 0.2]
    # Unfiltered: one competitor (index 1) beats the target → rank 2.
    assert rank_of(2, scores, filter_indices=set()) == 2
    # Filter out the competitor (a known true tail) → rank 1.
    assert rank_of(2, scores, filter_indices={1}) == 1


def test_metrics_from_ranks() -> None:
    m = metrics_from_ranks([1, 2, 4], ks=(1, 3, 10), leak_checked=True)
    assert m.num_queries == 3
    # MRR = (1/1 + 1/2 + 1/4)/3.
    assert m.mrr == pytest.approx((1 + 0.5 + 0.25) / 3, abs=1e-6)
    assert m.hits[1] == pytest.approx(1 / 3, abs=1e-6)
    assert m.hits[3] == pytest.approx(2 / 3, abs=1e-6)
    assert m.hits[10] == pytest.approx(1.0, abs=1e-6)
    assert m.leak_checked is True


def test_metrics_leak_checked_is_threaded_not_defaulted() -> None:
    # leak_checked is a required, caller-passed value (not a tautological default): passing False
    # is faithfully recorded, so the flag genuinely attests the guard ran.
    assert metrics_from_ranks([1], ks=(1,), leak_checked=False).leak_checked is False
    assert metrics_from_ranks([], ks=(1,), leak_checked=False).leak_checked is False


def test_metrics_empty() -> None:
    m = metrics_from_ranks([], ks=(1, 10), leak_checked=True)
    assert m.mrr == 0.0
    assert m.num_queries == 0
