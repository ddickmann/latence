"""Checks that the spike's own machinery is right before any number it produces is believed."""

from __future__ import annotations

import numpy as np
import pytest

from er import (
    ConstrainedClusterer,
    learn_then_test_bands,
    pairwise_and_bcubed,
    plain_closure,
    stable_ids,
)


def test_metrics_on_a_hand_computed_case() -> None:
    gold = {"a": "1", "b": "1", "c": "1", "d": "2"}
    pred = {"a": "x", "b": "x", "c": "y", "d": "y"}
    m = pairwise_and_bcubed(pred, gold)
    # gold pairs: ab ac bc = 3; pred pairs: ab, cd = 2; tp: ab = 1
    assert m["pairwise_precision"] == pytest.approx(0.5)
    assert m["pairwise_recall"] == pytest.approx(1 / 3)
    # b-cubed precision: a 2/2... a: |{a,b}∩{a,b,c}|/2=1, b: 1, c: 1/2, d: 1/2 -> 0.75
    assert m["bcubed_precision"] == pytest.approx(0.75)


def test_ltt_accept_band_holds_its_guarantee_on_a_known_distribution() -> None:
    """Calibrated probabilities that ARE true probabilities: the realised precision of the
    accepted set must be >= target in (almost) every repetition, as the guarantee states."""
    rng = np.random.default_rng(7)
    failures = abstained = 0
    reps = 200
    for _ in range(reps):
        p = rng.beta(0.6, 0.6, size=4000)
        y = (rng.random(4000) < p).astype(int)
        lab, test = slice(0, 2000), slice(2000, 4000)
        bands = learn_then_test_bands(p[lab], y[lab], target=0.95, delta=0.05)
        if bands.accept_at is None:
            abstained += 1  # conservative: no band granted, nothing auto-accepted
            continue
        sel = p[test] >= bands.accept_at
        if y[test][sel].mean() < 0.95:
            failures += 1
    # delta = 0.05 bounds the failure rate (with slack for the finite test half), and the
    # method must not abstain so often that it is useless on a well-separated distribution.
    assert failures / reps <= 0.10
    assert abstained / reps <= 0.25


def test_ltt_refuses_an_accept_band_when_scores_carry_no_signal() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(3000)
    y = (rng.random(3000) < 0.5).astype(int)  # labels independent of scores
    bands = learn_then_test_bands(p, y, target=0.95, delta=0.05)
    assert bands.accept_at is None  # no evidence -> no auto-accept: unknown stays unknown


def test_cannot_link_is_never_crossed_even_through_a_chain() -> None:
    cc = ConstrainedClusterer(["a", "b", "c", "d"])
    cc.add_cannot_link("a", "d")
    assert cc.try_union("a", "b")
    assert cc.try_union("c", "d")
    assert not cc.try_union("b", "c")  # would put a and d together via b~c
    clusters = cc.clusters()
    assert clusters["a"] != clusters["d"]
    assert cc.blocked_merges == 1


def test_contradicting_cannot_link_raises() -> None:
    cc = ConstrainedClusterer(["a", "b"])
    cc.try_union("a", "b")
    with pytest.raises(ValueError):
        cc.add_cannot_link("a", "b")


def test_plain_closure_chains_where_constrained_does_not() -> None:
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    closed = plain_closure("abcd", edges)
    assert len(set(closed.values())) == 1


def test_stable_ids_keep_the_previous_id_by_overlap() -> None:
    prev = {"a": "E1", "b": "E1", "c": "E2"}
    new = {"a": "x", "b": "x", "c": "y", "d": "y"}
    assert stable_ids(new, prev) == {"a": "E1", "b": "E1", "c": "E2", "d": "E2"}
