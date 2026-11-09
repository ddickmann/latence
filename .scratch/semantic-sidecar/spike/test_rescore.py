"""The second stage must add information only where it has some, and never loosen the guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from er import Bands
from rescore import feature_matrix, record_text, stage_two, uncertain_region


def test_record_text_leaves_missing_fields_out() -> None:
    row = {"name": "Northwind Ltd", "city": None, "postcode": float("nan"), "reg_id": "12345678", "industry": " "}
    assert record_text(row, ("name", "city", "postcode", "reg_id", "industry")) == "name: Northwind Ltd | reg_id: 12345678"


def test_uncertain_region_is_what_the_first_stage_left_open() -> None:
    p = np.array([0.01, 0.30, 0.60, 0.90, 0.60])
    veto = np.array([False, False, False, False, True])
    bands = Bands(accept_at=0.8, reject_below=0.2, target=0.95, delta=0.05, n_labelled=0)
    assert uncertain_region(p, bands, veto).tolist() == [False, True, True, False, False]
    # A first stage that abstained on the accept side leaves the top open as well.
    open_top = Bands(accept_at=None, reject_below=0.2, target=0.95, delta=0.05, n_labelled=0)
    assert uncertain_region(p, open_top, veto).tolist() == [False, True, True, True, False]


def test_a_scorer_that_cannot_answer_is_marked_missing_not_scored_zero() -> None:
    p1 = np.array([0.4, 0.5, 0.6, 0.7])
    X, names = feature_matrix(p1, {"decide": np.array([0.9, np.nan, 0.1, 0.8])})
    assert names == ["splink_logit", "decide", "decide_missing"]
    assert X[1, 1] == pytest.approx(0.8)  # the median of the answered pairs, not 0
    assert X[:, 2].tolist() == [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        feature_matrix(p1, {"decide": np.full(4, np.nan)})
    with pytest.raises(ValueError):
        feature_matrix(p1, {"decide": np.zeros(3)})


def _uncertain_world(n: int, seed: int, informative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-stage probabilities bunched in the middle (that is why the pairs are uncertain),
    and a second scorer that either separates the classes or is pure noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    p1 = np.clip(0.5 + 0.05 * rng.standard_normal(n), 0.3, 0.7)
    s = (3.0 * (2 * y - 1) + rng.standard_normal(n)) if informative else rng.standard_normal(n)
    return p1, s, y


def test_an_informative_scorer_resolves_uncertain_pairs_within_the_precision_target() -> None:
    p1, s, y = _uncertain_world(6000, 5, informative=True)
    idx = np.random.default_rng(1).permutation(len(y))
    out = stage_two(p1, {"rerank": s}, y, idx[:1000], idx[1000:2500], target=0.95, delta=0.025)
    assert out.bands.accept_at is not None
    rest = idx[2500:]  # pairs no label was read from
    accepted = rest[out.p2[rest] >= out.bands.accept_at]
    assert len(accepted) > 0.3 * len(rest) * y.mean()
    assert y[accepted].mean() >= 0.95
    assert abs(out.coefficients[out.feature_names.index("rerank")]) > abs(out.coefficients[0])


def test_a_noise_scorer_grants_no_accept_band() -> None:
    p1, s, y = _uncertain_world(6000, 9, informative=False)
    idx = np.random.default_rng(2).permutation(len(y))
    out = stage_two(p1, {"rerank": s}, y, idx[:1000], idx[1000:2500], target=0.95, delta=0.025)
    assert out.bands.accept_at is None  # no evidence -> the pairs stay with the expert


def test_overlapping_label_samples_are_refused() -> None:
    p1, s, y = _uncertain_world(100, 3, informative=True)
    with pytest.raises(ValueError):
        stage_two(p1, {"rerank": s}, y, np.arange(0, 60), np.arange(50, 100))
