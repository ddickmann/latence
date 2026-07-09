"""Unit tests for the temperature-scaling + precision-targeted-threshold calibrator."""

from __future__ import annotations

import pytest
from latence_linkpred_ultra.calibrator import (
    Calibrator,
    CalibratorConfig,
    sigmoid,
)


def test_probability_is_a_calibrated_sigmoid_not_raw() -> None:
    cal = Calibrator(CalibratorConfig(temperature=2.0))
    # A raw logit of 0 → 0.5; a large positive raw → near 1; the raw value itself is NOT returned.
    assert cal.probability(0.0, 2.0) == 0.5
    assert cal.probability(20.0, 2.0) > 0.99
    assert cal.probability(-20.0, 2.0) < 0.01
    # It is bounded to [0,1] — a raw score of 8.0 is never emitted verbatim as "confidence".
    p = cal.probability(8.0, 2.0)
    assert 0.0 <= p <= 1.0
    assert p != 8.0


def test_fit_without_labels_falls_back_to_config() -> None:
    cal = Calibrator(CalibratorConfig(temperature=3.0, min_probability=0.7))
    fit = cal.fit(None)
    assert fit.temperature == 3.0
    assert fit.threshold == 0.7


def test_fit_temperature_minimises_nll_on_separable_data() -> None:
    """On separable labelled data the fitted temperature yields well-calibrated probabilities."""
    # True edges have high raw scores, false edges low — clearly separable (>= min samples).
    labelled = [(6.0, True), (5.0, True), (5.5, True), (6.5, True)] + [
        (-5.0, False),
        (-6.0, False),
        (-5.5, False),
        (-6.5, False),
    ]
    cal = Calibrator(CalibratorConfig(target_precision=1.0, min_calibration_samples=4))
    fit = cal.fit(labelled)
    # A positive temperature was chosen from the grid.
    assert fit.temperature > 0
    # The precision-targeted threshold cleanly separates: it admits the positives (precision 1.0)
    # and excludes the negatives.
    pos = cal.probability(6.0, fit.temperature)
    neg = cal.probability(-6.0, fit.temperature)
    assert pos >= fit.threshold
    assert neg < fit.threshold


def test_precision_target_refuses_to_emit_when_unreachable() -> None:
    """If no cut reaches the target precision, the threshold is set so NOTHING is emitted."""
    # Fully overlapping classes: no threshold can achieve precision 1.0.
    labelled = [(1.0, True), (1.0, False)] * 4
    cal = Calibrator(CalibratorConfig(target_precision=1.0, min_calibration_samples=4))
    fit = cal.fit(labelled)
    # Every calibrated probability is below the threshold — the honest "don't assert" disposition.
    p = cal.probability(1.0, fit.temperature)
    assert p < fit.threshold


def test_determinism() -> None:
    labelled = [(4.0, True), (-4.0, False), (3.0, True), (-3.0, False)]
    a = Calibrator(CalibratorConfig()).fit(labelled)
    b = Calibrator(CalibratorConfig()).fit(labelled)
    assert a == b


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        Calibrator(CalibratorConfig(temperature=0.0))
    with pytest.raises(ValueError, match="target_precision"):
        Calibrator(CalibratorConfig(target_precision=1.5))


def test_sigmoid_stable_extremes() -> None:
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)
