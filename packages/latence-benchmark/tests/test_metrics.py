"""The confusion-triple currency: what it counts, and what it refuses to pretend."""

from __future__ import annotations

import pytest
from latence_benchmark.metrics import MetricScore, confusion, micro


def test_an_undefined_ratio_is_none_not_zero() -> None:
    """A system that predicted nothing has undefined precision, not 0% precision.

    Reporting 0.0 would make "we ran nothing" indistinguishable from "we were wrong about
    everything", and would drag a micro-average down as if it were data.
    """
    nothing_predicted = MetricScore("d", "n", true_positives=0, false_positives=0, false_negatives=3)
    assert nothing_predicted.precision is None
    assert nothing_predicted.recall == 0.0
    assert nothing_predicted.f1 is None

    no_gold = MetricScore("d", "n", true_positives=0, false_positives=2, false_negatives=0)
    assert no_gold.recall is None
    assert no_gold.precision == 0.0


def test_precision_recall_f1_are_the_standard_quantities() -> None:
    score = MetricScore("d", "n", true_positives=6, false_positives=2, false_negatives=3)
    assert score.precision == pytest.approx(0.75)
    assert score.recall == pytest.approx(6 / 9)
    assert score.f1 == pytest.approx(2 * 0.75 * (6 / 9) / (0.75 + 6 / 9))
    assert (score.gold, score.predicted) == (9, 8)


def test_a_repeated_prediction_is_charged_as_a_false_positive() -> None:
    """Multiset matching. De-duplicating predictions would forgive a Provider that emits the
    same span twice, and quietly inflate its precision."""
    score = confusion("d", "n", gold=["a", "b"], predicted=["a", "a", "b"])
    assert (score.true_positives, score.false_positives, score.false_negatives) == (2, 1, 0)


def test_a_duplicated_gold_item_needs_two_predictions() -> None:
    score = confusion("d", "n", gold=["a", "a"], predicted=["a"])
    assert (score.true_positives, score.false_negatives) == (1, 1)


def test_micro_sums_counts_rather_than_averaging_ratios() -> None:
    """A three-item slice must not swing the headline as hard as a three-thousand-item one."""
    big = MetricScore("language", "en", true_positives=900, false_positives=100, false_negatives=0)
    tiny = MetricScore("language", "zh", true_positives=0, false_positives=3, false_negatives=0)
    total = micro([big, tiny], dimension="language")
    assert total.precision is not None
    assert total.precision == pytest.approx(900 / 1003)
    # A macro average of the two precisions would be 0.45 — a very different, and here
    # misleading, story.
    assert total.precision > 0.85
    assert total.name == "MICRO"


def test_metric_serialisation_carries_the_counts_and_the_ratios() -> None:
    payload = MetricScore("language", "de", 3, 1, 1).as_dict()
    assert payload["dimension"] == "language"
    assert payload["name"] == "de"
    assert payload["true_positives"] == 3
    assert payload["precision"] == pytest.approx(0.75)
    assert payload["f1"] == pytest.approx(0.75)
