"""Zipf alpha, Good-Turing missing mass, Kneedle — each against its published definition."""

from __future__ import annotations

import math

from latence_label_profiler.stats import (
    fit_zipf,
    good_turing_missing_mass,
    kneedle,
)


def test_good_turing_missing_mass_is_singletons_over_total() -> None:
    """Good (1953): the probability the next observation is of an UNSEEN type is ``N1 / N``."""
    counts = {"a": 5, "b": 3, "c": 1, "d": 1}
    # N1 = 2 singletons, N = 10 observations.
    assert math.isclose(good_turing_missing_mass(counts), 0.2)


def test_good_turing_is_zero_when_nothing_was_seen_once() -> None:
    assert good_turing_missing_mass({"a": 4, "b": 2}) == 0.0


def test_good_turing_on_an_empty_corpus_is_total_ignorance_not_a_crash() -> None:
    """No observations means no evidence AGAINST the next mention being unseen: report 1.0."""
    assert good_turing_missing_mass({}) == 1.0


def test_zipf_alpha_recovers_a_synthetic_power_law() -> None:
    """A rank-frequency table built with a known exponent must fit back near that exponent."""
    true_alpha = 1.0
    counts = {f"t{r}": max(1, int(round(100000 / (r**true_alpha)))) for r in range(1, 200)}
    fit = fit_zipf(counts)
    assert 0.85 <= fit.rank_slope <= 1.15
    assert fit.n_types == 199
    assert fit.total_mass == sum(counts.values())


def test_zipf_alpha_near_one_is_reported_as_no_sharp_knee() -> None:
    """The honesty requirement: alpha ~ 1 means logarithmic coverage growth, so N is a budget."""
    counts = {f"t{r}": max(1, int(round(100000 / r))) for r in range(1, 400)}
    assert fit_zipf(counts).log_growth is True


def test_a_steep_distribution_is_not_reported_as_log_growth() -> None:
    counts = {f"t{r}": max(1, int(round(100000 / (r**2.5)))) for r in range(1, 200)}
    assert fit_zipf(counts).log_growth is False


def test_kneedle_finds_the_knee_of_a_concave_increasing_curve() -> None:
    """Satopaa et al. (2011): normalise, take y_norm - x_norm, the knee is its argmax."""
    xs = list(range(1, 21))
    ys = [1.0 - 0.6**x for x in xs]  # sharp saturation early
    knee = kneedle(xs, ys)
    assert knee.x is not None
    assert 2 <= knee.x <= 6
    assert knee.sharp is True


def test_kneedle_on_a_straight_line_reports_no_sharp_knee() -> None:
    xs = list(range(1, 41))
    ys = [x / 40.0 for x in xs]
    knee = kneedle(xs, ys)
    assert knee.max_difference < 1e-9
    assert knee.sharp is False


def test_kneedle_on_a_logarithmic_curve_reports_a_weak_knee() -> None:
    """The alpha ~ 1 case the ticket calls out: a knee exists numerically but is not sharp."""
    xs = list(range(1, 41))
    ys = [math.log1p(x) / math.log1p(40) for x in xs]
    knee = kneedle(xs, ys)
    assert knee.sharp is False


def test_kneedle_needs_three_points() -> None:
    knee = kneedle([1, 2], [0.5, 0.9])
    assert knee.x is None
    assert knee.sharp is False
