"""The selection core: greedy submodular max-coverage, its guarantee, and its certificate.

Every assertion here is against a *published* property, not against our own output:

* Nemhauser-Wolsey-Fisher (1978): for a monotone submodular ``f`` with ``f(empty) = 0``, greedy
  under a cardinality constraint ``|A| <= N`` returns ``f(A) >= (1 - 1/e) * f(OPT_N)``.
* Feige (1998): no polynomial algorithm beats ``(1 - 1/e)`` for max-coverage unless P = NP, which
  is why greedy is the *defensible* choice rather than the lazy one.
* The per-instance certificate ``f(OPT_N) <= f(A_i) + sum of the N largest marginal gains at A_i``
  is the standard submodular online bound; it must be a genuine UPPER bound on the brute-forced
  optimum on every instance we can brute force.

The instance in :func:`_suboptimal_instance` is chosen so greedy is STRICTLY worse than optimal.
A test whose instance greedy solves exactly would assert nothing about the bound.
"""

from __future__ import annotations

import math
from itertools import combinations

from latence_label_profiler.coverage import (
    WORST_CASE_RATIO,
    CoverageProblem,
    greedy_max_coverage,
)


def _suboptimal_instance() -> CoverageProblem:
    """A weighted instance where greedy is provably NOT optimal at N = 2.

    Elements ``a b c d`` with weights ``1.0 1.2 1.2 1.0``. The optimal pair is ``{A, B}``
    (all four elements, 4.4). Greedy first takes ``G`` (2.4 > 2.2), then can only add 1.0 more,
    landing on 3.4 -> ratio 0.7727, strictly between (1 - 1/e) and 1.
    """
    return CoverageProblem(
        weights={"a": 1.0, "b": 1.2, "c": 1.2, "d": 1.0},
        sets={
            "A": frozenset({"a", "b"}),
            "B": frozenset({"c", "d"}),
            "G": frozenset({"b", "c"}),
        },
    )


def _brute_force_optimum(problem: CoverageProblem, budget: int) -> float:
    best = 0.0
    for combo in combinations(sorted(problem.sets), budget):
        covered: set[str] = set()
        for name in combo:
            covered |= problem.sets[name]
        best = max(best, sum(problem.weights[e] for e in covered))
    return best


def test_greedy_is_strictly_suboptimal_on_this_instance() -> None:
    """Guard the guard: if greedy were optimal here, the (1 - 1/e) assertion would be vacuous."""
    problem = _suboptimal_instance()
    result = greedy_max_coverage(problem, budget=2)
    optimum = _brute_force_optimum(problem, 2)
    assert result.value < optimum
    assert result.selected == ("G", "A")


def test_greedy_lands_within_one_minus_one_over_e() -> None:
    problem = _suboptimal_instance()
    result = greedy_max_coverage(problem, budget=2)
    optimum = _brute_force_optimum(problem, 2)
    assert result.value >= WORST_CASE_RATIO * optimum
    assert math.isclose(WORST_CASE_RATIO, 1.0 - 1.0 / math.e)


def test_certificate_is_a_real_upper_bound_on_the_brute_forced_optimum() -> None:
    """``f(OPT_N) <= f(A_i) + sum of the N largest remaining gains`` — on every step."""
    problem = _suboptimal_instance()
    result = greedy_max_coverage(problem, budget=2)
    optimum = _brute_force_optimum(problem, 2)
    for step in result.steps:
        assert step.certificate_bound >= optimum - 1e-12
    assert result.certificate_bound >= optimum - 1e-12
    # And it is TIGHTER than the worst case: the per-corpus certificate is the number worth
    # reporting, exactly as the ticket asks.
    assert result.certificate_ratio > WORST_CASE_RATIO


def test_marginal_gains_are_monotonically_non_increasing() -> None:
    """Submodularity's observable consequence — the property a knee search depends on."""
    problem = CoverageProblem(
        weights={f"e{i}": float(40 - i) for i in range(40)},
        sets={f"s{i}": frozenset({f"e{i}", f"e{(i * 7) % 40}"}) for i in range(40)},
    )
    result = greedy_max_coverage(problem, budget=12)
    gains = [step.marginal_gain for step in result.steps]
    assert gains == sorted(gains, reverse=True)


def test_greedy_is_deterministic_under_label_ties() -> None:
    """Equal gains must break on the label, so the same problem yields the same byte output."""
    problem = CoverageProblem(
        weights={"x": 1.0, "y": 1.0},
        sets={"zeta": frozenset({"x"}), "alpha": frozenset({"y"})},
    )
    assert greedy_max_coverage(problem, budget=1).selected == ("alpha",)
    assert greedy_max_coverage(problem, budget=1).selected == ("alpha",)


def test_budget_larger_than_the_candidate_pool_stops_at_full_coverage() -> None:
    problem = _suboptimal_instance()
    result = greedy_max_coverage(problem, budget=99)
    assert result.value == sum(problem.weights.values())
    assert result.coverage_fraction == 1.0
    assert len(result.selected) == 3


def test_empty_problem_is_not_a_crash() -> None:
    result = greedy_max_coverage(CoverageProblem(weights={}, sets={}), budget=5)
    assert result.selected == ()
    assert result.value == 0.0
    assert result.coverage_fraction == 0.0
