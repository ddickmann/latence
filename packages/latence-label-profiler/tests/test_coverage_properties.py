"""The (1 - 1/e) guarantee and the certificate, as PROPERTIES over randomly generated instances.

One hand-built instance proves the bound holds once. These prove it holds on instances nobody
designed, including the degenerate shapes a real corpus produces: empty sets, duplicate sets,
one candidate covering everything, weights spanning orders of magnitude.

Both properties are theorems, so a counterexample here is a bug in the implementation, never a
flaky test: greedy's value must clear ``(1 - 1/e) * OPT`` (Nemhauser-Wolsey-Fisher 1978), and the
online certificate must never fall BELOW the brute-forced optimum, or every "provably within X%"
number the artifact prints is a lie.
"""

from __future__ import annotations

from itertools import combinations

from hypothesis import given, settings
from hypothesis import strategies as st
from latence_label_profiler.coverage import (
    WORST_CASE_RATIO,
    CoverageProblem,
    greedy_max_coverage,
)

_ELEMENTS = "abcdefgh"


def _brute_force(problem: CoverageProblem, budget: int) -> float:
    best = 0.0
    names = sorted(problem.sets)
    for size in range(0, min(budget, len(names)) + 1):
        for combo in combinations(names, size):
            covered: set[str] = set()
            for name in combo:
                covered |= problem.sets[name]
            best = max(best, sum(problem.weights[e] for e in covered))
    return best


_problems = st.builds(
    lambda weights, sets: CoverageProblem(
        weights=weights,
        sets={
            f"s{i:02d}": frozenset(members & set(weights))
            for i, members in enumerate(sets)
        },
    ),
    weights=st.dictionaries(
        st.sampled_from(list(_ELEMENTS)),
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
    sets=st.lists(
        st.sets(st.sampled_from(list(_ELEMENTS)), max_size=8), min_size=1, max_size=9
    ),
)


@settings(max_examples=250, deadline=None)
@given(problem=_problems, budget=st.integers(min_value=1, max_value=4))
def test_greedy_always_clears_the_nemhauser_bound(
    problem: CoverageProblem, budget: int
) -> None:
    result = greedy_max_coverage(problem, budget=budget)
    optimum = _brute_force(problem, budget)
    assert result.value >= WORST_CASE_RATIO * optimum - 1e-9
    assert result.value <= optimum + 1e-9
    assert len(result.selected) <= budget


@settings(max_examples=250, deadline=None)
@given(problem=_problems, budget=st.integers(min_value=1, max_value=4))
def test_the_certificate_never_understates_the_optimum(
    problem: CoverageProblem, budget: int
) -> None:
    """If this ever fails, every 'within X% of optimal' the artifact prints is unsound."""
    result = greedy_max_coverage(problem, budget=budget)
    optimum = _brute_force(problem, budget)
    assert result.certificate_bound >= optimum - 1e-9
    for step in result.steps:
        assert step.certificate_bound >= optimum - 1e-9
    assert 0.0 < result.certificate_ratio <= 1.0 + 1e-9


@settings(max_examples=250, deadline=None)
@given(problem=_problems, budget=st.integers(min_value=1, max_value=6))
def test_the_gain_curve_is_monotonically_non_increasing(
    problem: CoverageProblem, budget: int
) -> None:
    """Submodularity's observable consequence — and what makes a Kneedle knee well-defined."""
    gains = [step.marginal_gain for step in greedy_max_coverage(problem, budget=budget).steps]
    for earlier, later in zip(gains, gains[1:], strict=False):
        assert later <= earlier + 1e-9


@settings(max_examples=250, deadline=None)
@given(problem=_problems, budget=st.integers(min_value=1, max_value=6))
def test_selection_is_a_pure_function_of_the_instance(
    problem: CoverageProblem, budget: int
) -> None:
    """Determinism at the algorithm level, under whatever set iteration order the run happens to
    produce — the layer beneath the byte-identical artifact contract.
    """
    first = greedy_max_coverage(problem, budget=budget)
    second = greedy_max_coverage(problem, budget=budget)
    assert first.selected == second.selected
    assert first.value == second.value
