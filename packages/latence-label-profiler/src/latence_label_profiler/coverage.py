"""Greedy submodular max-coverage, with the per-corpus optimality certificate it yields for free.

The label-selection problem is exactly weighted max-coverage: pick ``A`` with ``|A| <= N``
maximising the mention mass covered by at least one selected type. That objective is monotone
submodular with ``f({}) = 0``, so:

* **Greedy is guaranteed** — Nemhauser, Wolsey & Fisher (1978):
  ``f(A_greedy) >= (1 - 1/e) f(OPT_N)``.
* **Nothing polynomial beats it** — Feige (1998): ``(1 - 1/e)`` is optimal for max-coverage unless
  P = NP. Greedy is therefore the *defensible* choice, not the lazy one.
* **Every run certifies itself** — for any prefix ``A_i``, submodularity gives
  ``f(OPT_N) <= f(A_i) + sum of the N largest marginal gains at A_i``. Taking the minimum over ``i``
  gives a per-corpus upper bound on the optimum that is usually far tighter than the 63.2% worst
  case, which is the number actually worth reporting: "these N labels are within X% of the best
  possible N *on this corpus*".

This module implements NAIVE greedy rather than CELF/lazy greedy on purpose. CELF returns the same
solution, but only naive greedy evaluates *every* candidate's marginal gain at *every* step — and
those gains are precisely what the certificate and the marginal-gain curve are made of. Paying
``O(N * |candidates|)`` set operations buys the artifact's entire uncertainty section.

Ties are broken by label, ascending. Without that, two candidates with equal gain would be ordered
by whatever the underlying container yielded, and the "same corpus + seed => byte-identical output"
contract would hold only for as long as the string-hash seed did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

#: ``1 - 1/e``. The Nemhauser-Wolsey-Fisher (1978) worst-case approximation ratio for greedy on a
#: monotone submodular objective under a cardinality constraint, and (Feige 1998) the best any
#: polynomial algorithm can guarantee for max-coverage unless P = NP. It is a FLOOR, not a
#: prediction: the per-instance certificate below is normally much tighter.
WORST_CASE_RATIO: float = 1.0 - 1.0 / math.e


@dataclass(frozen=True)
class CoverageProblem:
    """One max-coverage instance: element weights, and the element set each candidate covers."""

    weights: Mapping[str, float]
    sets: Mapping[str, frozenset[str]]

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))


@dataclass(frozen=True)
class GreedyStep:
    """One greedy selection, with the numbers that make the choice auditable after the fact."""

    rank: int
    label: str
    marginal_gain: float
    cumulative: float
    coverage_fraction: float
    #: ``f(OPT_N) <= this``, evaluated at the prefix this step produced.
    certificate_bound: float


@dataclass(frozen=True)
class GreedyResult:
    """The selection, the curve, and the certificate."""

    selected: tuple[str, ...]
    steps: tuple[GreedyStep, ...]
    value: float
    total_weight: float
    #: The tightest (minimum) certificate across every prefix, clamped to the total mass.
    certificate_bound: float
    candidates_considered: int
    worst_case_floor: float = field(default=WORST_CASE_RATIO)

    @property
    def coverage_fraction(self) -> float:
        """``C(N)`` — the share of total mention mass the selected types cover."""
        return self.value / self.total_weight if self.total_weight > 0 else 0.0

    @property
    def certificate_ratio(self) -> float:
        """``f(A) / upper bound on f(OPT_N)`` — the per-corpus optimality guarantee.

        ``1.0`` means the selection is provably optimal on this corpus. Anything above
        :data:`WORST_CASE_RATIO` is a stronger statement than the theorem alone can make.
        """
        if self.certificate_bound <= 0:
            return 1.0
        return min(1.0, self.value / self.certificate_bound)


def _gains(
    problem: CoverageProblem, covered: set[str], remaining: list[str]
) -> list[tuple[float, str]]:
    """Every remaining candidate's marginal gain, sorted by (-gain, label) for determinism."""
    scored = [
        (
            float(sum(problem.weights[e] for e in problem.sets[label] if e not in covered)),
            label,
        )
        for label in remaining
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def greedy_max_coverage(problem: CoverageProblem, *, budget: int) -> GreedyResult:
    """Select at most ``budget`` candidates by greedy marginal gain, certifying the result.

    Stops early when the best remaining gain is zero — adding a candidate that covers nothing new
    would spend one of the <= N label slots (which cost real GLiNER2 latency, DECISIONS.md D5b) to
    buy nothing.
    """
    total = problem.total_weight
    remaining = sorted(problem.sets)
    covered: set[str] = set()
    selected: list[str] = []
    steps: list[GreedyStep] = []
    value = 0.0
    bounds: list[float] = []

    for _ in range(max(0, budget)):
        if not remaining:
            break
        scored = _gains(problem, covered, remaining)
        bounds.append(value + sum(gain for gain, _ in scored[:budget]))
        best_gain, best_label = scored[0]
        if best_gain <= 0.0:
            break
        covered |= problem.sets[best_label]
        remaining.remove(best_label)
        selected.append(best_label)
        value += best_gain
        # The certificate for THIS prefix needs the gains at A_i, i.e. after the selection. It is
        # appended by the next iteration's first statement (or by the tail pass below), so the step
        # records a placeholder that is rewritten once that bound exists.
        steps.append(
            GreedyStep(
                rank=len(selected),
                label=best_label,
                marginal_gain=best_gain,
                cumulative=value,
                coverage_fraction=value / total if total > 0 else 0.0,
                certificate_bound=math.inf,
            )
        )
    if remaining:
        scored = _gains(problem, covered, remaining)
        bounds.append(value + sum(gain for gain, _ in scored[:budget]))
    else:
        bounds.append(value)

    # bounds[i] is the certificate evaluated at prefix A_i (bounds[0] is at the empty set), so step
    # i (1-based) carries bounds[i]. Clamp to the total mass: f is bounded by the whole universe, so
    # a bound above it is arithmetically valid but useless, and reporting it would overstate the
    # room greedy left on the table.
    resolved: list[GreedyStep] = []
    for step in steps:
        bound = bounds[step.rank] if step.rank < len(bounds) else bounds[-1]
        resolved.append(
            GreedyStep(
                rank=step.rank,
                label=step.label,
                marginal_gain=step.marginal_gain,
                cumulative=step.cumulative,
                coverage_fraction=step.coverage_fraction,
                certificate_bound=min(bound, total) if total > 0 else bound,
            )
        )
    best_bound = min(bounds) if bounds else 0.0
    if total > 0:
        best_bound = min(best_bound, total)

    return GreedyResult(
        selected=tuple(selected),
        steps=tuple(resolved),
        value=value,
        total_weight=total,
        certificate_bound=best_bound,
        candidates_considered=len(problem.sets),
    )
