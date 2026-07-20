"""Precision / recall / F1 — the second reporting currency of the harness.

The Parse suite scores *unit tests*, so its currency is the pass/fail/unsupported tally in
:class:`~latence_benchmark.suite.SplitScore`. The suites added by #187 score *labelled sets*
(entity spans, relation triples, coreferent mention pairs, PII spans), and a pass rate cannot
express those: a system that predicts nothing would score 100% "no false positives". The honest
currency for a labelled set is the confusion triple, so this module defines it once —
:class:`MetricScore` — and every new suite reports through it rather than inventing its own
shape (:mod:`latence_benchmark.suite` explains why that matters).

Two deliberate properties:

* **An undefined ratio is ``None``, never ``0.0``.** Precision with no predictions at all is not
  zero, it is undefined; reporting it as ``0.0`` would let an empty run look like a measured
  failure and, worse, drag a micro-average down as if it were data.
* **Matching is multiset matching.** :func:`confusion` counts duplicate identical items rather
  than collapsing them into a set, so a Provider that emits the same span twice is charged one
  false positive instead of being silently forgiven by de-duplication.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

T = TypeVar("T")


@dataclass(frozen=True)
class MetricScore:
    """One slice of a labelled-set measurement, as true/false positives and false negatives.

    :param dimension: What the slice varies — ``"language"``, ``"pii_type"``, ``"match"``. The
        results page groups by it, so one suite can report several orthogonal breakdowns
        (per-language *and* per-type) without either of them pretending to be the other.
    :param name: The slice within that dimension (``"de"``, ``"email"``, ``"span"``).
    """

    dimension: str
    name: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def gold(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float | None:
        """TP / predicted, or ``None`` when the system predicted nothing (undefined)."""
        return self.true_positives / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        """TP / gold, or ``None`` when the slice has no gold items (undefined)."""
        return self.true_positives / self.gold if self.gold else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall; ``None`` if either is undefined."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "name": self.name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def confusion(
    dimension: str,
    name: str,
    *,
    gold: Iterable[T],
    predicted: Iterable[T],
) -> MetricScore:
    """Score ``predicted`` against ``gold`` as a multiset match.

    Items are compared by equality/hash, so the *definition of a match* lives in the item type a
    suite chooses (an exact-boundary span key, a relation triple, a mention pair) rather than in
    a flag here. Duplicates count: emitting the same item twice against one gold item yields one
    true positive and one false positive.
    """
    gold_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    true_positives = sum(min(count, gold_counts[item]) for item, count in predicted_counts.items())
    return MetricScore(
        dimension=dimension,
        name=name,
        true_positives=true_positives,
        false_positives=sum(predicted_counts.values()) - true_positives,
        false_negatives=sum(gold_counts.values()) - true_positives,
    )


def micro(scores: Sequence[MetricScore], *, dimension: str, name: str = "MICRO") -> MetricScore:
    """Micro-average ``scores`` by summing their confusion counts.

    Micro rather than macro: summing the counts weights every *item* equally, so a slice with
    three gold spans cannot swing the headline number as hard as one with three thousand. A
    macro average over wildly uneven language slices would flatter or damn the system by
    accident.

    This is the raw sum and asks no questions of its inputs: summing is only *meaningful* when
    the scores partition one gold set, and the caller is what knows that. Callers aggregating a
    whole dimension must go through :func:`total`, which checks it.
    """
    return MetricScore(
        dimension=dimension,
        name=name,
        true_positives=sum(s.true_positives for s in scores),
        false_positives=sum(s.false_positives for s in scores),
        false_negatives=sum(s.false_negatives for s in scores),
    )


#: Dimensions whose slices are **alternative criteria over one gold set**, not a partition of it.
#:
#: A ``language`` or ``pii_type`` slice owns a disjoint share of the gold items, so the slices'
#: counts add up to the corpus and a micro-average over them is a real measurement. A ``match``
#: slice does not: it re-scores *the same* gold items under a stricter or looser rule for what
#: counts as a hit (``coverage`` vs ``exact span`` vs ``exact span+type``). Summing those counts
#: each gold item once per criterion — 35 gold spans become a denominator of 105 — and blends
#: three mutually redundant views of one detection set into a number that measures nothing.
#:
#: So no aggregate is defined here. That is not a gap to be filled with a different average:
#: there is no single quantity that "the match dimension overall" names.
CRITERIA_DIMENSIONS: frozenset[str] = frozenset({"match"})


class MetricAggregationError(ValueError):
    """An aggregate was requested over a dimension whose slices do not partition a gold set."""


def is_partition(dimension: str) -> bool:
    """Do this dimension's slices carve one gold set into disjoint parts (so a total exists)?"""
    return dimension not in CRITERIA_DIMENSIONS


def total(scores: Sequence[MetricScore], *, dimension: str, name: str = "MICRO") -> MetricScore:
    """The aggregate row for ``dimension`` — refusing the dimensions that have no aggregate.

    :raises MetricAggregationError: if ``dimension`` is a criteria dimension
        (:data:`CRITERIA_DIMENSIONS`). Loudly, never as a quietly wrong number: a published
        aggregate that multiply-counts its gold set is exactly the kind of confident fiction
        this harness exists to keep out of ``docs/BENCHMARKS.md``.
    """
    if not is_partition(dimension):
        msg = (
            f"Dimension {dimension!r} has no aggregate: its slices are alternative scoring "
            "criteria over the SAME gold items, not disjoint slices of them, so summing their "
            "confusion counts would count every gold item once per criterion. Report the "
            "criteria individually."
        )
        raise MetricAggregationError(msg)
    return micro(scores, dimension=dimension, name=name)
