"""The benchmark-suite seam: what a Stage benchmark is, and how a new one plugs in.

**This is the extension point issue #187 builds on.** A benchmark suite is one Stage measured
on one pinned public dataset. Everything around it — dataset acquisition, the license gate, the
CLI, the results page, the JSON artifact — is written against :class:`BenchmarkSuite` and
:class:`SuiteResult` only, so adding the Entity/Relation-Extraction, Disambiguation or
Redaction suites is a new module plus one :func:`register_suite` call, with no change to any of
the surrounding machinery.

To add a suite:

1. Write a class satisfying :class:`BenchmarkSuite` — ``name``, ``stage``, ``dataset``, a
   ``stack_pin`` describing the exact Providers+config the numbers were produced with, and a
   ``run(root)`` that scores a materialised dataset directory.
2. If it needs a new dataset, add a :class:`~latence_benchmark.datasets.DatasetSpec` with a full
   commit SHA and a **separately verified** corpus/toolkit license record (ADR-0012).
3. Call :func:`register_suite` on it at import time in ``latence_benchmark.__init__``.

``latence-benchmark run --suite <name>`` and the published results page then pick it up with no
further wiring. What a suite must NOT do is invent its own reporting shape. There are exactly
two currencies, because there are exactly two kinds of measurement:

* :class:`SplitScore` — pass/fail/unsupported, for a suite scoring **unit tests** (Parse).
  ``unsupported`` exists so a partially implemented scorer is visible in the results rather than
  flattering the pass rate.
* :class:`~latence_benchmark.metrics.MetricScore` — true/false positives and false negatives,
  for a suite scoring a **labelled set** (Entity/Relation Extraction, Disambiguation,
  Redaction). A pass rate cannot express these: a system predicting nothing would score a
  perfect "no false positives".

A suite fills whichever applies (or both) and leaves the other empty; the results page renders
only what is populated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_benchmark.metrics import MetricScore, total
from latence_benchmark.scoring import Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from latence_benchmark.datasets import DatasetSpec
    from latence_benchmark.scoring import TestResult


@dataclass(frozen=True)
class StackPin:
    """The exact pipeline configuration a suite's numbers were produced with.

    A benchmark number without the stack that produced it is a rumour, so this is recorded in
    every result and fingerprinted: :attr:`fingerprint` is a sha256 over the canonical JSON of
    the pin, which makes "same configuration" a one-token check for a reader comparing two
    published runs.
    """

    stage: str
    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Short, stable sha256 digest over the canonicalised pin."""
        canonical = json.dumps(
            {"stage": self.stage, "provider": self.provider, "config": dict(self.config)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "config": dict(self.config),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class SplitScore:
    """Per-split tally. ``pass_rate`` is over SCORED tests only (passed + failed)."""

    split: str
    passed: int
    failed: int
    unsupported: int

    @property
    def scored(self) -> int:
        return self.passed + self.failed

    @property
    def total(self) -> int:
        return self.scored + self.unsupported

    @property
    def pass_rate(self) -> float | None:
        """Fraction of scored tests that passed, or ``None`` when nothing was scorable."""
        return self.passed / self.scored if self.scored else None

    @classmethod
    def from_results(cls, split: str, results: Sequence[TestResult]) -> SplitScore:
        tally = {outcome: 0 for outcome in Outcome}
        for result in results:
            tally[result.outcome] += 1
        return cls(
            split=split,
            passed=tally[Outcome.PASSED],
            failed=tally[Outcome.FAILED],
            unsupported=tally[Outcome.UNSUPPORTED],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "passed": self.passed,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "scored": self.scored,
            "pass_rate": self.pass_rate,
        }


@dataclass(frozen=True)
class SuiteResult:
    """Everything a reader needs to trust — and reproduce — one suite's numbers."""

    suite: str
    stage: str
    dataset: DatasetSpec
    stack_pin: StackPin
    splits: tuple[SplitScore, ...]
    #: Documents whose Parse failed outright. Their tests are counted as failures, never
    #: skipped — a parser that crashes on a file did not pass that file's tests.
    document_failures: int = 0
    documents_scored: int = 0
    notes: tuple[str, ...] = ()
    #: Labelled-set measurements, one per slice (see :class:`~latence_benchmark.metrics.
    #: MetricScore`). Empty for a unit-test suite.
    metrics: tuple[MetricScore, ...] = ()
    #: Which metric dimension carries this suite's headline number, e.g. ``"language"``. The
    #: micro-average over that dimension is what the CLI prints and the page leads with.
    headline_dimension: str | None = None
    #: ISO date this measurement was produced. The results page is assembled from however many
    #: runs it took to cover every Stage (see :func:`~latence_benchmark.results.carry_forward`),
    #: so each measurement states its own date rather than inheriting the page's.
    measured_on: str = ""

    def metrics_for(self, dimension: str) -> tuple[MetricScore, ...]:
        """This suite's metric slices along one dimension, in the order the suite emitted them."""
        return tuple(m for m in self.metrics if m.dimension == dimension)

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Metric dimensions present, first-seen order (the order the suite considers useful)."""
        return tuple(dict.fromkeys(m.dimension for m in self.metrics))

    @property
    def headline(self) -> MetricScore | None:
        """Micro-average over :attr:`headline_dimension`, or ``None`` for a unit-test suite.

        Goes through :func:`~latence_benchmark.metrics.total`, so naming a *criteria* dimension
        (whose slices re-score one gold set under different rules) as the headline raises rather
        than leading the page with a number whose denominator is a multiple of the gold set.
        """
        if self.headline_dimension is None:
            return None
        slices = self.metrics_for(self.headline_dimension)
        return total(slices, dimension=self.headline_dimension) if slices else None

    @property
    def overall(self) -> SplitScore:
        return SplitScore(
            split="OVERALL",
            passed=sum(s.passed for s in self.splits),
            failed=sum(s.failed for s in self.splits),
            unsupported=sum(s.unsupported for s in self.splits),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "stage": self.stage,
            "measured_on": self.measured_on,
            "dataset": {
                "name": self.dataset.name,
                "repo_id": self.dataset.repo_id,
                "revision": self.dataset.revision,
                "revision_kind": self.dataset.revision_kind,
                "homepage": self.dataset.homepage,
                "citation": self.dataset.citation,
                "license": {
                    "dataset_license": self.dataset.license.dataset_license,
                    "code_license": self.dataset.license.code_license,
                    "commercial_use": self.dataset.license.commercial_use,
                    "attribution_required": self.dataset.license.attribution_required,
                    "verified_on": self.dataset.license.verified_on,
                    "source_url": self.dataset.license.source_url,
                },
            },
            "stack_pin": self.stack_pin.as_dict(),
            "splits": [s.as_dict() for s in self.splits],
            "overall": self.overall.as_dict(),
            "documents_scored": self.documents_scored,
            "document_failures": self.document_failures,
            "notes": list(self.notes),
            "metrics": [m.as_dict() for m in self.metrics],
            "headline_dimension": self.headline_dimension,
            "headline": self.headline.as_dict() if self.headline is not None else None,
        }


@runtime_checkable
class BenchmarkSuite(Protocol):
    """One Stage benchmarked on one pinned public dataset."""

    # Declared as read-only properties, not attributes, so an immutable (frozen-dataclass)
    # suite satisfies the protocol — a benchmark suite that could be mutated mid-run is
    # exactly what a reproducible harness must not allow.

    @property
    def name(self) -> str:
        """Registry key, e.g. ``"parse"``."""
        ...

    @property
    def stage(self) -> str:
        """The pipeline Stage under measurement, e.g. ``"parse"``."""
        ...

    @property
    def dataset(self) -> DatasetSpec:
        """The pinned dataset this suite scores against."""
        ...

    @property
    def stack_pin(self) -> StackPin:
        """The exact Providers + config the numbers are produced with."""
        ...

    def run(self, root: Path, *, limit: int | None = None) -> SuiteResult:
        """Score a materialised dataset directory and return the result.

        ``limit`` caps the documents scored per split; a capped run must record itself as a
        clearly-labelled subset in :attr:`SuiteResult.notes`.
        """
        ...


_REGISTRY: dict[str, BenchmarkSuite] = {}


def register_suite(suite: BenchmarkSuite) -> None:
    """Register ``suite`` under its ``name``. Re-registering the same name is refused."""
    if suite.name in _REGISTRY:
        msg = f"A benchmark suite named {suite.name!r} is already registered."
        raise ValueError(msg)
    _REGISTRY[suite.name] = suite


def suite_names() -> Sequence[str]:
    """Registered suite names, sorted."""
    return sorted(_REGISTRY)


def get_suite(name: str) -> BenchmarkSuite:
    """Return the registered suite called ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(suite_names()) or "<none>"
        msg = f"Unknown benchmark suite {name!r}. Available: {available}"
        raise KeyError(msg) from None
