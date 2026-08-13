# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The second fusion variant: per-signal RRF weights **fitted on dev, reported on test**.

**Why this exists.** Every S5 ladder leg fuses its signals with ``RrfFuser`` at equal weights, and
the measured result is that every graph leg (``kg``, ``hyperedges``, ``coe``, ``packer``) *reduces*
recall against ``dense+bm25``. That is exactly what equal-weight RRF must do when a noisier list is
appended to a strong pool: each list contributes ``weight_i / (k + rank_i)``, so a list that is
right 30% of the time pulls with the same force as one that is right 70% of the time. The
equal-weight number therefore answers "do these signals help *at equal weight*?" — not "do these
signals carry usable information?". This module answers the second question, and only the second.

**The honesty constraint, which is the whole design.** A weight fitted on a set of queries and
reported on the same queries measures the fitting, not the retrieval — it is strictly worse than
the untuned baseline it would replace, because it looks better while meaning less. So:

* every dataset's usable queries are partitioned into a **dev** and a **test** half by a stable
  hash of the query id (:func:`is_dev`) — no RNG, no ordering dependence, no ``--limit``
  dependence: a query is on the same side of the split in every run, on every machine, whatever
  else was sampled alongside it;
* the weights are fitted **only** on dev (:func:`tune_weights` never receives a test query, and
  :class:`TunedLeg` records both id sets so a test can assert it);
* the reported metrics are computed **only** on test;
* and the *comparison* number — the equal-weight baseline the delta is taken against — is
  recomputed on the **same test queries** from the **same retrieved lists**
  (:class:`TunedLeg.equal`), so the delta is a difference in fusion and nothing else. Comparing a
  tuned test-split number against the equal-weight full-set cell would confound the fusion change
  with the split.

**The search.** Coordinate ascent over :data:`WEIGHT_GRID`, starting from all-ones (the
equal-weight baseline), maximising **mean nDCG@10** as defined by :mod:`benchmark.s5.metrics` — the
same definition the report prints, so the objective and the headline are the same quantity. The
first signal is pinned at ``1.0`` (see :func:`~benchmark.s5.legs.tunable_signals`: RRF weights are
scale-invariant, so one degree of freedom is redundant), a value is only adopted on a *strict*
improvement (ties keep the incumbent, so the baseline is never displaced by noise), and the sweep
stops as soon as a full pass changes nothing. Five signals × seven values would be 16 807 grid
points; coordinate ascent visits at most ``passes × 4 × 7`` — the reason the search is a defensible
cost rather than an overnight job.

A full-grid alternative ships in the library as
:func:`latence_retrieval.tuning.sweep_fusion_weights`; it is not used here for two reasons worth
stating rather than hiding: the cartesian product over five signals blows past its own
``max_combinations`` guard, and its nDCG is not the S5 set-gold nDCG (ideal capped at
``min(|gold|, k)``), so tuning against it would optimise a different number from the one reported.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from benchmark.s5.legs import (
    LEG_TABLE,
    SignalRanking,
    fuse_signals,
    leg_signals,
    tunable_signals,
)
from benchmark.s5.metrics import NDCG_K, CellAccumulator, ndcg_at_k

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from benchmark.s5.legs import Runner

#: Salt for the dev/test hash. Changing it re-draws every split, so it is a versioned constant
#: recorded in every cell rather than a literal buried in the hash call.
SPLIT_SALT = "s5-fusion-2026-08"

#: Share of queries in the dev (fitting) half, in permille of the hash bucket. 30% is enough to
#: fit at most four weights while leaving 70% — the majority — as the untouched reporting set.
DEV_PERMILLE = 300

#: The weight values coordinate ascent may try. ``0.0`` is included on purpose: "drop this list
#: entirely" is a legitimate answer, and one the equal-weight run cannot express.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

#: Ceiling on coordinate-ascent passes. The loop exits earlier the moment a pass adopts nothing,
#: and :attr:`TuningResult.converged` records which of the two actually happened — so a weight
#: vector that merely ran out of budget is never presented as a settled optimum.
MAX_PASSES = 5

#: How many dev queries the fit actually uses. Bruch et al. (TOIS 2023) find "just a handful of
#: labeled queries" suffices for fusion weights, and an uncapped fit would re-fuse tens of
#: thousands of queries per grid point. The cap is applied to the dev half in query-id order, so
#: which queries are used is as deterministic as the split itself — and it can only ever shrink
#: the FITTING set, never touch the reporting one.
DEV_TUNE_CAP = 300

#: The quantity coordinate ascent maximises, named the way the report prints it.
OBJECTIVE = f"ndcg@{NDCG_K}"

T = TypeVar("T")


def is_dev(query_id: str) -> bool:
    """Whether ``query_id`` belongs to the dev (fitting) half of the split.

    A stable keyed digest, not :func:`hash` — Python's string hash is salted per process, so a
    split built on it would silently differ between the tuning run and any later audit of it. This
    is a pure function of the id, so the split is reproducible from the id alone and does not
    depend on how many queries were sampled or in what order they arrived.
    """
    digest = hashlib.blake2b(
        f"{SPLIT_SALT}:{query_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % 1000 < DEV_PERMILLE


@dataclass(frozen=True)
class Split(Generic[T]):
    """One dataset's queries partitioned into the fitting half and the reporting half."""

    dev: tuple[T, ...]
    test: tuple[T, ...]

    def describe(self, tuned_on: int) -> dict[str, Any]:
        """The split's audit record — what goes into the cell so a reader can re-derive it."""
        return {
            "scheme": "blake2b(salt:query_id) % 1000 < dev_permille",
            "salt": SPLIT_SALT,
            "dev_permille": DEV_PERMILLE,
            "dev_queries": len(self.dev),
            "dev_queries_tuned_on": tuned_on,
            "dev_tune_cap": DEV_TUNE_CAP,
            "test_queries": len(self.test),
        }


def split_queries(items: Sequence[T], key: Callable[[T], str]) -> Split[T]:
    """Partition ``items`` by :func:`is_dev` on ``key(item)``, preserving input order.

    Disjoint and exhaustive by construction — every item lands on exactly one side — so no query
    can be both fitted on and reported on, and none can be quietly dropped.
    """
    dev = tuple(item for item in items if is_dev(key(item)))
    test = tuple(item for item in items if not is_dev(key(item)))
    return Split(dev=dev, test=test)


@dataclass(frozen=True)
class DevRun:
    """One dev query as the tuner sees it: its retrieved lists and its gold. Nothing else.

    Deliberately not a :class:`~benchmark.s5.goldmap.ResolvedGold`: the tuner has no business
    holding a query's text or its position in the dataset, and a narrow record is what makes
    "the tuner never saw a test query" a checkable property rather than a promise.
    """

    query_id: str
    rankings: tuple[SignalRanking, ...]
    gold: frozenset[str]


@dataclass(frozen=True)
class TuningResult:
    """The fitted weights and the audit trail of how they were reached."""

    weights: dict[str, float]
    objective: str
    pinned: str | None
    tunable: tuple[str, ...]
    grid: tuple[float, ...]
    dev_query_ids: tuple[str, ...]
    dev_score_equal: float
    dev_score_best: float
    evaluations: int
    passes: int
    #: Whether a full pass adopted nothing (a local optimum) rather than the loop hitting
    #: :data:`MAX_PASSES`. Recorded because "the search stopped" and "the search ran out of
    #: budget" are different claims, and only the first one licenses calling the weights fitted.
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """The JSON record for the cell — weights first, then how they were found."""
        return {
            "weights": dict(self.weights),
            "objective": self.objective,
            "method": "coordinate-ascent",
            "pinned_signal": self.pinned,
            "tunable_signals": list(self.tunable),
            "grid": list(self.grid),
            "grid_evaluations": self.evaluations,
            "passes": self.passes,
            "max_passes": MAX_PASSES,
            "converged": self.converged,
            f"dev_{self.objective}_equal_weight": round(self.dev_score_equal, 4),
            f"dev_{self.objective}_tuned": round(self.dev_score_best, 4),
        }


def dev_objective(
    runner: Runner, runs: Sequence[DevRun], weights: Mapping[str, float] | None
) -> float:
    """Mean nDCG@10 of ``weights`` over ``runs`` — the number coordinate ascent climbs.

    Scored against the *fused* ranking, before the ``packer`` rung's budgeted selection: the
    packer consumes a fused list and does not change what fusion should optimise, and leaving it
    out keeps one weight vector per leg-family rather than one per token budget.
    """
    if not runs:
        msg = "dev_objective needs at least one dev run; an empty fitting set tunes nothing."
        raise ValueError(msg)
    total = 0.0
    for run in runs:
        fused = fuse_signals(runner.fuser, run.rankings, weights)
        total += ndcg_at_k([c.id for c in fused], run.gold)
    return total / len(runs)


def tune_weights(runner: Runner, leg: str, runs: Sequence[DevRun]) -> TuningResult:
    """Fit ``leg``'s per-signal weights on ``runs`` by coordinate ascent over :data:`WEIGHT_GRID`.

    Starts at all-ones — the equal-weight baseline — and only adopts a value that *strictly*
    improves the dev objective, so the fit can never return something dev-worse than the baseline
    and a tie always keeps the incumbent. A leg with nothing tunable (a single-signal leg) returns
    the all-ones vector after a single evaluation, because with one list the RRF weight is a pure
    monotone scale and every value ranks identically.
    """
    signals = leg_signals(leg)
    tunable = tunable_signals(leg)
    weights = {name: 1.0 for name in signals}
    equal = dev_objective(runner, runs, weights)
    best = equal
    evaluations = 1
    passes = 0
    converged = not tunable
    for _ in range(MAX_PASSES):
        if not tunable:
            break
        adopted = False
        for name in tunable:
            for value in WEIGHT_GRID:
                if value == weights[name]:
                    continue
                trial = dict(weights)
                trial[name] = value
                score = dev_objective(runner, runs, trial)
                evaluations += 1
                if score > best + 1e-12:
                    best = score
                    weights[name] = value
                    adopted = True
        passes += 1
        if not adopted:
            converged = True
            break
    return TuningResult(
        weights=weights,
        objective=OBJECTIVE,
        pinned=signals[0] if tunable else None,
        tunable=tunable,
        grid=WEIGHT_GRID,
        dev_query_ids=tuple(run.query_id for run in runs),
        dev_score_equal=equal,
        dev_score_best=best,
        evaluations=evaluations,
        passes=passes,
        converged=converged,
    )


@dataclass(frozen=True)
class ScoredQuery:
    """One usable query with everything a fusion variant needs in order to be scored on it."""

    query_id: str
    question: str
    vector: Any
    gold: frozenset[str]
    chain: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True)
class TunedLeg:
    """One (dataset, leg) weighted cell: the fit, and BOTH variants scored on the test half."""

    tuning: TuningResult
    split: dict[str, Any]
    tuned: dict[str, float | int | None]
    equal: dict[str, float | int | None]
    dev_query_ids: tuple[str, ...] = field(default=())
    test_query_ids: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class InsufficientSplit:
    """Refusal: the split left one half empty, so nothing can be honestly fitted or reported."""

    dev_queries: int
    test_queries: int

    def to_dict(self) -> dict[str, Any]:
        """The cell body — a stated refusal, never a zero or a same-split number."""
        return {
            "status": "insufficient-split",
            "reason": (
                "the dev/test split left a half empty; fitting and reporting on the same "
                "queries would measure the fit, not the retrieval"
            ),
            "dev_queries": self.dev_queries,
            "test_queries": self.test_queries,
        }


def run_tuned_leg(
    runner: Runner, leg: str, queries: Sequence[ScoredQuery]
) -> TunedLeg | InsufficientSplit:
    """Fit ``leg``'s weights on the dev half of ``queries``; score both variants on the test half.

    Two retrieval passes over disjoint halves — never one over everything — so the dev lists and
    the test lists are produced by exactly the same code and neither half is retrieved twice. On
    the test half each query is retrieved **once** and fused **twice** (equal, then tuned), which
    is what makes :attr:`TunedLeg.equal` and :attr:`TunedLeg.tuned` differ by the fusion alone.
    """
    split = split_queries(queries, key=lambda q: q.query_id)
    if not split.dev or not split.test:
        return InsufficientSplit(dev_queries=len(split.dev), test_queries=len(split.test))

    fitting = sorted(split.dev, key=lambda q: q.query_id)[:DEV_TUNE_CAP]
    runs = [
        DevRun(
            query_id=q.query_id,
            rankings=tuple(runner.signals(q.question, q.vector, leg)),
            gold=q.gold,
        )
        for q in fitting
    ]
    tuning = tune_weights(runner, leg, runs)

    tuned_cell = CellAccumulator()
    equal_cell = CellAccumulator()
    for query in split.test:
        started = time.perf_counter()
        rankings = runner.signals(query.question, query.vector, leg)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        for weights, cell in ((tuning.weights, tuned_cell), (None, equal_cell)):
            outcome = runner.finish(
                query.question, rankings, leg, weights=weights, retrieval_ms=retrieval_ms
            )
            cell.add(
                outcome.ranked,
                query.gold,
                query.chain,
                outcome.latency_ms,
                tokens_packed=outcome.tokens_packed,
                budget=outcome.budget,
            )
    return TunedLeg(
        tuning=tuning,
        split=split.describe(len(runs)),
        tuned=tuned_cell.summary(),
        equal=equal_cell.summary(),
        dev_query_ids=tuple(q.query_id for q in split.dev),
        test_query_ids=tuple(q.query_id for q in split.test),
    )


def is_tunable_leg(leg: str) -> bool:
    """Whether ``leg`` fuses more than one list — i.e. whether its weights mean anything.

    ``dense`` and the ablation legs are single-signal, so their tuned cell is the equal-weight
    cell restricted to the test half: still worth writing (the weighted table needs its reference
    row on the same queries), but honestly labelled as a fit over nothing. One table lookup
    (:data:`~benchmark.s5.legs.LEG_TABLE`), declared per leg instead of re-derived here.
    """
    return LEG_TABLE[leg].tunable


__all__ = [
    "DEV_PERMILLE",
    "DEV_TUNE_CAP",
    "MAX_PASSES",
    "OBJECTIVE",
    "SPLIT_SALT",
    "WEIGHT_GRID",
    "DevRun",
    "InsufficientSplit",
    "ScoredQuery",
    "Split",
    "TunedLeg",
    "TuningResult",
    "dev_objective",
    "is_dev",
    "is_tunable_leg",
    "run_tuned_leg",
    "split_queries",
    "tune_weights",
]
