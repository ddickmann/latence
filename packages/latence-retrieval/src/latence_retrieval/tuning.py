# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
#
# DELIBERATE, BOUNDED EXCEPTION -- not a header nobody updated at the relicence. This
# repository's outbound licence is PolyForm-Noncommercial-1.0.0 (root LICENSE); this module is
# one of seven the copyright holder keeps under Apache-2.0 ON PURPOSE, so the query-time
# retrieval primitives stay permissively reusable -- commercially included -- by anyone who
# wants only them. The grant is his to make and safe to make: sole copyright holder, no
# third-party and no copyleft code in this file. The exact set is authoritative in LICENSE,
# section "SCOPE: THE RETAINED APACHE-2.0 MODULES", and pinned by
# packages/latence-core/tests/test_publish_readiness.py, which fails the build if an eighth
# file acquires the grant or one of the seven loses it.
"""Offline fusion tuning: evaluate a :class:`~latence_retrieval.fuser.Fuser`, sweep its weights.

**Why this module exists (T-FUSE-1).** The audit found that
:class:`~latence_retrieval.fuser.WeightedFuser` already implements the per-list normalised weighted
sum Bruch et al. (TOIS 2023) recommend over RRF — but that **every** call site in every harness is
``RrfFuser()`` with the default ``k=60`` and **uniform** weights (``run_benchmark.py:396``,
``build_questions.py:690``, ``rfc_bench.py:82``). No harness passed weights and no code tuned them,
so the stronger fuser was reachable in principle and unreachable in practice. The gap was wiring
and tooling, not the algorithm.

This module is the tooling half:

* :class:`FusionRun` — one labelled question: its gold document ids and the per-mode ranked lists
  the first stage returned for it. A harness records these once and reuses them for every fusion
  variant, so the comparison is over **the same candidate lists** and cannot be confounded by a
  different retrieval run.
* :func:`evaluate_fusion` — fuse each run and score the result (recall@k, MRR, nDCG@k).
* :func:`sweep_fusion_weights` — enumerate a weight grid and report every combination, best first.
  Bruch's own finding is that "just a handful of labeled queries" suffices, which is what makes
  this affordable at all.
* :func:`compare_fusers` / :func:`format_comparison_table` — the RRF-vs-tuned table the ticket asks
  for.

**It reports, it does not prefer.** RRF is not deprecated and is not disadvantaged here: it stays
the default and the no-labels fallback (there is nothing to tune without labels), and the table
prints whatever the measurement says. D6's own reading of the RRF paper — ``k = 60`` "was fixed
during a pilot investigation and not altered", with the paper's Table 1 optimum actually at 80 —
is a reason to *measure* k, not a reason to assume the weighted fuser wins.

Pure, deterministic and offline (ADR-0048): no randomness, no network, no state. The same runs and
the same grid always produce byte-identical rows.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from latence_retrieval.contracts import Candidate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from latence_retrieval.fuser import Fuser

_DEFAULT_MAX_COMBINATIONS = 4096


class FusionRun(BaseModel):
    """One labelled question and the per-mode ranked lists its first stage produced.

    ``rankings`` is keyed by **query mode name** (``"dense"``, ``"sparse"``, ``"bm25"``,
    ``"graph"``, …) exactly as
    :attr:`~latence_retrieval.orchestrator.OrchestrationPipeline.fuse_weights` is, so a weight
    tuned here transfers to the orchestrator verbatim. Recording the *lists* (not re-running
    retrieval per variant) is what makes the comparison honest: every fuser under test sees the
    identical candidate pool.

    Frozen and ``extra="forbid"`` like every contract in this library. ``gold_ids`` must be
    non-empty (a question with no gold document cannot score anything) and ``rankings`` must carry
    at least one list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(min_length=1)
    gold_ids: tuple[str, ...]
    rankings: dict[str, tuple[Candidate, ...]]

    @model_validator(mode="after")
    def _non_empty(self) -> FusionRun:
        if not self.gold_ids:
            msg = f"FusionRun {self.query_id!r} needs non-empty 'gold_ids' to be scoreable."
            raise ValueError(msg)
        if not self.rankings:
            msg = f"FusionRun {self.query_id!r} needs at least one entry in 'rankings'."
            raise ValueError(msg)
        return self


class FusionScore(BaseModel):
    """The retrieval quality of one fusion configuration over a labelled slice.

    ``recall_at_k`` is the mean per-query fraction of gold ids present in the fused top-``k``;
    ``mrr`` the mean reciprocal rank of the *first* gold hit (0 when none is retrieved); and
    ``ndcg_at_k`` the mean binary-gain nDCG@k against the ideal ranking of that query's gold set. A
    query with no gold hit scores zero on all three rather than being dropped — dropping it would
    silently select the queries the system happens to answer, which is precisely the shape D9 warns
    about.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: int
    k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class WeightSweepRow(BaseModel):
    """One weight combination and what it scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: dict[str, float]
    score: FusionScore


class WeightSweep(BaseModel):
    """The whole grid, best first — the artifact a harness reports rather than a single number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int
    rows: tuple[WeightSweepRow, ...]

    @property
    def best(self) -> WeightSweepRow:
        """The best-scoring combination (the first row; ties are broken deterministically)."""
        return self.rows[0]


class FuserComparisonRow(BaseModel):
    """One named fusion configuration and its score on the shared question set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    weights: dict[str, float] | None
    score: FusionScore


def evaluate_fusion(
    runs: Sequence[FusionRun],
    fuser: Fuser,
    *,
    weights: Mapping[str, float] | None = None,
    k: int = 10,
) -> FusionScore:
    """Fuse every run with ``fuser`` and score the fused lists at cut-off ``k``.

    ``weights`` is keyed by mode name; for each run the weight vector is built in **sorted mode
    order**, matching how the orchestrator aligns
    :attr:`~latence_retrieval.orchestrator.OrchestrationPipeline.fuse_weights` to its fetch legs,
    so a weight tuned here means the same thing there. A mode a given run does not carry is simply
    absent from that run's vector (never zero-padded into a list that does not exist), and a mode
    with no configured weight defaults to ``1.0``.
    """
    runs = list(runs)
    if not runs:
        msg = "evaluate_fusion needs at least one labelled FusionRun."
        raise ValueError(msg)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        msg = f"evaluate_fusion 'k' must be a positive integer, got {k!r}."
        raise ValueError(msg)

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    gains: list[float] = []
    for run in runs:
        modes = sorted(run.rankings)
        rankings = [list(run.rankings[mode]) for mode in modes]
        vector = (
            [float(weights.get(mode, 1.0)) for mode in modes] if weights is not None else None
        )
        fused = fuser.fuse(rankings, vector)
        top = [candidate.id for candidate in fused[:k]]
        gold = set(run.gold_ids)
        recalls.append(len(gold.intersection(top)) / len(gold))
        reciprocal_ranks.append(_reciprocal_rank(top, gold))
        gains.append(_ndcg(top, gold, k))
    count = len(runs)
    return FusionScore(
        queries=count,
        k=k,
        recall_at_k=sum(recalls) / count,
        mrr=sum(reciprocal_ranks) / count,
        ndcg_at_k=sum(gains) / count,
    )


def sweep_fusion_weights(
    runs: Sequence[FusionRun],
    *,
    grid: Mapping[str, Sequence[float]],
    k: int = 10,
    fuser: Fuser | None = None,
    max_combinations: int = _DEFAULT_MAX_COMBINATIONS,
) -> WeightSweep:
    """Score every weight combination in ``grid`` and return them best-first.

    ``grid`` maps a **mode name** to the candidate weights to try for it; the sweep is the cartesian
    product in sorted-mode order, so the enumeration (and therefore the tie-breaking) is
    deterministic. ``fuser`` defaults to a per-query-min-max
    :class:`~latence_retrieval.fuser.WeightedFuser` — the family being tuned — but any ``Fuser``
    works, including :class:`~latence_retrieval.fuser.RrfFuser` if a caller wants to check whether
    weighting RRF's per-list pull helps.

    Guards, so a sweep fails at the boundary rather than after an hour: a mode no run carries is a
    caller error (the weight would silently do nothing); an empty grid has nothing to sweep; and the
    product is capped by ``max_combinations`` so a five-mode ten-value grid does not quietly become
    100,000 evaluations.

    Rows are ordered by descending ``recall@k``, then ``nDCG@k``, then ``MRR``, then by the weight
    values in sorted-mode order — a total order, so the "best" row never depends on dict iteration.
    """
    runs = list(runs)
    if not runs:
        msg = "sweep_fusion_weights needs at least one labelled FusionRun."
        raise ValueError(msg)
    if not grid:
        msg = "sweep_fusion_weights needs a non-empty 'grid' of mode → candidate weights."
        raise ValueError(msg)

    known = {mode for run in runs for mode in run.rankings}
    unknown = sorted(set(grid) - known)
    if unknown:
        msg = (
            f"sweep_fusion_weights grid names mode(s) {unknown} that no FusionRun carries; "
            f"the runs carry {sorted(known)}. A weight for an absent list tunes nothing."
        )
        raise ValueError(msg)

    modes = sorted(grid)
    axes = [list(grid[mode]) for mode in modes]
    for mode, values in zip(modes, axes, strict=True):
        if not values:
            msg = f"sweep_fusion_weights grid[{mode!r}] is empty; give it at least one weight."
            raise ValueError(msg)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"sweep_fusion_weights grid[{mode!r}] must hold numbers, got {value!r}."
                raise ValueError(msg)
    total = math.prod(len(values) for values in axes)
    if total > max_combinations:
        msg = (
            f"sweep_fusion_weights would evaluate {total} weight combinations, above the "
            f"max_combinations cap of {max_combinations}. Coarsen the grid or raise the cap "
            "deliberately."
        )
        raise ValueError(msg)

    if fuser is None:
        from latence_retrieval.fuser import WeightedFuser

        fuser = WeightedFuser()

    rows: list[WeightSweepRow] = []
    for combination in itertools.product(*axes):
        weights = {mode: float(value) for mode, value in zip(modes, combination, strict=True)}
        score = evaluate_fusion(runs, fuser, weights=weights, k=k)
        rows.append(WeightSweepRow(weights=weights, score=score))
    rows.sort(
        key=lambda row: (
            -row.score.recall_at_k,
            -row.score.ndcg_at_k,
            -row.score.mrr,
            tuple(row.weights[mode] for mode in modes),
        )
    )
    return WeightSweep(k=k, rows=tuple(rows))


def compare_fusers(
    runs: Sequence[FusionRun],
    entries: Mapping[str, tuple[Fuser, Mapping[str, float] | None]],
    *,
    k: int = 10,
) -> tuple[FuserComparisonRow, ...]:
    """Score several named fusion configurations on **the same** questions and candidate lists.

    ``entries`` maps a display name to ``(fuser, weights_or_None)``; insertion order is preserved,
    so the caller controls the table's row order (typically the RRF baseline first). Because every
    entry fuses the same recorded :class:`FusionRun` lists, a difference between rows is a
    difference in *fusion* and nothing else.
    """
    return tuple(
        FuserComparisonRow(
            name=name,
            weights=dict(weights) if weights is not None else None,
            score=evaluate_fusion(runs, fuser, weights=weights, k=k),
        )
        for name, (fuser, weights) in entries.items()
    )


def format_comparison_table(rows: Sequence[FuserComparisonRow]) -> str:
    """Render :func:`compare_fusers` output as a fixed-width table (deterministic, no rounding lie).

    Metrics print to four decimals — enough to see a real difference, few enough not to imply a
    precision three questions cannot support.
    """
    rows = list(rows)
    if not rows:
        return ""
    k = rows[0].score.k
    headers = ("fuser", "weights", f"recall@{k}", f"ndcg@{k}", "mrr", "queries")
    body = [
        (
            row.name,
            _format_weights(row.weights),
            f"{row.score.recall_at_k:.4f}",
            f"{row.score.ndcg_at_k:.4f}",
            f"{row.score.mrr:.4f}",
            str(row.score.queries),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(line[column]) for line in body))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[column] for column in range(len(headers))),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(line)).rstrip()
        for line in body
    )
    return "\n".join(lines)


def _format_weights(weights: Mapping[str, float] | None) -> str:
    if weights is None:
        return "uniform"
    return " ".join(f"{mode}={weights[mode]:g}" for mode in sorted(weights))


def _reciprocal_rank(ranked_ids: Sequence[str], gold: set[str]) -> float:
    for position, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in gold:
            return 1.0 / position
    return 0.0


def _ndcg(ranked_ids: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-gain nDCG@k: DCG of the fused top-k over the DCG of the ideal ranking."""
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, candidate_id in enumerate(ranked_ids, start=1)
        if candidate_id in gold
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


__all__: list[str] = [
    "FuserComparisonRow",
    "FusionRun",
    "FusionScore",
    "WeightSweep",
    "WeightSweepRow",
    "compare_fusers",
    "evaluate_fusion",
    "format_comparison_table",
    "sweep_fusion_weights",
]
