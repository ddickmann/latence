# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The S5 metrics: recall@k, nDCG@10, gold-chain coverage, latency — set-gold, not single-gold.

Every metric here treats gold as a **set of chunk ids** rather than one relevant document,
because that is what the datasets actually provide once resolution has run: a 2Wiki question has
two supporting paragraphs, a ViDoSeek question one page's worth of chunks, a MultiHop-RAG
question up to four articles. Two consequences the definitions make explicit rather than fudge:

* **recall@k** is ``|gold ∩ top-k| / |gold|`` — the standard set recall, so a query whose page
  split into three chunks is not counted as "solved" by retrieving one of them. The published
  ViDoSeek/OHR-Bench baselines score at page level, which is the same quantity when a page is
  one chunk and a strictly harder one when it is not; the report says so rather than quietly
  switching to a hit-rate.
* **nDCG@10** uses binary gains with the ideal ranking capped at ``min(|gold|, 10)``, so a query
  with 30 gold chunks cannot score below one with 2 purely for having more gold.

The chain metrics only apply where the dataset publishes an ordered/grouped chain (2Wiki's
``evidences``, MultiHop-RAG's multi-document ``evidence_list``): ``chain_coverage`` is the
fraction of the chain's slots with at least one chunk in the top-k, and ``full_chain_hit`` is 1
only when **every** slot is covered — the metric that actually distinguishes a multi-hop
retriever from a lucky single-hop one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: The cut-offs every (dataset × leg) cell reports.
RECALL_KS = (1, 5, 10)
NDCG_K = 10
CHAIN_K = 10


def recall_at_k(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    """``|gold ∩ top-k| / |gold|`` — 0.0 for empty gold (an unusable query, never scored)."""
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def ndcg_at_k(ranked: Sequence[str], gold: frozenset[str], k: int = NDCG_K) -> float:
    """Binary-gain nDCG@k with the ideal ranking capped at ``min(|gold|, k)``."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, chunk_id in enumerate(ranked[:k])
        if chunk_id in gold
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def chain_metrics(
    ranked: Sequence[str], chain: Sequence[frozenset[str]], k: int = CHAIN_K
) -> tuple[float, float] | None:
    """``(coverage, full_hit)`` over a gold chain's slots, or ``None`` when there is no chain."""
    if len(chain) < 2:
        return None
    top = set(ranked[:k])
    covered = sum(1 for slot in chain if slot & top)
    return (covered / len(chain), 1.0 if covered == len(chain) else 0.0)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of ``values`` (0.0 for an empty sequence)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass
class CellAccumulator:
    """Running metrics for one (dataset × leg) cell — one ``add`` per scored query."""

    recall: dict[int, list[float]] = field(
        default_factory=lambda: {k: [] for k in RECALL_KS}
    )
    ndcg: list[float] = field(default_factory=list)
    chain_coverage: list[float] = field(default_factory=list)
    full_chain: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    tokens_packed: list[int] = field(default_factory=list)
    budget: int | None = None
    queries: int = 0

    def add(
        self,
        ranked: Sequence[str],
        gold: frozenset[str],
        chain: Sequence[frozenset[str]],
        latency_ms: float,
        *,
        tokens_packed: int | None = None,
        budget: int | None = None,
    ) -> None:
        """Score one query into the cell."""
        self.queries += 1
        for k in RECALL_KS:
            self.recall[k].append(recall_at_k(ranked, gold, k))
        self.ndcg.append(ndcg_at_k(ranked, gold))
        chained = chain_metrics(ranked, chain)
        if chained is not None:
            coverage, full_hit = chained
            self.chain_coverage.append(coverage)
            self.full_chain.append(full_hit)
        self.latency_ms.append(latency_ms)
        if tokens_packed is not None:
            self.tokens_packed.append(tokens_packed)
        if budget is not None:
            self.budget = budget

    def summary(self) -> dict[str, float | int | None]:
        """The cell's JSON-serialisable metric row."""
        out: dict[str, float | int | None] = {"queries": self.queries}
        for k in RECALL_KS:
            out[f"recall@{k}"] = _mean(self.recall[k])
        out[f"ndcg@{NDCG_K}"] = _mean(self.ndcg)
        out["chain_coverage@10"] = _mean(self.chain_coverage) if self.chain_coverage else None
        out["full_chain_hit@10"] = _mean(self.full_chain) if self.full_chain else None
        out["latency_ms_median"] = round(percentile(self.latency_ms, 0.5), 2)
        out["latency_ms_p95"] = round(percentile(self.latency_ms, 0.95), 2)
        out["chain_queries"] = len(self.chain_coverage)
        if self.tokens_packed:
            out["tokens_packed_mean"] = round(
                sum(self.tokens_packed) / len(self.tokens_packed), 1
            )
            out["tokens_packed_max"] = max(self.tokens_packed)
        out["token_budget"] = self.budget
        return out


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


__all__ = [
    "CHAIN_K",
    "NDCG_K",
    "RECALL_KS",
    "CellAccumulator",
    "chain_metrics",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
]
