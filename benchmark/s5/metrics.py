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
* **doc_recall@k** is the same set recall after both sides have been deduplicated to their source
  DOCUMENTS (:func:`documents_of`). It exists because the published comparison line is a document
  quantity — HippoRAG-2 reports passage recall@5 — and ten retrieved chunks cover far fewer than
  ten distinct documents, so a chunk-level ``recall@10`` set beside a passage ``recall@5`` is a
  metric mismatch, not a gap. It is reported *alongside* the chunk-level number, never instead of
  it, and only for cells whose caller supplied the document view.

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
    from collections.abc import Callable, Iterable, Sequence

#: The cut-offs every (dataset × leg) cell reports.
RECALL_KS = (1, 5, 10)
NDCG_K = 10
CHAIN_K = 10

#: The cut-offs the DOCUMENT-level recall reports. ``5`` is there because it is the cut-off the
#: de-facto comparison line publishes: HippoRAG-2 (arXiv:2502.14802) reports **passage** recall@5
#: (MuSiQue 74.7, HotpotQA 96.3). Our ``recall@k`` counts gold CHUNKS, and ten chunks cover far
#: fewer than ten distinct documents, so comparing the two is a metric mismatch rather than a
#: measurement of a gap. :func:`documents_of` deduplicates a ranked chunk list to its source
#: documents so the like-for-like number exists beside the chunk-level one instead of in a footnote.
DOCUMENT_RECALL_KS = (5, 10)


def recall_at_k(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    """``|gold ∩ top-k| / |gold|`` — 0.0 for empty gold (an unusable query, never scored)."""
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def documents_of(
    chunk_ids: Iterable[str], document_of: Callable[[str], str | None]
) -> list[str]:
    """``chunk_ids`` mapped to their source documents, deduplicated, **first occurrence wins**.

    Order matters: this is a *ranked* list on the way into ``recall@k``, so a document must take the
    position of its best-ranked chunk, not its last. A chunk with no known document is dropped
    rather than mapped to a placeholder — a placeholder would merge every unknown chunk into one
    fake document and inflate the number in exactly the direction nobody would notice.
    """
    out: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        document = document_of(chunk_id)
        if document is None or document in seen:
            continue
        seen.add(document)
        out.append(document)
    return out


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
    #: Document-level recall, collected only for queries the caller supplied a document view of.
    #: An empty dict therefore means "not measured", never "measured as zero" — which is why
    #: :meth:`summary` omits the keys entirely rather than printing ``0.0``.
    document_recall: dict[int, list[float]] = field(
        default_factory=lambda: {k: [] for k in DOCUMENT_RECALL_KS}
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
        ranked_documents: Sequence[str] | None = None,
        gold_documents: frozenset[str] | None = None,
    ) -> None:
        """Score one query into the cell.

        ``ranked_documents``/``gold_documents`` are the same query seen at DOCUMENT granularity
        (:func:`documents_of`). Supply both or neither: one without the other would score a real
        ranking against an empty gold set and report the result as a zero.
        """
        self.queries += 1
        for k in RECALL_KS:
            self.recall[k].append(recall_at_k(ranked, gold, k))
        if ranked_documents is not None and gold_documents is not None:
            for k in DOCUMENT_RECALL_KS:
                self.document_recall[k].append(recall_at_k(ranked_documents, gold_documents, k))
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
        for k in DOCUMENT_RECALL_KS:
            if self.document_recall[k]:
                out[f"doc_recall@{k}"] = _mean(self.document_recall[k])
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
    "DOCUMENT_RECALL_KS",
    "NDCG_K",
    "RECALL_KS",
    "CellAccumulator",
    "chain_metrics",
    "documents_of",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
]
