# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Gold→chunk resolution: the single biggest correctness risk in the S5 harness, isolated.

Every dataset in the S5 matrix names its gold evidence differently — 2Wiki by
``(title, sentence_index)``, MultiHop-RAG by article title/url, OHR-Bench by
``doc_name + page_idx``, ViDoSeek by ``file + reference_page``, UDA by a literal evidence
sentence, MuSiQue/HotpotQA by a supporting-paragraph file. The retrieval matrix scores
*chunk ids*, so every one of those references must be mapped onto the ids the S3 export
actually emitted. **A wrong mapping produces a plausible-looking recall number over gold that
does not exist** — silent, and indistinguishable from a weak retriever. So the mapping lives
here: a pure module with no I/O, one narrow index protocol, and a resolution *rate* reported
per dataset so a mapping that quietly stops working is visible in the output rather than
folded into a metric.

Three reference kinds cover all seven datasets, in descending order of precision:

* :attr:`RefKind.PAGE` — a page of a paginated document. Resolved through the page char
  spans reconstructed from the S2 per-page artifacts (:mod:`benchmark.s5.pagemap`): a chunk
  resolves when its ``[char_start, char_end)`` provenance span **overlaps** the page's span.
* :attr:`RefKind.TEXT` — a literal evidence sentence inside a known document (UDA). Resolved
  by whitespace-normalised substring containment against the chunk text, restricted to that
  document's chunks.
* :attr:`RefKind.FILE` — the whole source document is the gold unit (MultiHop-RAG articles,
  2Wiki paragraphs, MuSiQue/HotpotQA supporting paragraphs — each is one short document, so
  document-level gold *is* the published unit, not a weakening of it).

A reference that resolves to zero chunks is **unresolved** and counted as such; it never
silently becomes an empty gold set that every retriever "misses".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class RefKind(StrEnum):
    """How a gold reference names its evidence — see the module docstring."""

    PAGE = "page"
    TEXT = "text"
    FILE = "file"


@dataclass(frozen=True)
class GoldRef:
    """One gold evidence reference, already normalised onto the export's naming.

    ``file_name`` is the export's ``provenance.file_name`` (the S3 input markdown's name), so
    the dataset-specific naming (a ``.pdf`` basename, a ``domain/doc`` pair, an article title)
    has already been translated by the dataset loader. ``page`` is **1-based** and meaningful
    only for :attr:`RefKind.PAGE`; ``text`` only for :attr:`RefKind.TEXT`. ``slot`` is the
    position of this reference in the query's gold *chain* (0 for single-hop datasets and for
    datasets whose gold is an unordered set) — chain metrics group by it.
    """

    kind: RefKind
    file_name: str
    page: int | None = None
    text: str | None = None
    slot: int = 0


@dataclass(frozen=True)
class GoldQuery:
    """One benchmark query plus its gold references, before resolution."""

    query_id: str
    question: str
    refs: tuple[GoldRef, ...]
    question_type: str | None = None


@dataclass(frozen=True)
class ResolvedGold:
    """A query whose gold references have been mapped onto export chunk ids.

    ``chunk_ids`` is the union over all resolved references — the set recall@k scores against.
    ``chain`` holds one chunk-id set per **slot** that resolved to at least one chunk, in slot
    order; a query with fewer than two occupied slots has no chain and contributes to neither
    chain metric. ``resolved_refs`` / ``total_refs`` are the per-query resolution counters the
    harness aggregates into the per-dataset resolution rate.
    """

    query_id: str
    question: str
    chunk_ids: frozenset[str]
    chain: tuple[frozenset[str], ...]
    resolved_refs: int
    total_refs: int
    question_type: str | None = None

    @property
    def is_usable(self) -> bool:
        """Whether this query can be scored at all — it resolved to at least one gold chunk."""
        return bool(self.chunk_ids)


@runtime_checkable
class ChunkIndex(Protocol):
    """The narrow, read-only view of an export that gold resolution needs.

    Three lookups, each deterministic and each returning chunk ids in export order. Keeping
    them a Protocol is what makes :func:`resolve` a pure function that a unit test can drive
    with a hand-written index of five chunks — no parquet, no corpus, no model.
    """

    def chunks_in_file(self, file_name: str) -> Sequence[str]:
        """Every chunk id emitted from ``file_name`` (empty when the file is not in the export)."""
        ...

    def chunks_on_page(self, file_name: str, page: int) -> Sequence[str]:
        """Chunk ids whose char span overlaps 1-based ``page`` of ``file_name``."""
        ...

    def chunks_containing(self, file_name: str, text: str) -> Sequence[str]:
        """Chunk ids in ``file_name`` whose text contains ``text`` (whitespace-normalised)."""
        ...


def resolve_ref(ref: GoldRef, index: ChunkIndex) -> tuple[str, ...]:
    """The chunk ids one gold reference names — empty when it does not resolve."""
    if ref.kind is RefKind.PAGE:
        if ref.page is None:
            return ()
        return tuple(index.chunks_on_page(ref.file_name, ref.page))
    if ref.kind is RefKind.TEXT:
        if not ref.text:
            return ()
        return tuple(index.chunks_containing(ref.file_name, ref.text))
    return tuple(index.chunks_in_file(ref.file_name))


def resolve(query: GoldQuery, index: ChunkIndex) -> ResolvedGold:
    """Map ``query``'s gold references onto export chunk ids, counting what did not resolve.

    Every reference is attempted; the resolved ones contribute their chunk ids to the union and
    to their chain slot. The counters are the honest part: a query where 1 of 3 references
    resolved is *scored* (its one gold chunk is real) but it drags the dataset's resolution
    rate down, which is exactly the signal that says "do not trust this dataset's recall yet".
    """
    union: set[str] = set()
    slots: dict[int, set[str]] = {}
    resolved = 0
    for ref in query.refs:
        ids = resolve_ref(ref, index)
        if ids:
            resolved += 1
            union.update(ids)
            slots.setdefault(ref.slot, set()).update(ids)
    chain = tuple(frozenset(slots[slot]) for slot in sorted(slots)) if len(slots) > 1 else ()
    return ResolvedGold(
        query_id=query.query_id,
        question=query.question,
        chunk_ids=frozenset(union),
        chain=chain,
        resolved_refs=resolved,
        total_refs=len(query.refs),
        question_type=query.question_type,
    )


@dataclass(frozen=True)
class ResolutionReport:
    """Per-dataset gold-resolution health — the number that gates trusting a recall figure.

    ``rate`` is references-resolved / references-total. Below :data:`TRUST_THRESHOLD` the
    harness flags the dataset loudly in ``metrics.json`` and the report instead of quietly
    printing a recall built on a broken mapping.
    """

    dataset: str
    queries_total: int
    queries_usable: int
    refs_total: int
    refs_resolved: int

    @property
    def rate(self) -> float:
        """Fraction of gold references that resolved to at least one export chunk."""
        return self.refs_resolved / self.refs_total if self.refs_total else 0.0

    @property
    def trustworthy(self) -> bool:
        """Whether the resolution rate clears :data:`TRUST_THRESHOLD`."""
        return self.rate >= TRUST_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for ``metrics.json``."""
        return {
            "dataset": self.dataset,
            "queries_total": self.queries_total,
            "queries_usable": self.queries_usable,
            "refs_total": self.refs_total,
            "refs_resolved": self.refs_resolved,
            "resolution_rate": round(self.rate, 4),
            "trustworthy": self.trustworthy,
        }


#: Gold-resolution rate below which a dataset's retrieval numbers are FLAGGED, not reported as
#: fact (the ticket's ~90% bar). A rate under this means the mapping — not the retriever — is
#: what the recall number is measuring.
TRUST_THRESHOLD = 0.90


def report(dataset: str, resolved: Iterable[ResolvedGold]) -> ResolutionReport:
    """Aggregate per-query resolution counters into the dataset's :class:`ResolutionReport`."""
    items = list(resolved)
    return ResolutionReport(
        dataset=dataset,
        queries_total=len(items),
        queries_usable=sum(1 for item in items if item.is_usable),
        refs_total=sum(item.total_refs for item in items),
        refs_resolved=sum(item.resolved_refs for item in items),
    )


__all__ = [
    "TRUST_THRESHOLD",
    "ChunkIndex",
    "GoldQuery",
    "GoldRef",
    "RefKind",
    "ResolutionReport",
    "ResolvedGold",
    "report",
    "resolve",
    "resolve_ref",
]
