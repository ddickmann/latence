# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Gold→chunk resolution: the single biggest correctness risk in the S5 harness, isolated.

Every dataset in the S5 matrix names its gold evidence differently — 2Wiki by
``(title, sentence_index)``, MultiHop-RAG by article title/url, OHR-Bench by
``doc_name + page_idx``, ViDoSeek by ``file + reference_page``, UDA by ``doc_name`` (+ a page
inside the ``q_uid`` for the financial half), MuSiQue/HotpotQA by a supporting-paragraph file.
The retrieval matrix scores *chunk ids*, so every one of those references must be mapped onto
the ids the S3 export actually emitted. **A wrong mapping produces a plausible-looking recall
number over gold that does not exist** — silent, and indistinguishable from a weak retriever. So
the mapping lives here: a pure module with no I/O, one narrow index protocol, and a resolution
*rate* reported per dataset so a mapping that quietly stops working is visible in the output
rather than folded into a metric.

Two reference kinds cover all seven datasets:

* :attr:`RefKind.PAGE` — a page of a paginated document. Resolved through the page char spans
  reconstructed from the S2 per-page artifacts (:mod:`benchmark.s5.pagemap`): a chunk resolves
  when its ``[char_start, char_end)`` provenance span **overlaps** the page's span.
* :attr:`RefKind.FILE` — the whole source document is the gold unit (MultiHop-RAG articles,
  2Wiki paragraphs, MuSiQue/HotpotQA supporting paragraphs, UDA papers — each is the published
  gold unit, not a weakening of it).

Two properties keep the resulting numbers honest, and both are recorded per reference rather
than assumed uniform across a dataset:

**Granularity.** A ``PAGE`` reference whose document has no reconstructable page map may, when
the loader marks it :attr:`GoldRef.allow_document_fallback`, resolve at :attr:`Granularity.
DOCUMENT` instead — coarser gold, but real gold. Silently mixing the two would let a report imply
page precision it does not have, so every resolved reference records which granularity it
actually used and :class:`ResolutionReport` carries the mix. The fallback fires **only** when the
index has no page map at all for that document; a page map that exists but does not contain the
referenced page is a genuine miss and stays unresolved.

**Exclusions.** A query whose evidence document was never acquired into this corpus cannot be
retrieved by any retriever, and counting it as a miss would understate recall and read as a
retrieval failure when it is a corpus-coverage fact. Such a query carries an
:class:`ExclusionReason`; it is dropped from both the resolution denominator and the scored set,
and the reason plus the count is reported. This is the *only* sanctioned way for a query to leave
the denominator — an ordinary reference that fails to resolve is **unresolved** and counted as
such, never silently an empty gold set that every retriever "misses".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class RefKind(StrEnum):
    """How a gold reference names its evidence — see the module docstring."""

    PAGE = "page"
    FILE = "file"


class Granularity(StrEnum):
    """The precision a reference *actually* resolved at — recorded, never assumed."""

    PAGE = "page"
    DOCUMENT = "document"


class ExclusionReason(StrEnum):
    """Why a query is unanswerable against this corpus and so leaves the denominator.

    Both values are corpus-coverage facts measured against the S3 export, not judgements about
    a query: :attr:`CORPUS_MISSING_DOMAIN` for a whole domain that was never acquired (OHR-Bench
    ``notes/``, the GNHK handwriting set), :attr:`CORPUS_MISSING_DOCUMENT` for an individual
    document that is absent.
    """

    CORPUS_MISSING_DOMAIN = "corpus-missing-domain"
    CORPUS_MISSING_DOCUMENT = "corpus-missing-document"


@dataclass(frozen=True)
class GoldRef:
    """One gold evidence reference, already normalised onto the export's naming.

    ``file_name`` is the export's ``provenance.file_name`` (the S3 input markdown's name), so
    the dataset-specific naming (a ``.pdf`` basename, a ``domain/doc`` pair, an article title)
    has already been translated by the dataset loader. ``page`` is **1-based** and meaningful
    only for :attr:`RefKind.PAGE`. ``slot`` is the position of this reference in the query's gold
    *chain* (0 for single-hop datasets and for datasets whose gold is an unordered set) — chain
    metrics group by it. ``allow_document_fallback`` says the whole document is acceptable
    (coarser) gold when the page spans cannot be reconstructed; it is set by the loader for the
    datasets where the page is a refinement of a document-level reference that is itself
    certain, and left off where a page miss must stay a miss.
    """

    kind: RefKind
    file_name: str
    page: int | None = None
    slot: int = 0
    allow_document_fallback: bool = False


@dataclass(frozen=True)
class GoldQuery:
    """One benchmark query plus its gold references, before resolution.

    ``exclusion`` marks a query whose evidence is not in this corpus at all (see
    :class:`ExclusionReason`). Its references are kept for auditability but are never resolved
    and never counted against the resolution rate.
    """

    query_id: str
    question: str
    refs: tuple[GoldRef, ...]
    question_type: str | None = None
    exclusion: ExclusionReason | None = None


@dataclass(frozen=True)
class ResolvedGold:
    """A query whose gold references have been mapped onto export chunk ids.

    ``chunk_ids`` is the union over all resolved references — the set recall@k scores against.
    ``chain`` holds one chunk-id set per **slot** that resolved to at least one chunk, in slot
    order; a query with fewer than two occupied slots has no chain and contributes to neither
    chain metric. ``resolved_refs`` / ``total_refs`` are the per-query resolution counters the
    harness aggregates into the per-dataset resolution rate, and ``page_refs`` / ``document_refs``
    split the resolved ones by the granularity they actually landed at.
    """

    query_id: str
    question: str
    chunk_ids: frozenset[str]
    chain: tuple[frozenset[str], ...]
    resolved_refs: int
    total_refs: int
    question_type: str | None = None
    page_refs: int = 0
    document_refs: int = 0
    exclusion: ExclusionReason | None = None
    excluded_refs: int = 0

    @property
    def is_usable(self) -> bool:
        """Whether this query can be scored at all — it resolved to at least one gold chunk."""
        return bool(self.chunk_ids)


@runtime_checkable
class ChunkIndex(Protocol):
    """The narrow, read-only view of an export that gold resolution needs.

    Three lookups, each deterministic and each returning chunk ids in export order, plus the
    one predicate the document-level fallback turns on. Keeping them a Protocol is what makes
    :func:`resolve` a pure function that a unit test can drive with a hand-written index of five
    chunks — no parquet, no corpus, no model.
    """

    def chunks_in_file(self, file_name: str) -> Sequence[str]:
        """Every chunk id emitted from ``file_name`` (empty when the file is not in the export)."""
        ...

    def chunks_on_page(self, file_name: str, page: int) -> Sequence[str]:
        """Chunk ids whose char span overlaps 1-based ``page`` of ``file_name``."""
        ...

    def has_page_map(self, file_name: str) -> bool:
        """Whether ``file_name``'s page char spans were reconstructed at all.

        The distinction the fallback rests on: ``False`` means *page-level gold is unavailable
        for this document* (its S2 artifacts are missing or its reconstruction did not reproduce
        the assembled markdown), whereas ``True`` with an empty
        :meth:`chunks_on_page` means *this page reference is wrong* — a miss, not a coarsening.
        """
        ...


@dataclass(frozen=True)
class RefResolution:
    """What one gold reference resolved to, and at which granularity.

    ``granularity`` is ``None`` exactly when ``chunk_ids`` is empty: an unresolved reference has
    no precision to report.
    """

    chunk_ids: tuple[str, ...]
    granularity: Granularity | None = None


def _document_level(ref: GoldRef, index: ChunkIndex) -> RefResolution:
    ids = tuple(index.chunks_in_file(ref.file_name))
    return RefResolution(ids, Granularity.DOCUMENT if ids else None)


def resolve_ref(ref: GoldRef, index: ChunkIndex) -> RefResolution:
    """The chunk ids one gold reference names, with the granularity it landed at.

    A ``PAGE`` reference resolves at page precision whenever the document has a reconstructed
    page map. Without one it falls back to the whole document **only** if the loader marked the
    reference :attr:`GoldRef.allow_document_fallback`; otherwise it stays unresolved, because
    silently widening a page reference to document level is the recall inflation this module
    exists to prevent.
    """
    if ref.kind is RefKind.PAGE:
        if ref.page is not None and index.has_page_map(ref.file_name):
            ids = tuple(index.chunks_on_page(ref.file_name, ref.page))
            return RefResolution(ids, Granularity.PAGE if ids else None)
        if not ref.allow_document_fallback:
            return RefResolution(())
        return _document_level(ref, index)
    return _document_level(ref, index)


def resolve(query: GoldQuery, index: ChunkIndex) -> ResolvedGold:
    """Map ``query``'s gold references onto export chunk ids, counting what did not resolve.

    Every reference is attempted; the resolved ones contribute their chunk ids to the union and
    to their chain slot. The counters are the honest part: a query where 1 of 3 references
    resolved is *scored* (its one gold chunk is real) but it drags the dataset's resolution
    rate down, which is exactly the signal that says "do not trust this dataset's recall yet".

    An **excluded** query short-circuits: its references are not attempted and contribute
    nothing to either the numerator or the denominator, so a domain this corpus never acquired
    reads as a coverage gap with a stated reason instead of a retrieval failure.
    """
    if query.exclusion is not None:
        return ResolvedGold(
            query_id=query.query_id,
            question=query.question,
            chunk_ids=frozenset(),
            chain=(),
            resolved_refs=0,
            total_refs=0,
            question_type=query.question_type,
            exclusion=query.exclusion,
            excluded_refs=len(query.refs),
        )
    union: set[str] = set()
    slots: dict[int, set[str]] = {}
    resolved = 0
    pages = 0
    documents = 0
    for ref in query.refs:
        outcome = resolve_ref(ref, index)
        if not outcome.chunk_ids:
            continue
        resolved += 1
        if outcome.granularity is Granularity.PAGE:
            pages += 1
        else:
            documents += 1
        union.update(outcome.chunk_ids)
        slots.setdefault(ref.slot, set()).update(outcome.chunk_ids)
    chain = tuple(frozenset(slots[slot]) for slot in sorted(slots)) if len(slots) > 1 else ()
    return ResolvedGold(
        query_id=query.query_id,
        question=query.question,
        chunk_ids=frozenset(union),
        chain=chain,
        resolved_refs=resolved,
        total_refs=len(query.refs),
        question_type=query.question_type,
        page_refs=pages,
        document_refs=documents,
    )


@dataclass(frozen=True)
class ResolutionReport:
    """Per-dataset gold-resolution health — the number that gates trusting a recall figure.

    ``rate`` is references-resolved / references-total over the **non-excluded** queries. Below
    :data:`TRUST_THRESHOLD` the harness flags the dataset loudly in ``metrics.json`` and the
    report instead of quietly printing a recall built on a broken mapping. ``refs_page`` /
    ``refs_document`` are the granularity mix, and ``exclusions`` maps each
    :class:`ExclusionReason` to the number of queries it removed — both stated so the report can
    describe the actual precision and coverage rather than imply uniform ones.
    """

    dataset: str
    queries_total: int
    queries_usable: int
    refs_total: int
    refs_resolved: int
    queries_excluded: int = 0
    refs_page: int = 0
    refs_document: int = 0
    refs_excluded: int = 0
    exclusions: Mapping[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        """Fraction of gold references that resolved to at least one export chunk."""
        return self.refs_resolved / self.refs_total if self.refs_total else 0.0

    @property
    def page_share(self) -> float:
        """Fraction of the *resolved* references that landed at page precision."""
        return self.refs_page / self.refs_resolved if self.refs_resolved else 0.0

    @property
    def trustworthy(self) -> bool:
        """Whether the resolution rate clears :data:`TRUST_THRESHOLD`."""
        return self.rate >= TRUST_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form for ``metrics.json`` and ``resolution/<dataset>.json``."""
        return {
            "dataset": self.dataset,
            "queries_total": self.queries_total,
            "queries_usable": self.queries_usable,
            "queries_excluded": self.queries_excluded,
            "refs_total": self.refs_total,
            "refs_resolved": self.refs_resolved,
            "refs_excluded": self.refs_excluded,
            "resolution_rate": round(self.rate, 4),
            "granularity": {"page": self.refs_page, "document": self.refs_document},
            "page_share_of_resolved": round(self.page_share, 4),
            "exclusions": dict(sorted(self.exclusions.items())),
            "trustworthy": self.trustworthy,
        }


#: Gold-resolution rate below which a dataset's retrieval numbers are FLAGGED, not reported as
#: fact (the ticket's ~90% bar). A rate under this means the mapping — not the retriever — is
#: what the recall number is measuring.
TRUST_THRESHOLD = 0.90


def report(dataset: str, resolved: Iterable[ResolvedGold]) -> ResolutionReport:
    """Aggregate per-query resolution counters into the dataset's :class:`ResolutionReport`."""
    items = list(resolved)
    exclusions: dict[str, int] = {}
    for item in items:
        if item.exclusion is not None:
            exclusions[str(item.exclusion)] = exclusions.get(str(item.exclusion), 0) + 1
    return ResolutionReport(
        dataset=dataset,
        queries_total=len(items),
        queries_usable=sum(1 for item in items if item.is_usable),
        refs_total=sum(item.total_refs for item in items),
        refs_resolved=sum(item.resolved_refs for item in items),
        queries_excluded=sum(1 for item in items if item.exclusion is not None),
        refs_page=sum(item.page_refs for item in items),
        refs_document=sum(item.document_refs for item in items),
        refs_excluded=sum(item.excluded_refs for item in items),
        exclusions=exclusions,
    )


__all__ = [
    "TRUST_THRESHOLD",
    "ChunkIndex",
    "ExclusionReason",
    "GoldQuery",
    "GoldRef",
    "Granularity",
    "RefKind",
    "RefResolution",
    "ResolutionReport",
    "ResolvedGold",
    "report",
    "resolve",
    "resolve_ref",
]
