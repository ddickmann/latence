"""PageIndexResolver — document-scoped offset→page resolution behind one narrow seam.

A :class:`~latence_core.contracts.PageMap` is **document**-scoped state: contiguous
spans covering a document's whole assembled markdown, invariant across every one of its
chunks. Before this module every downstream Stage learned that state the hard way —
each one had to know that :class:`~latence_core.pagemap.PageOffsetIndex` exists, that
``ChunkRecord.page_map`` may be ``None``, that the index must be rebuilt per chunk (or
cached by hand, keyed on ``document_record_id``), that a half-open span needs *two*
lookups with the ``max(end - 1, start)`` clamp, and that the ``None`` case degrades to
the chunk's inherited page range. Five call sites in core alone re-derived that
knowledge; exactly one of them remembered to cache. That is the definition of a missing
seam: the same complexity reappearing across N callers.

This module is that seam. A caller now knows **one** thing:

    resolver.pages_for_span(chunk, start, end)  ->  (page_start, page_end) | None

``None`` means "this document's page map is not available — degrade to whatever page
range you already hold", which is the identical fallback every call site hand-rolled.
Everything else — per-document memoisation of the underlying
:class:`~latence_core.pagemap.PageOffsetIndex`, absorbing the map from whichever record
carries it, the end-of-span clamp, and cross-document drift aggregation for the Quality
Report — lives here, once.

**Where the map comes from.** Two routes, in precedence order:

1. **Seeded** — :meth:`PageIndexResolver.from_documents` (or the ``page_maps`` mapping
   the constructor takes) builds the resolver from the parsed
   :class:`~latence_core.contracts.DocumentRecord`\\ s, which are the authoritative,
   already-persisted copy (ADR-0017: files are the source of truth). This is the route a
   Runner should use when it has the Documents checkpoint in hand.
2. **Absorbed from the stream** — :meth:`PageIndexResolver.observe` adopts the
   ``page_map`` a :class:`~latence_core.contracts.ChunkRecord` carries, if any, and
   memoises it. This is what makes the seam work for a Provider whose Capability
   signature is ``(chunks)`` and nothing else: the Chunk Stage stamps the map on a
   document's **first** chunk only (schema v18) and every later chunk of that document
   resolves from the memoised index.

   **A Stage MUST ``observe`` every chunk it is handed**, at the top of its loop, before
   it decides whether that chunk is interesting. A document's first chunk is very often
   the *least* interesting one — too short for the extractor's ``min_chars`` floor, no
   self-declaration to match, no PII to mask — and if the resolver only saw the chunks a
   Stage actually queried, the map would be skipped exactly when the document's first
   chunk produced nothing, silently degrading every later chunk of that document to its
   inherited page range. (A lookup also absorbs, as a backstop, so a forgotten
   ``observe`` costs completeness rather than correctness on the chunks it did see.)

   Route 2 also puts an invariant on the chunk **stream**: the map-carrying chunk must
   SURVIVE to the consumer. A Stage that removes chunks — Content Screening quarantines
   them — can drop exactly that one record and take the whole document's page resolution
   with it. :func:`rehome_page_maps` is where that invariant is kept, applied by the
   Runner at the point chunks leave the stream so every Screening Provider inherits it.

Route 2 is why the map no longer has to ride on all N chunks. For a 519-page document
chunked at 640 tokens that is ~30 KB of JSON on 1 record instead of ~399 — ~92% of every
chunk checkpoint line was the duplicated map, and every checkpoint write, streaming
read-back and decode paid for it.

**What it keeps alive (ADR-0033).** A resolver is built per Stage invocation and the Stage's
Provider is handed the WHOLE corpus stream in one call, so "memoise per document" meant, until
R3, "retain every document's page map for the length of the run": ~284 KB per 519-page document
(279 KB of :class:`~latence_core.contracts.PageMap` objects, 5 KB of index), i.e. ~284 MB at 1k
documents and ~2.8 GB at 10k — inside a Stage ADR-0033's *Honest memory boundary* declares
``O(checkpoint batch)`` on both sides. Absorbed maps are therefore held in a **bounded LRU
window** (:data:`DEFAULT_MAX_LIVE_DOCUMENTS`, overridable per resolver): chunks arrive grouped by
document, so only the documents a Provider holds *in flight* — at most its batch, 16 for every
shipped batching Provider — can need an index at once. Nothing is traded for the bound on a
document-grouped stream: eviction folds the evicted index's drift into the corpus roll-up, and a
chunk arriving for an already-evicted document is COUNTED and warned
(:attr:`PageIndexResolver.evicted_reentries`), never silently mis-paged (ADR-0034). Seeded maps
(route 1) are the caller's own materialised state and are pinned, not windowed.

**Never raises, never guesses silently.** Resolution degrades exactly as the ported
``PageOffsetIndex`` does (Provenance resolution must not be the thing that fails a run),
and a degradation is *counted* rather than swallowed: :attr:`PageIndexResolver.unresolved`
is the number of lookups that found no map, instead of the caller silently emitting an
inherited page range (ADR-0034: no silent errors — failures are observable). The count has
a consumer: the Quality Report's drift audit
(``quality_report_builder._DriftAggregator``) reads it and warns, counts-only, when chunks
went unaudited — a section that merely shrank would otherwise read as a confident
``clean``. The *preventable* half of the degradation is prevented rather than reported —
see :func:`rehome_page_maps`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from latence_core.pagemap import PageDriftDiagnostics, PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from latence_core.contracts import ChunkRecord, DocumentRecord, PageMap

logger = logging.getLogger("latence.provenance")

#: How many documents' page indices one resolver keeps live at once (ADR-0033).
#:
#: The stream a Stage is handed is grouped by document — the Chunk Stage emits a document's
#: chunks together and checkpoint order is preserved — so a resolver only needs an index for
#: the documents its caller holds IN FLIGHT. That is bounded by the Provider's batch: every
#: shipped batching Provider uses ``batch_size`` 16, and the widest shape (a whole batch
#: observed before any of it is resolved, as ``latence_extract_gliner2`` does) needs one index
#: per distinct document in that batch. 64 is 4× that, so no shipped stack can evict a document
#: it is still working on; a Provider that deliberately holds more must say so by constructing
#: the resolver with a larger ``max_live_documents``, and is TOLD to when it does not.
DEFAULT_MAX_LIVE_DOCUMENTS = 64


def rehome_page_maps(
    kept: Sequence[ChunkRecord], *, dropped_from: Iterable[ChunkRecord]
) -> list[ChunkRecord]:
    """Re-stamp each document's page map onto the first chunk of it that SURVIVED.

    Route 2 (above) rests on an invariant that is about the chunk **stream**, not about any
    one record: a document's map rides its first chunk, so that chunk must reach the
    consumer. Any Stage that REMOVES chunks can break it — Content Screening quarantines a
    chunk carrying a prompt-injection marker, and a document's opening chunk (title page,
    header block, a pasted instruction) is a likely place for one. Dropping it destroyed
    page resolution for the *whole* document: every later chunk resolved to ``None`` and
    every mention/relation/PII span in it degraded to its chunk's inherited page range.
    Before C1 the map rode all N chunks, so filtering could not do that — the invariant is
    what C1 introduced, and this is where it is kept.

    Call it at the point chunks leave the stream, with ``kept`` the survivors and
    ``dropped_from`` the stream they were selected out of (the Runner does this for every
    Content-Screening Provider, so a third-party screener inherits the guarantee). A
    document whose map-carrier survived, or which never had a map, is returned untouched —
    identical objects, so an unfiltered stream is byte-identical to its input.

    O(len(kept)) time and one map reference per document (the maps already exist in
    ``dropped_from``); it materialises nothing the caller was not already holding.
    """
    carried: dict[str, PageMap] = {}
    for chunk in dropped_from:
        if chunk.page_map is not None:
            carried.setdefault(chunk.document_record_id, chunk.page_map)
    if not carried:
        return list(kept)
    seen: set[str] = set()
    restamped: list[ChunkRecord] = []
    for chunk in kept:
        document_record_id = chunk.document_record_id
        if document_record_id in seen or chunk.page_map is not None:
            seen.add(document_record_id)
            restamped.append(chunk)
            continue
        seen.add(document_record_id)
        page_map = carried.get(document_record_id)
        restamped.append(
            chunk if page_map is None else chunk.model_copy(update={"page_map": page_map})
        )
    return restamped


class PageIndexResolver:
    """Memoised offset→page resolution for a whole corpus, keyed by document.

    One live resolver per Stage invocation. It holds at most one
    :class:`~latence_core.pagemap.PageOffsetIndex` per ``document_record_id``, built
    lazily on first use and reused for every subsequent lookup against that document —
    so a 400-chunk document builds the ``bisect`` array once, not 400 times.

    **Memoised, not accumulated.** The indices it absorbs from the stream live in an LRU
    window of ``max_live_documents`` documents; the ``PageMap`` an absorbed index was built
    from is reachable only from that index, so evicting the index frees the map with it. That
    is what keeps a doc-level Stage's page-map retention proportional to the documents its
    Provider holds in flight rather than to the corpus (ADR-0033) — see
    :data:`DEFAULT_MAX_LIVE_DOCUMENTS` for why the default cannot bite a shipped stack, and
    :attr:`evicted_reentries` for what happens, loudly, if it ever does. Maps handed in at
    construction (route 1) are the caller's own state and are pinned outside the window.

    Not a Pydantic model: like the indices it owns, this is live state with mutable
    diagnostics, distinct from the serialized :class:`~latence_core.contracts.PageMap`
    contracts it is built from.
    """

    __slots__ = (
        "_diag",
        "_evicted",
        "_evicted_reentries",
        "_indices",
        "_max_live",
        "_pinned",
        "_seen",
        "_unresolved",
        "_warned_reentry",
    )

    def __init__(
        self,
        page_maps: Mapping[str, PageMap] | None = None,
        *,
        max_live_documents: int = DEFAULT_MAX_LIVE_DOCUMENTS,
    ) -> None:
        """Seed the resolver with ``document_record_id -> PageMap`` (optional).

        An empty resolver is perfectly usable: it absorbs each document's map from the
        first record it sees that carries one.

        ``max_live_documents`` bounds how many ABSORBED document indices are kept live at
        once (LRU). Raise it only for a Provider that resolves against more distinct
        documents than it has observed most recently — a batch wider than the default.
        """
        if max_live_documents < 1:
            msg = f"max_live_documents must be >= 1, got {max_live_documents}"
            raise ValueError(msg)
        # Seeded maps are PINNED: the caller already materialised them (they came off
        # DocumentRecords it holds), route 1 promises resolution in any order, and dropping
        # them would free nothing the caller was not keeping anyway.
        self._pinned: dict[str, PageMap] = dict(page_maps) if page_maps else {}
        # LRU by last use. An absorbed PageMap is referenced ONLY by its index here, so
        # eviction releases the map (the ~279 KB of a 519-page document) and not just the
        # ~5 KB bisect array.
        self._indices: OrderedDict[str, PageOffsetIndex] = OrderedDict()
        self._max_live = max_live_documents
        # Drift for documents whose index has been discarded is folded in here as it is
        # produced, so the aggregate survives even if an index is never queried again.
        self._diag = PageDriftDiagnostics()
        self._unresolved = 0
        self._seen = len(self._pinned)
        # A bounded ring of the ids most recently evicted — enough to TELL a re-entry
        # (window too small) apart from a document that simply never had a map.
        self._evicted: OrderedDict[str, None] = OrderedDict()
        self._evicted_reentries = 0
        self._warned_reentry = False

    @classmethod
    def from_documents(cls, documents: Iterable[DocumentRecord]) -> PageIndexResolver:
        """Seed from parsed documents — the authoritative copy of each page map.

        Documents without a page map (a ``PARSE_ERROR`` record) are skipped; their
        chunks, if any, simply resolve to ``None`` and the caller degrades.
        """
        return cls({doc.record_id: doc.page_map for doc in documents if doc.page_map is not None})

    # -- ingestion ----------------------------------------------------------

    def observe(self, chunk: ChunkRecord) -> None:
        """Adopt the page map ``chunk`` carries, if any (idempotent, ~free when known).

        Call this for EVERY chunk a Stage is handed, before deciding whether the chunk is
        worth processing — see the module docstring for why skipping the uninteresting
        ones silently loses the document's map.
        """
        self._index_for(chunk)

    # -- resolution ---------------------------------------------------------

    def page_for(self, chunk: ChunkRecord, offset: int) -> int | None:
        """The 1-based source page of ``offset`` (a parent-document character offset).

        ``None`` when this chunk's document has no page map available — the caller
        degrades to the page range it already holds. Never raises: an out-of-range
        offset is recovered by the underlying index and recorded as drift.
        """
        index = self._index_for(chunk)
        if index is None:
            self._unresolved += 1
            return None
        return index.page_for_offset(offset)

    def pages_for_span(
        self, chunk: ChunkRecord, start: int, end: int
    ) -> tuple[int, int] | None:
        """The ``(page_start, page_end)`` a half-open ``[start, end)`` span falls on.

        ``end`` is **exclusive**, so ``page_end`` is resolved from the last *included*
        character (``max(end - 1, start)``) — a record straddling a page boundary
        therefore records both pages, and a zero-length span records one. Owning that
        clamp here is the point: it used to be re-derived, identically, at every call
        site. ``None`` when the document has no page map available.
        """
        index = self._index_for(chunk)
        if index is None:
            self._unresolved += 1
            return None
        return (
            index.page_for_offset(start),
            index.page_for_offset(max(end - 1, start)),
        )

    # -- diagnostics --------------------------------------------------------

    @property
    def diagnostics(self) -> PageDriftDiagnostics:
        """Drift aggregated across every document this resolver has resolved through.

        The corpus-level roll-up the per-chunk ``PageOffsetIndex`` construction made
        impossible: previously each throwaway index carried its own diagnostics and no
        caller summed them.
        """
        total = self._diag
        for index in self._indices.values():
            diag = index.diagnostics
            total = total.model_copy(
                update={
                    "resolutions": total.resolutions + diag.resolutions,
                    "recovered": total.recovered + diag.recovered,
                    "max_drift_chars": max(total.max_drift_chars, diag.max_drift_chars),
                }
            )
        return total

    @property
    def unresolved(self) -> int:
        """Lookups that found no page map for their document (the counted degradation)."""
        return self._unresolved

    @property
    def documents(self) -> int:
        """How many distinct documents this resolver has learned a page map for.

        Cumulative, not live: evicting a document's index (see
        :data:`DEFAULT_MAX_LIVE_DOCUMENTS`) does not un-learn it. :attr:`live_documents` is
        the retained window.
        """
        return self._seen

    @property
    def live_documents(self) -> int:
        """How many document indices are retained right now (``<= max_live_documents``)."""
        return len(self._indices)

    @property
    def evicted_reentries(self) -> int:
        """Lookups for a document whose index had been evicted and that carried no map.

        Non-zero means the caller resolves against more distinct documents than it keeps in
        flight most recently — i.e. ``max_live_documents`` is smaller than the caller's
        window — and those lookups degraded to the inherited page range. They are also
        counted in :attr:`unresolved`; this is the half that names the CAUSE, so the
        degradation is never silent (ADR-0034). The fix is the constructor argument.
        """
        return self._evicted_reentries

    # -- internals ----------------------------------------------------------

    def _index_for(self, chunk: ChunkRecord) -> PageOffsetIndex | None:
        """The memoised index for ``chunk``'s document, adopting a carried map if new."""
        document_record_id = chunk.document_record_id
        index = self._indices.get(document_record_id)
        if index is not None:
            self._indices.move_to_end(document_record_id)
            return index
        page_map = self._pinned.get(document_record_id)
        if page_map is None:
            # Absorb from the stream: the Chunk Stage stamps the map on a document's
            # first chunk (and a pre-v18 record carries it on every chunk), so the first
            # record of a document teaches the resolver and the rest resolve for free.
            carried = chunk.page_map
            if carried is None:
                if document_record_id in self._evicted:
                    self._note_reentry()
                return None
            page_map = carried
            # A re-absorbed document was already counted; the ring is what tells them apart.
            if document_record_id in self._evicted:
                del self._evicted[document_record_id]
            else:
                self._seen += 1
        else:
            # A pinned map can always be rebuilt into an index, so its id leaving the window
            # is not a re-entry to report when it comes back.
            self._evicted.pop(document_record_id, None)
        index = PageOffsetIndex(page_map)
        self._indices[document_record_id] = index
        self._evict_beyond_window()
        return index

    def _evict_beyond_window(self) -> None:
        """Drop least-recently-used document indices until the window fits."""
        while len(self._indices) > self._max_live:
            document_record_id, evicted = self._indices.popitem(last=False)
            diag = evicted.diagnostics
            # Fold the evicted index's drift into the running aggregate BEFORE it dies, so
            # the corpus-level roll-up the Quality Report's drift section reads (ADR-0008)
            # is exactly what an unbounded resolver would have produced.
            self._diag = self._diag.model_copy(
                update={
                    "resolutions": self._diag.resolutions + diag.resolutions,
                    "recovered": self._diag.recovered + diag.recovered,
                    "max_drift_chars": max(self._diag.max_drift_chars, diag.max_drift_chars),
                }
            )
            self._evicted[document_record_id] = None
            while len(self._evicted) > self._max_live:
                self._evicted.popitem(last=False)

    def _note_reentry(self) -> None:
        """Count — and, once, say — that the live window was too small for this caller."""
        self._evicted_reentries += 1
        if self._warned_reentry:
            return
        self._warned_reentry = True
        # Counts only: no record id, no document id, no content (ADR-0034).
        logger.warning(
            "page index: a chunk arrived for a document whose page index had already been "
            "evicted from the %d-document live window and carried no page map of its own; "
            "its page provenance is inherited, not resolved. Construct the "
            "PageIndexResolver with max_live_documents >= the number of documents this "
            "caller holds in flight. Further occurrences are counted in "
            "PageIndexResolver.evicted_reentries.",
            self._max_live,
        )
