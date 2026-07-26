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
2. **Absorbed from the stream** — a lookup whose document is not yet known will adopt
   the ``page_map`` the passed :class:`~latence_core.contracts.ChunkRecord` carries, if
   any, and memoise it. This is what makes the seam work for a Provider whose Capability
   signature is ``(chunks)`` and nothing else: the Chunk Stage stamps the map on a
   document's **first** chunk only (schema v18) and every later chunk of that document
   resolves from the memoised index.

Route 2 is why the map no longer has to ride on all N chunks. For a 519-page document
chunked at 640 tokens that is ~30 KB of JSON on 1 record instead of ~399 — ~92% of every
chunk checkpoint line was the duplicated map, and every checkpoint write, streaming
read-back and decode paid for it.

**Never raises, never guesses silently.** Resolution degrades exactly as the ported
``PageOffsetIndex`` does (Provenance resolution must not be the thing that fails a run),
and a degradation is *counted* rather than swallowed: :attr:`PageIndexResolver.unresolved`
is the number of lookups that found no map, so a Quality Report can surface the gap
instead of the caller silently emitting an inherited page range (ADR-0034: no silent
errors — failures are observable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.pagemap import PageDriftDiagnostics, PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from latence_core.contracts import ChunkRecord, DocumentRecord, PageMap


class PageIndexResolver:
    """Memoised offset→page resolution for a whole corpus, keyed by document.

    One live resolver per Stage invocation. It holds at most one
    :class:`~latence_core.pagemap.PageOffsetIndex` per ``document_record_id``, built
    lazily on first use and reused for every subsequent lookup against that document —
    so a 400-chunk document builds the ``bisect`` array once, not 400 times.

    Not a Pydantic model: like the indices it owns, this is live state with mutable
    diagnostics, distinct from the serialized :class:`~latence_core.contracts.PageMap`
    contracts it is built from.
    """

    __slots__ = ("_diag", "_indices", "_page_maps", "_unresolved")

    def __init__(self, page_maps: Mapping[str, PageMap] | None = None) -> None:
        """Seed the resolver with ``document_record_id -> PageMap`` (optional).

        An empty resolver is perfectly usable: it absorbs each document's map from the
        first record it sees that carries one.
        """
        self._page_maps: dict[str, PageMap] = dict(page_maps) if page_maps else {}
        self._indices: dict[str, PageOffsetIndex] = {}
        # Drift for documents whose index has been discarded is folded in here as it is
        # produced, so the aggregate survives even if an index is never queried again.
        self._diag = PageDriftDiagnostics()
        self._unresolved = 0

    @classmethod
    def from_documents(cls, documents: Iterable[DocumentRecord]) -> PageIndexResolver:
        """Seed from parsed documents — the authoritative copy of each page map.

        Documents without a page map (a ``PARSE_ERROR`` record) are skipped; their
        chunks, if any, simply resolve to ``None`` and the caller degrades.
        """
        return cls({doc.record_id: doc.page_map for doc in documents if doc.page_map is not None})

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

    def resolves(self, chunk: ChunkRecord) -> bool:
        """Whether this chunk's document has a page map (no lookup, no drift recorded).

        For the callers whose fallback is not "inherit a page range" but a genuinely
        different code path (a redaction plan that must decide *before* detection whether
        page resolution is available at all).
        """
        return self._index_for(chunk) is not None

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
        """How many distinct documents this resolver holds a page map for."""
        return len(self._page_maps)

    # -- internals ----------------------------------------------------------

    def _index_for(self, chunk: ChunkRecord) -> PageOffsetIndex | None:
        """The memoised index for ``chunk``'s document, adopting a carried map if new."""
        document_record_id = chunk.document_record_id
        index = self._indices.get(document_record_id)
        if index is not None:
            return index
        page_map = self._page_maps.get(document_record_id)
        if page_map is None:
            # Absorb from the stream: the Chunk Stage stamps the map on a document's
            # first chunk (and a pre-v18 record carries it on every chunk), so the first
            # record of a document teaches the resolver and the rest resolve for free.
            carried = chunk.page_map
            if carried is None:
                return None
            page_map = carried
            self._page_maps[document_record_id] = carried
        index = PageOffsetIndex(page_map)
        self._indices[document_record_id] = index
        return index
