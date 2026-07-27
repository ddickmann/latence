"""PageIndexResolver — STATELESS offset→page resolution from a self-describing chunk.

A chunk needs to answer exactly one page question: **which source page does an offset
INSIDE me fall on?** An entity mention, a PII span, a citation — each is found at some
offset within a chunk's text, and the record that cites it must name the page that offset
really came from, not merely the chunk's whole ``[page_start, page_end]`` range
(ADR-0021/0031).

Since v19 the chunk answers that itself. The Chunk Stage stamps a
:class:`~latence_core.contracts.PageSlice` on every chunk — exactly the document page spans
overlapping that chunk's own ``[char_start, char_end)``, in the document's original
coordinates. Typically one span; two when the chunk straddles a page break. This module is
the seam that reads it, and a caller knows one thing:

    resolver.pages_for_span(chunk, start, end)  ->  (page_start, page_end)

Everything else — the half-open end-of-span clamp, and the corpus-level drift roll-up the
Quality Report reads — lives here, once. It used to be re-derived, identically, at every
call site.

**Why stateless, and what was deleted.** Until v19 the page map was DOCUMENT-scoped state
that rode ONE record of the chunk stream (the document's first chunk), and this resolver
adopted it from whichever record it saw first, memoised per ``document_record_id``. That
made page resolution depend on two things that are not properties of the chunk being
resolved:

* **Order.** A Provider that resolves in any order other than stream order — every
  batching extractor does — asked for a page for chunk *N* before the resolver had seen
  chunk 0, and got ``None``.
* **Survival.** Any Stage that REMOVES chunks could remove the map-carrying one and take
  the whole document's page resolution with it.

Both degraded SILENTLY to the chunk's inherited page range: a plausible-looking wrong page,
indistinguishable from a correct one in the output. Four audit rounds patched the
consequences — ``rehome_page_maps()`` to re-home the map past a filtering Stage, then an LRU
retention window to bound what memoisation kept alive, then a fix to that window's own
justification — and never the shape. Each layer made the wrong shape more robust.

A slice removes the dependency by removing the shared state. There is no per-document
memoisation, no absorption from the stream, no retention window and no ``observe`` — so
there is nothing for an ordering or a dropped record to be wrong about. The failure mode is
not mitigated; it is **inexpressible**. What is kept is the payload win that shape was
bought for: ~2 spans (~60 bytes) per chunk against the ~30 KB whole-document map a 519-page
document would otherwise duplicate onto ~399 records — ~99.8% of it, with no run-scoped
sidecar and no Runner involvement, so ADR-0033's memory boundary is untouched.

**No silent errors (ADR-0034).** A chunk carrying no slice raises
:class:`~latence_core.errors.PageSliceMissingError`. It does not return ``None`` for the
caller to degrade on, because "degrade" meant emitting a page that looks right and is not.

**Cross-chunk spans are not this seam's job.** A relation's covering span runs from its head
mention to its tail mention and those may live in different chunks, so no single chunk can
answer for it. It does not need to: both endpoints already carry their own exactly-resolved
pages, and the covering span's pages are their union. That is what
:func:`pages_for_covering_span` computes — arithmetic on two records, no index, no lookup.
The previous code instead resolved the covering span against an arbitrary "anchor" chunk of
the document, which only worked because the anchor happened to carry the WHOLE document's
map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.errors import PageSliceMissingError
from latence_core.pagemap import PageDriftDiagnostics, PageOffsetIndex

if TYPE_CHECKING:
    from latence_core.contracts import ChunkRecord, Provenance


def pages_for_covering_span(
    head: Provenance, tail: Provenance
) -> tuple[int, int] | None:
    """The ``(page_start, page_end)`` of the span covering two already-resolved records.

    A relation's covering span is ``[min(starts), max(ends))`` over its two endpoint
    mentions. Because pages increase monotonically with offset, the page of ``min(starts)``
    is simply the earlier endpoint's own ``page_start``, and the page of ``max(ends) - 1``
    is the later endpoint's own ``page_end`` — both of which were resolved exactly, from
    their own chunk's slice, when those mentions were emitted. So the answer is a ``min``
    and a ``max``, not a lookup, and it is correct even when the endpoints are in different
    chunks (which an anchor-chunk lookup could not be).

    ``None`` only when neither endpoint recorded a page at all, which leaves the caller
    nothing to be right or wrong with.
    """
    starts = [p for p in (head.page_start, tail.page_start) if p is not None]
    ends = [p for p in (head.page_end, tail.page_end) if p is not None]
    if not starts and not ends:
        return None
    page_start = min(starts) if starts else min(ends)
    page_end = max(ends) if ends else max(starts)
    return page_start, page_end


class PageIndexResolver:
    """Offset→page resolution over a chunk's own page spans. Holds no document state.

    Construct one per Stage invocation and hand it every chunk you resolve against; it
    carries nothing between calls except the drift ROLL-UP (which aggregates per-resolution
    *results*, and so cannot make one chunk's answer depend on another's). Resolving the
    same chunk twice, resolving chunks in any order, and resolving a chunk whose siblings
    were never seen are all identical operations.

    Not a Pydantic model: like the index it builds, this is live state with mutable
    diagnostics, distinct from the serialized :class:`~latence_core.contracts.PageSlice`
    contracts it reads.
    """

    __slots__ = ("_diag",)

    def __init__(self) -> None:
        self._diag = PageDriftDiagnostics()

    # -- resolution ---------------------------------------------------------

    def page_for(self, chunk: ChunkRecord, offset: int) -> int:
        """The 1-based source page of ``offset`` (a parent-document character offset).

        Raises :class:`~latence_core.errors.PageSliceMissingError` if the chunk carries no
        page spans. An out-of-range offset never raises — it is recovered by the underlying
        index and recorded as drift, because Provenance resolution must not be the thing
        that fails a run.
        """
        index = self._index_for(chunk)
        page = index.page_for_offset(offset)
        self._fold(index.diagnostics)
        return page

    def pages_for_span(self, chunk: ChunkRecord, start: int, end: int) -> tuple[int, int]:
        """The ``(page_start, page_end)`` a half-open ``[start, end)`` span falls on.

        ``end`` is **exclusive**, so ``page_end`` is resolved from the last *included*
        character (``max(end - 1, start)``) — a record straddling a page boundary therefore
        records both pages, and a zero-length span records one. Owning that clamp here is
        the point: it used to be re-derived, identically, at every call site.

        Raises :class:`~latence_core.errors.PageSliceMissingError` if the chunk carries no
        page spans.
        """
        index = self._index_for(chunk)
        pages = (
            index.page_for_offset(start),
            index.page_for_offset(max(end - 1, start)),
        )
        self._fold(index.diagnostics)
        return pages

    # -- diagnostics --------------------------------------------------------

    @property
    def diagnostics(self) -> PageDriftDiagnostics:
        """Drift aggregated across every resolution this resolver has performed.

        The corpus-level roll-up a per-chunk, throwaway ``PageOffsetIndex`` made impossible:
        each index carried its own diagnostics and no caller summed them. This aggregates
        RESULTS — counts of what already happened — so it is order-independent and holds no
        document state.
        """
        return self._diag

    # -- internals ----------------------------------------------------------

    def _index_for(self, chunk: ChunkRecord) -> PageOffsetIndex:
        """A fresh index over this chunk's own spans — the whole of the resolver's "state".

        A slice is 1-2 spans, so this is a two-element list comprehension: cheaper than the
        bookkeeping any cache would need, and it is what keeps the resolver genuinely
        stateless. Because the index is per-call, its diagnostics are exactly that call's
        and the caller folds them into the roll-up.
        """
        page_slice = chunk.page_slice
        if page_slice is None:
            raise PageSliceMissingError(chunk.record_id)
        return PageOffsetIndex(page_slice)

    def _fold(self, diag: PageDriftDiagnostics) -> None:
        """Accumulate one call's drift into the corpus roll-up (a result, not a state)."""
        self._diag = self._diag.model_copy(
            update={
                "resolutions": self._diag.resolutions + diag.resolutions,
                "recovered": self._diag.recovered + diag.recovered,
                "max_drift_chars": max(self._diag.max_drift_chars, diag.max_drift_chars),
            }
        )
