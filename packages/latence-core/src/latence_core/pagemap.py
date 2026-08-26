"""PageOffsetIndex — offset→page resolution over a parsed document (ADR asset port).

Ported faithfully — as-is — from the dataset-intelligence ``page_mapper.py`` asset,
landing in S2/S8. The
job is Provenance integrity: after Parse concatenates a document's pages into one
markdown string, *any* downstream character offset (a chunk boundary, an entity
mention span) must resolve back to the original source page it came from — so a
knowledge-graph edge can cite "page 7 of the contract", not "character 41,213 of
an opaque blob".

The algorithm has four parts, kept verbatim from the source asset:

1. **Cumulative page boundaries.** A :class:`PageMap` records, per page, the
   ``[char_start, char_end)`` half-open span the page occupies in the assembled
   text. Boundaries are cumulative and contiguous by construction.
2. **``bisect`` O(log n) offset→page.** :meth:`PageOffsetIndex.page_for_offset`
   binary-searches the cumulative starts — O(log n) in the page count, so it is
   cheap even for a 10k-page document queried millions of times during Chunk/NER.
3. **Fuzzy drift-recovery fallback.** Real parsers drift: a page's recorded
   length and its actual contribution to the assembled text disagree (a trailing
   form-feed swallowed, whitespace normalised). When an offset lands in a gap or
   past the end, the index does not crash — it recovers the *nearest* plausible
   page and records the drift magnitude.
4. **Drift diagnostics + never-crash degradation.** Every recovery is counted and
   the worst drift retained, surfaced as :class:`PageDriftDiagnostics` for the
   Quality Report. A structurally impossible query (empty map) degrades to page 1
   with a diagnostic, never an exception — Provenance resolution must not be the
   thing that fails a run.
"""

from __future__ import annotations

import bisect

from pydantic import BaseModel, ConfigDict

from latence_core.contracts import PageMap, PageSlice, PageSpan

#: What an index can be built over: a whole document's map, or ONE chunk's slice of it.
#:
#: The two are structurally identical where the algorithm touches them — an ascending,
#: contiguous run of :class:`~latence_core.contracts.PageSpan`\\ s plus the document's
#: ``total_chars`` — so the ported resolution logic is shared verbatim rather than forked. They
#: differ only in coverage: a :class:`~latence_core.contracts.PageMap` starts at page 1 and
#: character 0, a :class:`~latence_core.contracts.PageSlice` starts wherever its chunk does.
PageSpans = PageMap | PageSlice


class PageDriftDiagnostics(BaseModel):
    """Drift diagnostics for the Quality Report (ADR-0008).

    ``resolutions`` is how many offset→page lookups were performed;
    ``recovered`` is how many needed the fuzzy fallback (offset outside any exact
    span); ``max_drift_chars`` is the worst gap between a queried offset and the
    span it was recovered to. All-exact is ``recovered == 0``.
    """

    model_config = ConfigDict(extra="forbid")

    resolutions: int = 0
    recovered: int = 0
    max_drift_chars: int = 0

    @property
    def clean(self) -> bool:
        return self.recovered == 0


class PageOffsetIndex:
    """O(log n) offset→page resolver with fuzzy drift recovery (never crashes).

    Built once per document from its :class:`PageMap`; queried many times. Not a
    Pydantic model — it is a live index with mutable diagnostics, distinct from
    the serialized :class:`PageMap` contract it is built from.
    """

    def __init__(self, page_map: PageSpans) -> None:
        self._page_map = page_map
        # Cumulative starts for bisect. pages are validated contiguous/ordered, so
        # ``_starts[i]`` is page i+1's char_start and the array is sorted ascending.
        self._starts: list[int] = [p.char_start for p in page_map.pages]
        self._total = page_map.total_chars
        self._diag = PageDriftDiagnostics()

    @property
    def diagnostics(self) -> PageDriftDiagnostics:
        """The running drift diagnostics (a snapshot copy — safe to serialize)."""
        return self._diag.model_copy()

    @property
    def page_count(self) -> int:
        """How many page spans this index covers — the document's page count.

        This is what an index *weighs*: the retained ``PageMap`` is one
        :class:`~latence_core.contracts.PageSpan` per page (~550 bytes measured on a
        519-page document, ~279 KB in total), so a caller that has to bound how much page
        state it keeps live bounds THIS, not the number of documents — a 1-page note and a
        519-page contract differ by ~500× (:data:`latence_core.page_index.DEFAULT_MAX_LIVE_PAGES`).
        """
        return len(self._starts)

    def page_for_offset(self, offset: int) -> int:
        """Return the 1-based source page for a character ``offset`` in the text.

        Exact when ``offset`` falls inside a page's span. Otherwise the fuzzy
        fallback recovers the nearest plausible page and records the drift:

        * a negative offset clamps to the first page covered (drift = |offset|);
        * an offset at/after the assembled end clamps to the last page covered
          (drift = offset - (total_chars - 1), floored at 0);
        * over a :class:`~latence_core.contracts.PageSlice`, an offset BEFORE the
          slice's first page clamps to it (drift = distance);
        * an in-range offset that a ``bisect`` places is exact — the contiguous
          invariant means there are no interior gaps, but the fallback still runs
          if the map is somehow degenerate, so resolution never raises.
        """
        self._diag.resolutions += 1
        n = len(self._starts)
        if n == 0:  # pragma: no cover - PageMap forbids an empty page list
            self._record_drift(abs(offset))
            return 1

        if offset < 0:
            self._record_drift(-offset)
            return self._page_number(0)

        # Past the end: clamp to the last page. drift measured from the last valid
        # character index (total_chars - 1), floored at 0.
        if offset >= self._total:
            last_valid = max(self._total - 1, 0)
            self._record_drift(max(offset - last_valid, 0))
            return self._page_number(n - 1)

        # bisect_right - 1 gives the page whose char_start is the greatest value
        # <= offset: O(log n). With contiguous spans this is always the exact page.
        idx = bisect.bisect_right(self._starts, offset) - 1
        if idx < 0:
            # Before the FIRST span this index covers. Unreachable over a whole document map
            # (offset >= 0 and _starts[0] == 0), but reachable over a chunk's PageSlice, whose
            # first span starts where its chunk does: a caller asking about an offset that
            # precedes the chunk gets the chunk's first page, and the distance is recorded as
            # drift rather than passed off as an exact resolution.
            self._record_drift(self._starts[0] - offset)
            idx = 0
        return self._page_number(idx)

    def page_span(self, page_number: int) -> PageSpan:
        """The recorded span for a 1-based ``page_number`` (raises if out of range).

        Addressed by the page's own number, not by position, so this reads the same over a
        whole document map (which starts at page 1) and over a chunk's slice of one (which
        starts at whatever page its chunk does).
        """
        pages = self._page_map.pages
        first = pages[0].page_number
        last = first + len(pages) - 1
        if not first <= page_number <= last:
            msg = f"Page {page_number} out of range {first}..{last}"
            raise IndexError(msg)
        return pages[page_number - first]

    def _page_number(self, index: int) -> int:
        return self._page_map.pages[index].page_number

    def _record_drift(self, drift: int) -> None:
        self._diag = self._diag.model_copy(
            update={
                "recovered": self._diag.recovered + 1,
                "max_drift_chars": max(self._diag.max_drift_chars, drift),
            }
        )
