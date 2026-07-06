"""OffsetIndex — stripped-content→original-markdown offset resolution (S4 coord fix).

The companion resolver to :class:`~latence_core.pagemap.PageOffsetIndex`, one level
finer. Chunk ``content`` is markup-stripped (``chunking.strip_markup`` deletes 18
markdown constructs) while the chunk's own ``[char_start, char_end)`` stays in the
ORIGINAL parsed markdown. That is enough for the *chunk's* own span, but a downstream
Stage that finds a **sub-chunk** offset (an entity mention's chunk-local match) cannot
recover its true original offset by ``char_start + local``: markup stripping only ever
*deletes* characters, so once markup was stripped before the mention within its chunk,
``char_start + local`` is a strict lower bound, not the true offset (ADR-0022 corrected,
ADR-0031).

The fix carries a compact run-length :class:`~latence_core.contracts.OffsetMap` forward
on each chunk (the map the chunking engine already builds while stripping, then used to
discard). This index resolves a chunk-local offset — or half-open local span — through
that map back to exact original-markdown coordinates, so Entity Extraction stamps a
mention's TRUE offset and page at source. A markup-free chunk has a single identity
segment, so resolution is exactly ``char_start + local`` — no behavioural change on the
plain-text path.

Like :class:`PageOffsetIndex`, this is a live index built once per chunk and queried
many times; it never raises — an out-of-range local offset clamps into the chunk so
Provenance resolution is never the thing that fails a run.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latence_core.contracts import OffsetMap


class OffsetIndex:
    """O(log n) chunk-local→original-markdown offset resolver (never crashes).

    Built once per chunk from its :class:`~latence_core.contracts.OffsetMap`. Not a
    Pydantic model — a live index, distinct from the serialized ``OffsetMap`` contract
    it is built from (the same split as :class:`PageOffsetIndex`/:class:`PageMap`).
    """

    def __init__(self, offset_map: OffsetMap) -> None:
        # Segments are validated ordered/first-at-0, so the parallel arrays are sorted
        # ascending and bisect finds the run covering any local offset.
        self._local_starts: list[int] = [seg.local_start for seg in offset_map.segments]
        self._doc_starts: list[int] = [seg.doc_start for seg in offset_map.segments]
        self._length = offset_map.length

    def resolve(self, local: int) -> int:
        """The original-markdown offset of chunk-local character ``local``.

        Exact when ``local`` lands in the chunk's content. A negative offset clamps to
        the chunk start; an offset at/after the content length clamps to the last
        character — resolution never raises so a malformed sub-chunk offset degrades
        rather than aborting the run.
        """
        if local <= 0:
            return self._doc_starts[0]
        # Clamp past-the-end to the last real character so resolve stays total.
        if self._length and local >= self._length:
            local = self._length - 1
        idx = bisect.bisect_right(self._local_starts, local) - 1
        if idx < 0:  # pragma: no cover - local>0 and _local_starts[0]==0 make this unreachable
            idx = 0
        return self._doc_starts[idx] + (local - self._local_starts[idx])

    def resolve_span(self, local_start: int, local_end: int) -> tuple[int, int]:
        """A half-open chunk-local span → its half-open span in the original markdown.

        The start resolves directly; the exclusive end resolves from the last included
        character (``local_end - 1``) plus one, so a span whose interior contained
        stripped markup still covers the full original extent (the same exclusive-end
        convention the chunking engine uses for a chunk's own ``char_end``). An empty
        or inverted local span collapses to a zero-width span at the resolved start.
        """
        start = self.resolve(local_start)
        if local_end <= local_start:
            return start, start
        end = self.resolve(local_end - 1) + 1
        # Original coordinates are monotonic in local, but clamp defensively so the
        # span is never inverted for a downstream contract that forbids it.
        return start, max(end, start)
