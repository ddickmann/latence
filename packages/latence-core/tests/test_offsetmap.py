"""OffsetIndex: chunk-local→original-markdown offset resolution (S4 coordinate fix).

Unit tests for the resolver that turns a chunk's stripped-content offsets back into
original parsed-markdown offsets through its carried :class:`OffsetMap`: the identity
case (markup-free chunk), the gap case (markup stripped before a position), exclusive
half-open span ends, and never-crash clamping.
"""

from __future__ import annotations

from latence_core.contracts import OffsetMap, OffsetSegment
from latence_core.offsetmap import OffsetIndex


def _map(segments: list[tuple[int, int]], length: int) -> OffsetMap:
    return OffsetMap(
        segments=[OffsetSegment(local_start=ls, doc_start=ds) for ls, ds in segments],
        length=length,
    )


def test_identity_map_resolves_to_char_start_plus_local() -> None:
    # A markup-free chunk: single (0, char_start) segment ⇒ resolve == char_start + local.
    idx = OffsetIndex(_map([(0, 100)], length=10))
    for k in range(10):
        assert idx.resolve(k) == 100 + k
    assert idx.resolve_span(3, 8) == (103, 108)


def test_gap_segment_shifts_later_offsets_past_the_lower_bound() -> None:
    # Models "**Hi** Alice." stripped to "Hi Alice.": a gap after "Hi" (local 2).
    # segments: local 0->orig 2 (identity within), local 2->orig 6 (after the stripped ** ).
    idx = OffsetIndex(_map([(0, 2), (2, 6)], length=9))
    assert idx.resolve(0) == 2  # 'H'
    assert idx.resolve(1) == 3  # 'i'
    assert idx.resolve(2) == 6  # ' ' — jumped past the stripped "**"
    assert idx.resolve(3) == 7  # 'A' of Alice — TRUE offset, not the 2+3=5 lower bound
    # Half-open span for "Alice" (local [3, 8)) covers the exact original extent.
    assert idx.resolve_span(3, 8) == (7, 12)


def test_span_exclusive_end_spans_interior_stripped_markup() -> None:
    # "Acme**Corp**" stripped to "AcmeCorp": the interior ** is a gap the span must cover.
    # local 0->0 ("Acme"), local 4->6 ("Corp", after stripped **). length 8.
    idx = OffsetIndex(_map([(0, 0), (4, 6)], length=8))
    # The mention "AcmeCorp" local [0, 8): original end is resolve(7)+1 = 9+1 = 10, so the
    # covering span [0, 10) includes the interior stripped "**".
    assert idx.resolve_span(0, 8) == (0, 10)


def test_resolve_clamps_out_of_range_without_crashing() -> None:
    idx = OffsetIndex(_map([(0, 5), (3, 20)], length=6))
    # Negative clamps to the chunk start; past-the-end clamps to the last character.
    assert idx.resolve(-4) == 5
    assert idx.resolve(999) == idx.resolve(5)
    # An empty/inverted local span collapses to a zero-width span at the start.
    assert idx.resolve_span(4, 4) == (idx.resolve(4), idx.resolve(4))
    assert idx.resolve_span(4, 2) == (idx.resolve(4), idx.resolve(4))


def test_resolve_original_is_the_inverse_of_resolve() -> None:
    # Identity map: original offset - char_start == local (a markup-free chunk).
    idx = OffsetIndex(_map([(0, 100)], length=10))
    for k in range(10):
        assert idx.resolve_original(100 + k) == k

    # Gap map ("**Hi** Alice." → "Hi Alice."): original→local inverts resolve exactly on kept chars.
    idx = OffsetIndex(_map([(0, 2), (2, 6)], length=9))
    for local in range(9):
        assert idx.resolve_original(idx.resolve(local)) == local
    # 'A' of Alice is at original 7 → content-local 3 (NOT the naive 7 - 2 = 5 lower bound).
    assert idx.resolve_original(7) == 3


def test_resolve_original_gap_and_bounds_clamp_without_crashing() -> None:
    # local 0->orig 2 ("Hi"), local 2->orig 6 (after a stripped "**" at original [4,6)). length 9.
    idx = OffsetIndex(_map([(0, 2), (2, 6)], length=9))
    # An original offset inside the stripped gap [4, 6) has no surviving character → clamps to the
    # start of the next surviving run (content-local 2).
    assert idx.resolve_original(4) == 2
    assert idx.resolve_original(5) == 2
    # Before the chunk clamps to content start 0; far past the chunk clamps to content length.
    assert idx.resolve_original(0) == 0
    assert idx.resolve_original(-5) == 0
    assert idx.resolve_original(9999) == idx._length  # noqa: SLF001 - assert the exclusive bound
