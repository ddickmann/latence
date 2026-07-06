"""PageMap + PageOffsetIndex: the ported offset->page asset (S2 Provenance integrity).

Covers the four parts of the ported algorithm: cumulative contiguous boundaries,
O(log n) exact offset->page, fuzzy drift-recovery fallback, and never-crash
degradation with drift diagnostics.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import PageMap, PageSpan
from latence_core.pagemap import PageOffsetIndex


def test_from_page_texts_builds_contiguous_map() -> None:
    a = PageMap.from_page_texts(["alpha", "beta", "gamma"], separator="\n\n")
    # Assembled text is the pages joined by the separator.
    assert a.text == "alpha\n\nbeta\n\ngamma"
    assert a.page_map.total_chars == len(a.text)
    # Three contiguous spans, page-numbered 1..3, covering the whole text.
    assert [s.page_number for s in a.page_map.pages] == [1, 2, 3]
    prev = 0
    for span in a.page_map.pages:
        assert span.char_start == prev
        prev = span.char_end
    assert prev == a.page_map.total_chars


def test_empty_pages_coerced_to_single_empty_page() -> None:
    a = PageMap.from_page_texts([])
    assert a.text == ""
    assert len(a.page_map.pages) == 1
    assert a.page_map.total_chars == 0


def test_offset_to_page_is_exact_within_spans() -> None:
    a = PageMap.from_page_texts(["alpha", "beta", "gamma"])
    idx = PageOffsetIndex(a.page_map)
    # First char of each page resolves to that page.
    for span in a.page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
        assert idx.page_for_offset(span.char_end - 1) == span.page_number
    # No fuzzy recovery for any in-range offset.
    assert idx.diagnostics.recovered == 0
    assert idx.diagnostics.clean


def test_offset_past_end_clamps_to_last_page_with_drift() -> None:
    a = PageMap.from_page_texts(["one", "two"])
    idx = PageOffsetIndex(a.page_map)
    total = a.page_map.total_chars
    assert idx.page_for_offset(total + 50) == 2  # last page
    diag = idx.diagnostics
    assert diag.recovered == 1
    # drift measured from the last valid char index.
    assert diag.max_drift_chars == (total + 50) - (total - 1)


def test_negative_offset_clamps_to_first_page_with_drift() -> None:
    a = PageMap.from_page_texts(["one", "two"])
    idx = PageOffsetIndex(a.page_map)
    assert idx.page_for_offset(-7) == 1
    assert idx.diagnostics.recovered == 1
    assert idx.diagnostics.max_drift_chars == 7


def test_diagnostics_accumulate_and_snapshot_is_independent() -> None:
    a = PageMap.from_page_texts(["one", "two"])
    idx = PageOffsetIndex(a.page_map)
    snap0 = idx.diagnostics
    idx.page_for_offset(-3)
    idx.page_for_offset(9999)
    # The earlier snapshot did not mutate.
    assert snap0.resolutions == 0
    diag = idx.diagnostics
    assert diag.resolutions == 2
    assert diag.recovered == 2


def test_page_span_lookup_and_out_of_range() -> None:
    a = PageMap.from_page_texts(["one", "two"])
    idx = PageOffsetIndex(a.page_map)
    assert idx.page_span(1).page_number == 1
    with pytest.raises(IndexError):
        idx.page_span(3)


def test_pagemap_rejects_non_contiguous_spans() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        PageMap(
            pages=[
                PageSpan(page_number=1, char_start=0, char_end=5),
                PageSpan(page_number=2, char_start=7, char_end=10),  # gap at 5..7
            ],
            total_chars=10,
        )


def test_pagemap_rejects_wrong_total_chars() -> None:
    with pytest.raises(ValueError, match="total_chars"):
        PageMap(pages=[PageSpan(page_number=1, char_start=0, char_end=5)], total_chars=99)


def test_pagemap_rejects_out_of_order_page_numbers() -> None:
    with pytest.raises(ValueError, match="1..n in order"):
        PageMap(
            pages=[
                PageSpan(page_number=2, char_start=0, char_end=5),
                PageSpan(page_number=1, char_start=5, char_end=10),
            ],
            total_chars=10,
        )


def test_pagespan_rejects_inverted_span() -> None:
    with pytest.raises(ValueError, match="inverted"):
        PageSpan(page_number=1, char_start=10, char_end=3)
