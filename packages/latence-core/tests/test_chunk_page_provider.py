"""``chunk.page`` — the page-preferred Chunker (ADR-0064).

The three claims this Provider makes, each with its own test:

1. a document with a MEASURED page map is cut on its page boundaries — one chunk per page;
2. a document with no such map falls back to the windowed budget at 1024 tokens, and an
   ``ASSUMED_SINGLE_PAGE`` map counts as "no such map" (ADR-0060: an absent page map is a value,
   not a ``page 1``);
3. a page over the budget still SPLITS, because a page-sized chunk past the model window is a
   silent truncation and a split is not.

Plus the invariant that makes the whole thing usable: the S3 round-trip (chunk → offset → original
page) and the v19 ``page_slice`` are exact, identically to the two window strategies — which is
asserted by resolving every emitted offset back through the parent's own page index rather than by
trusting the arithmetic.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    PageMapOrigin,
    Provenance,
)
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.chunk import (
    PAGE_CHUNK_MAX_TOKENS,
    MarkdownChunker,
    PageChunker,
    SentenceWindowChunker,
)


def _doc(
    pages: list[str], *, origin: PageMapOrigin = PageMapOrigin.PARSER
) -> DocumentRecord:
    assembly = PageMap.from_page_texts(pages, origin=origin)
    prov = Provenance(
        source_uri="file:///report.pdf",
        file_name="report.pdf",
        file_type="pdf",
        file_size=1234,
        document_id="sha256:report",
        char_start=0,
        char_end=assembly.page_map.total_chars,
        page_start=1,
        page_end=len(assembly.page_map.pages),
    )
    return DocumentRecord(
        record_id="sha256:report",
        provenance=prov,
        classification=Classification(language="en", category="finance", sensitivity="internal"),
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


# -- 1. one chunk per page ---------------------------------------------------------------------


def test_a_measured_page_map_yields_exactly_one_chunk_per_page() -> None:
    doc = _doc(
        [
            "Alpha content on the first page about revenue.",
            "Beta content on the second page about costs.",
            "Gamma content on the third page about staffing.",
        ]
    )
    chunks = list(PageChunker().chunk([doc]))
    assert len(chunks) == 3
    assert [c.provenance.page_start for c in chunks] == [1, 2, 3]
    assert [c.provenance.page_end for c in chunks] == [1, 2, 3]
    # Each chunk IS its page. The trailing separator falls inside the page's own span (the page
    # map's ``char_end`` covers it), so the chunk covers it too — carrying the page verbatim is
    # more faithful than trimming to a boundary the map does not draw.
    assert [c.content.strip() for c in chunks] == [
        "Alpha content on the first page about revenue.",
        "Beta content on the second page about costs.",
        "Gamma content on the third page about staffing.",
    ]


def test_no_chunk_ever_straddles_a_page_boundary() -> None:
    """The property the window strategies do NOT have, stated as the property rather than a count.

    ``chunk.markdown`` on the same document produces a chunk whose ``page_slice`` holds two spans;
    ``chunk.page`` cannot, by construction. Asserting both sides is what makes this a difference
    between strategies rather than a fact about this fixture.
    """
    pages = [f"Page {i} reports the regional revenue figure." for i in (1, 2, 3)]
    doc = _doc(pages)

    page_chunks = list(PageChunker().chunk([doc]))
    assert len(page_chunks) == 3
    assert all(c.page_slice is not None and len(c.page_slice.pages) == 1 for c in page_chunks)
    assert all(c.provenance.page_start == c.provenance.page_end for c in page_chunks)

    # The same document through the window strategy at a budget larger than the whole document:
    # ONE chunk, spanning all three pages, with a two-span page_slice. That is the behaviour
    # `chunk.page` exists to replace when the source really is paginated.
    window_chunks = list(MarkdownChunker({"max_tokens": 512, "overlap_tokens": 0}).chunk([doc]))
    assert len(window_chunks) == 1
    assert (window_chunks[0].provenance.page_start, window_chunks[0].provenance.page_end) == (1, 3)


def test_a_blank_page_produces_no_chunk() -> None:
    """A page with nothing on it is not an empty chunk — a ChunkRecord needs content to be one."""
    doc = _doc(["First page has text.", "   ", "Third page has text."])
    chunks = list(PageChunker().chunk([doc]))
    assert len(chunks) == 2
    assert [c.provenance.page_start for c in chunks] == [1, 3]


# -- 2. the fallback, and ADR-0060 ---------------------------------------------------------------


def test_an_assumed_single_page_map_falls_back_to_the_window_not_one_giant_chunk() -> None:
    """ADR-0060: an ASSUMED_SINGLE_PAGE map means "no page structure", not "one page".

    Treating its single span as a page would emit one chunk for a whole ``.txt`` file *while
    claiming it was a page*. The origin is read, not merely the map's presence.
    """
    long_text = ("Sentence number one about the quarter. " * 400).strip()
    doc = _doc([long_text], origin=PageMapOrigin.ASSUMED_SINGLE_PAGE)
    chunks = list(PageChunker().chunk([doc]))
    assert len(chunks) > 1
    assert all(c.token_count <= PAGE_CHUNK_MAX_TOKENS for c in chunks)


def test_the_fallback_is_the_reference_strategy_at_the_new_default_budget() -> None:
    """The no-page-map path is ``chunk.markdown`` at 1024 — same engine, larger budget.

    Asserted as byte-identity against the reference Provider configured with this strategy's own
    defaults, so the fallback cannot quietly become a third segmentation nobody reviewed.
    """
    long_text = ("Sentence number one about the quarter. " * 400).strip()
    doc = _doc([long_text], origin=PageMapOrigin.ASSUMED_SINGLE_PAGE)
    page = [c.model_dump(mode="json") for c in PageChunker().chunk([doc])]
    reference = MarkdownChunker(
        {"max_tokens": PAGE_CHUNK_MAX_TOKENS, "overlap_tokens": 64, "min_tokens": 16}
    )
    assert page == [c.model_dump(mode="json") for c in reference.chunk([doc])]


def test_the_existing_chunkers_keep_their_512_default() -> None:
    """The committed benchmark baselines depend on it, so it is pinned rather than assumed.

    A global default change would silently invalidate every published number, which is why 1024 is
    the default *of the new strategy* and of nothing else.
    """
    assert MarkdownChunker()._budget.max_tokens == 512  # noqa: SLF001 - the default is the subject
    assert (
        SentenceWindowChunker()._budget.max_tokens == 512  # noqa: SLF001 - the default is the subject
    )
    assert PageChunker()._budget.max_tokens == PAGE_CHUNK_MAX_TOKENS  # noqa: SLF001 - ditto
    assert PAGE_CHUNK_MAX_TOKENS == 1024


# -- 3. an over-budget page splits ---------------------------------------------------------------


def test_a_page_over_the_budget_splits_rather_than_overflowing_the_model_window() -> None:
    """A page-sized chunk past the model window is a silent truncation; a split is visible."""
    short_page = "A short second page."
    long_page = ("Revenue grew steadily across every region this quarter. " * 60).strip()
    doc = _doc([long_page, short_page])
    chunks = list(PageChunker({"max_tokens": 60, "overlap_tokens": 0, "min_tokens": 4}).chunk([doc]))

    from_page_one = [c for c in chunks if c.provenance.page_start == 1]
    from_page_two = [c for c in chunks if c.provenance.page_start == 2]
    assert len(from_page_one) > 1  # the over-budget page split
    assert len(from_page_two) == 1  # the within-budget page did not
    # Every piece stays inside the budget — that is what "still splits" has to mean.
    assert all(c.token_count <= 60 for c in from_page_one)


def test_a_split_never_merges_a_remnant_across_a_page_boundary() -> None:
    """``min_tokens`` folds a short tail back WITHIN a page — never into the previous page.

    Merging across a page boundary is precisely the thing this strategy exists not to do, and a
    trailing remnant is where a naive implementation would do it.
    """
    long_page = ("Revenue grew steadily across every region this quarter. " * 20).strip()
    tiny_page = "Tiny."
    doc = _doc([long_page, tiny_page])
    chunks = list(PageChunker({"max_tokens": 60, "overlap_tokens": 0, "min_tokens": 40}).chunk([doc]))
    last = chunks[-1]
    assert last.content == "Tiny."
    assert last.provenance.page_start == 2


# -- the invariant that makes it usable ----------------------------------------------------------


def test_every_offset_round_trips_to_its_real_source_page() -> None:
    """S3: chunk → offset → original page, resolved through the PARENT's index, not arithmetic."""
    pages = [
        "Alpha content on the first page about revenue and growth.",
        "Beta content on the second page about costs and staffing.",
        ("Gamma content on the third page. " * 30).strip(),
    ]
    doc = _doc(pages)
    assert doc.page_map is not None
    index = PageOffsetIndex(doc.page_map)

    chunks = list(PageChunker({"max_tokens": 40, "overlap_tokens": 0, "min_tokens": 4}).chunk([doc]))
    assert chunks
    for ch in chunks:
        start, end = ch.provenance.char_start, ch.provenance.char_end
        assert start is not None and end is not None
        # The span really is this chunk's own text in the ORIGINAL document (no markup here, so
        # the slice is exact) — the strongest statement of "the offsets point where they claim".
        assert doc.content[start:end] == ch.content
        assert index.page_for_offset(start) == ch.provenance.page_start
        assert index.page_for_offset(end - 1) == ch.provenance.page_end
    assert index.diagnostics.clean  # no fuzzy drift recovery was needed anywhere


def test_a_sub_chunk_offset_resolves_through_the_carried_offset_map_after_markup_stripping() -> None:
    """The rebase must survive markup stripping, which is where an off-by-N would hide.

    The engine strips markup and records a run-length local→original map. Shifting a page-local
    chunk onto document coordinates has to shift that map too, or every downstream mention span
    lands N characters early — silently, and only on pages after the first.
    """
    doc = _doc(
        [
            "Page one is plain.",
            "**Bold** intro and then the name Acme Corp appears here.",
        ]
    )
    chunks = list(PageChunker().chunk([doc]))
    second = chunks[1]
    assert second.offset_map is not None
    local = second.content.index("Acme Corp")
    doc_start, doc_end = OffsetIndex(second.offset_map).resolve_span(local, local + len("Acme Corp"))
    assert doc.content[doc_start:doc_end] == "Acme Corp"


def test_the_page_slice_is_exact_for_every_chunk() -> None:
    """v19: each chunk answers its own sub-chunk page questions, with no document map in hand."""
    pages = ["First page text here.", ("Second page text. " * 40).strip(), "Third page text."]
    doc = _doc(pages)
    chunks = list(PageChunker({"max_tokens": 30, "overlap_tokens": 0, "min_tokens": 4}).chunk([doc]))
    for ch in chunks:
        assert ch.page_slice is not None
        assert [p.page_number for p in ch.page_slice.pages] == [ch.provenance.page_start]
        assert ch.page_slice.origin is PageMapOrigin.PARSER


def test_it_is_deterministic_and_skips_a_parse_error_document() -> None:
    doc = _doc(["Alpha page.", "Beta page."])
    first = [c.model_dump(mode="json") for c in PageChunker().chunk([doc])]
    second = [c.model_dump(mode="json") for c in PageChunker().chunk([doc])]
    assert first == second

    broken = doc.model_copy(update={"disposition": Disposition.PARSE_ERROR, "content": ""})
    assert list(PageChunker().chunk([broken])) == []


def test_it_declares_a_conformance_valid_profile() -> None:
    profile = PageChunker.profile()
    assert profile.compute == "cpu"
    assert profile.model_id is None
    assert profile.deterministic is True
    assert profile.license_verified is True


@pytest.mark.parametrize("bad", [{"max_tokens": 0}, {"max_tokens": 10, "overlap_tokens": 10}])
def test_a_degenerate_budget_is_rejected_loudly(bad: dict[str, int]) -> None:
    """A bad budget is a ValueError from ChunkBudget at construction, never a silent clamp."""
    with pytest.raises(ValueError, match="tokens"):
        PageChunker(bad)
