# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Page-span reconstruction — byte-exact against the real S2 artifacts, or it does not ship."""

from __future__ import annotations

import pytest
from benchmark.s5.datasets import S2_OUTPUT
from benchmark.s5.pagemap import (
    DocumentPages,
    PageMapError,
    build_document_pages,
    header_fields,
    load_dataset_pages,
)

_PAGES = S2_OUTPUT / "pages" / "vidoseek"
_MARKDOWN = S2_OUTPUT / "markdown" / "vidoseek"

requires_s2 = pytest.mark.skipif(
    not _PAGES.is_dir(), reason="the S2 page artifacts are not on this machine"
)


def test_overlap_lookup_is_inclusive_of_every_touched_page() -> None:
    doc = DocumentPages(doc_id="d", starts=(0, 100, 200), ends=(90, 190, 290))
    assert doc.pages_overlapping(0, 50) == (1,)
    assert doc.pages_overlapping(50, 150) == (1, 2)
    assert doc.pages_overlapping(95, 99) == ()
    assert doc.pages_overlapping(10, 10) == ()
    assert doc.span(3) == (200, 290)
    assert doc.span(4) is None


def test_header_fields_reads_the_model_back_rather_than_trusting_the_caller() -> None:
    header = "<!-- latence-campaign s2-ocr | doc_id=d | dataset=x | pages=2 | model=m | revision=r -->"
    assert header_fields(f"{header}\n\nbody") == {
        "doc_id": "d",
        "dataset": "x",
        "pages": "2",
        "model": "m",
        "revision": "r",
    }
    assert header_fields("no header here") == {}


@requires_s2
def test_reconstruction_is_byte_exact_on_real_documents() -> None:
    """If this drifts, every page-level gold reference is silently wrong — so it must fail here."""
    docs = sorted(d for d in _PAGES.iterdir() if d.is_dir())[:15]
    checked = 0
    for doc_dir in docs:
        markdown = _MARKDOWN / f"{doc_dir.name}.md"
        if not markdown.is_file():
            continue
        pages = build_document_pages(doc_dir, markdown)
        text = markdown.read_text(encoding="utf-8")
        assert pages.page_count == int(header_fields(text)["pages"])
        # Every reconstructed span must slice the real markdown back to that page's artifact.
        for page in range(1, pages.page_count + 1):
            start, end = pages.span(page) or (0, 0)
            assert text[start:end] == _normalized_page(doc_dir, page)
        checked += 1
    assert checked >= 5


def _normalized_page(doc_dir, page: int) -> str:  # type: ignore[no-untyped-def]
    from latence_benchmark.campaign.ocr_lib import normalize_html_tables

    return normalize_html_tables(
        (doc_dir / f"{page:04d}.md").read_text(encoding="utf-8")
    ).text


@requires_s2
def test_a_document_without_a_header_is_a_loud_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    (doc_dir / "0001.md").write_text("page one", encoding="utf-8")
    markdown = tmp_path / "doc.md"
    markdown.write_text("page one\n", encoding="utf-8")
    with pytest.raises(PageMapError):
        build_document_pages(doc_dir, markdown)


@requires_s2
def test_load_dataset_pages_honours_the_only_filter() -> None:
    wanted = sorted(d.name for d in _PAGES.iterdir() if d.is_dir())[:3]
    loaded = load_dataset_pages(S2_OUTPUT, "vidoseek", only=frozenset(wanted))
    assert set(loaded) <= set(wanted)
    assert loaded
