# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The regression that would have caught the page-1 corpus (ADR-0060).

An OCR'd multi-page document is produced exactly the way S2 produces one — per-page markdown
normalized and assembled by :func:`ocr_lib.assemble_document_with_pages` — written to a folder
with its page-map sidecar, and then driven through the REAL ingest path the S3 pipeline wires:
``source.local_folder → parser.plaintext → chunk``. A chunk whose text came from page N must
report page N.

On the pre-ADR code every one of these assertions returns 1, which is how 3,470 documents and
236,000 chunks shipped with `provenance.page_start == page_end == 1` and nothing failed. So the
suite also pins the two properties that keep the fix honest: the assembled markdown is
byte-identical to what S2 already wrote (the S3 corpus and ``benchmark/s5/pagemap.py`` are built
on those bytes), and a document with NO page information degrades visibly rather than silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_benchmark.campaign import ocr_lib
from latence_core.contracts import (
    PAGE_MAP_SIDECAR_SUFFIX,
    Disposition,
    DocumentRecord,
    PageMapOrigin,
)
from latence_core.stages.chunk import MarkdownChunker
from latence_core.stages.source import LocalFolderSource
from latence_core.storage import Storage
from latence_parser_plaintext import PlaintextParser

#: Distinctive, chunk-sized pages so a chunk can be traced back to the page it came from.
#: Each page is long enough that the chunker cannot swallow the whole document in one chunk.
PAGE_TEXTS: tuple[str, ...] = tuple(
    f"# Section {n}\n\n" + " ".join(f"page{n}word{i}" for i in range(120)) for n in range(1, 8)
)

_DOC_KWARGS = {
    "doc_id": "paper-0001",
    "dataset": "ohr_bench",
    "model": "lightonai/LightOnOCR-2-1B",
    "revision": "abc123",
}


def _write_ocr_output(root: Path, *, with_sidecar: bool = True) -> ocr_lib.AssembledDocument:
    """Write what the S2 worker writes: the assembled markdown (+ its page map)."""
    assembly = ocr_lib.assemble_document_with_pages(list(PAGE_TEXTS), **_DOC_KWARGS)
    root.mkdir(parents=True, exist_ok=True)
    markdown = root / "paper-0001.md"
    markdown.write_text(assembly.text, encoding="utf-8")
    if with_sidecar:
        markdown.with_name(markdown.name + PAGE_MAP_SIDECAR_SUFFIX).write_text(
            ocr_lib.page_map_sidecar_json(assembly), encoding="utf-8"
        )
    return assembly


def _ingest(root: Path, *, parser_config: dict[str, object] | None = None) -> DocumentRecord:
    """Source → Parse over ``root``, exactly as the S3 pipeline.yaml wires it."""
    uri = f"file://{root}"
    storage = Storage.for_uri(uri)
    source = LocalFolderSource({"path": uri, "extensions": ["md", "markdown"]})
    parser = PlaintextParser(parser_config)
    parsed = list(parser.parse(source.produce(storage)))
    assert len(parsed) == 1, "the sidecar must not be ingested as a second document"
    return parsed[0]


# -- the regression ----------------------------------------------------------------------------


def test_a_chunk_from_page_n_reports_page_n_through_the_real_pipeline(tmp_path: Path) -> None:
    assembly = _write_ocr_output(tmp_path)
    document = _ingest(tmp_path)

    assert document.disposition is Disposition.PARSED
    assert document.page_map is not None
    assert len(document.page_map.pages) == len(PAGE_TEXTS)
    assert document.page_map.origin is PageMapOrigin.SIDECAR
    assert document.content == assembly.text

    chunks = list(MarkdownChunker({"max_tokens": 64, "overlap_tokens": 0}).chunk([document]))
    assert len(chunks) > len(PAGE_TEXTS), "expected several chunks per page to test resolution"

    # Every page must be witnessed by at least one chunk claiming it — the whole document
    # collapsing to page 1 is exactly the shipped defect.
    claimed = {chunk.provenance.page_start for chunk in chunks}
    assert claimed == set(range(1, len(PAGE_TEXTS) + 1))

    # And each chunk's claim must be TRUE: the page whose marker words it contains.
    for chunk in chunks:
        first, last = chunk.provenance.page_start, chunk.provenance.page_end
        assert first is not None and last is not None
        for page_number in range(1, len(PAGE_TEXTS) + 1):
            if f"page{page_number}word" in chunk.content:
                assert first <= page_number <= last, (
                    f"chunk {chunk.record_id} contains page-{page_number} text but claims "
                    f"pages {first}..{last}"
                )

    # The chunk is self-describing about it, too (ADR-0034), origin included.
    late = next(chunk for chunk in chunks if chunk.provenance.page_start == len(PAGE_TEXTS))
    assert late.page_slice is not None
    assert late.page_slice.origin is PageMapOrigin.SIDECAR
    assert late.page_slice.pages[-1].page_number == len(PAGE_TEXTS)


# -- visible degradation ------------------------------------------------------------------------


def test_without_a_sidecar_the_assumption_is_recorded_not_hidden(tmp_path: Path) -> None:
    _write_ocr_output(tmp_path, with_sidecar=False)
    document = _ingest(tmp_path)

    assert document.page_map is not None
    assert document.page_map.origin is PageMapOrigin.ASSUMED_SINGLE_PAGE
    chunks = list(MarkdownChunker({"max_tokens": 64, "overlap_tokens": 0}).chunk([document]))
    # Still page 1 everywhere — but every chunk now SAYS that is an assumption, so the
    # 236k-chunks-cite-page-1 corpus would have been detectable instead of plausible.
    assert {chunk.provenance.page_start for chunk in chunks} == {1}
    assert all(
        chunk.page_slice is not None
        and chunk.page_slice.origin is PageMapOrigin.ASSUMED_SINGLE_PAGE
        for chunk in chunks
    )


def test_a_declared_paginated_corpus_refuses_the_unpaginated_document(tmp_path: Path) -> None:
    _write_ocr_output(tmp_path, with_sidecar=False)
    document = _ingest(tmp_path, parser_config={"require_page_map": True})

    assert document.disposition is Disposition.PARSE_ERROR
    assert document.error is not None and PAGE_MAP_SIDECAR_SUFFIX in document.error
    # A PARSE_ERROR document is skipped by Chunk, so nothing reaches the corpus claiming page 1.
    assert list(MarkdownChunker().chunk([document])) == []


# -- byte-identity of the artifact the S3 corpus was built from -----------------------------------


def test_assembled_markdown_is_byte_identical_to_the_pre_sidecar_assembly() -> None:
    """``assemble_document`` is the spans function's ``.text`` — one join, no drift."""
    assembly = ocr_lib.assemble_document_with_pages(list(PAGE_TEXTS), **_DOC_KWARGS)
    assert assembly.text == ocr_lib.assemble_document(list(PAGE_TEXTS), **_DOC_KWARGS)
    # The exact shape S2 wrote: header, blank line, pages joined by blank lines, trailing \n.
    header = ocr_lib.provenance_header(page_count=len(PAGE_TEXTS), **_DOC_KWARGS)
    assert assembly.text == header + "\n\n" + "\n\n".join(PAGE_TEXTS) + "\n"


def test_page_spans_tile_the_assembled_text_exactly() -> None:
    assembly = ocr_lib.assemble_document_with_pages(list(PAGE_TEXTS), **_DOC_KWARGS)
    spans = assembly.page_spans

    assert [number for number, _, _ in spans] == list(range(1, len(PAGE_TEXTS) + 1))
    assert spans[0][1] == 0
    assert spans[-1][2] == len(assembly.text)
    for previous, following in zip(spans, spans[1:], strict=False):
        assert previous[2] == following[1]  # contiguous: no character belongs to no page
    # Each page's own text lands inside its own span.
    for (_, start, end), page in zip(spans, PAGE_TEXTS, strict=True):
        assert page in assembly.text[start:end]


def test_the_sidecar_suffix_matches_the_framework_constant() -> None:
    """``ocr_lib`` is pushed to the pod as a bare file and cannot import core — so pin it."""
    assert ocr_lib.PAGE_MAP_SIDECAR_SUFFIX == PAGE_MAP_SIDECAR_SUFFIX


def test_the_sidecar_is_deterministic_and_omits_origin() -> None:
    assembly = ocr_lib.assemble_document_with_pages(list(PAGE_TEXTS), **_DOC_KWARGS)
    payload = ocr_lib.page_map_sidecar_json(assembly)
    assert payload == ocr_lib.page_map_sidecar_json(assembly)
    decoded = json.loads(payload)
    # A file cannot vouch for how its boundaries were obtained; the consumer stamps that.
    assert "origin" not in decoded
    assert decoded["total_chars"] == len(assembly.text)
