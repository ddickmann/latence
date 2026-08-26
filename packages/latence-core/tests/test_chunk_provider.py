"""Unit tests for the reference MarkdownChunker Provider.

Focuses on the S3 acceptance criterion: Chunk preserves offsets, page mapping,
Provenance and Classification on every chunk — verified by a round-trip test
(chunk → offset → original page via the ported PageOffsetIndex).
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
)
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.chunk import MarkdownChunker
from latence_core.stages.redaction import HybridPIIRedactor


def _doc(pages: list[str], *, sensitivity: str = "confidential") -> DocumentRecord:
    assembly = PageMap.from_page_texts(pages)
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
        classification=Classification(language="en", category="finance", sensitivity=sensitivity),
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


def test_chunker_preserves_provenance_and_classification() -> None:
    doc = _doc([("The revenue report shows strong growth this quarter. " * 20).strip()])
    chunker = MarkdownChunker({"max_tokens": 40, "overlap_tokens": 8, "min_tokens": 4})
    chunks = list(chunker.chunk([doc]))
    assert len(chunks) > 1
    for ch in chunks:
        # Source-identity Provenance carried onto every chunk.
        assert ch.provenance.document_id == doc.provenance.document_id
        assert ch.provenance.file_name == "report.pdf"
        # Classification inherited unchanged.
        assert ch.classification == doc.classification
        assert ch.document_record_id == doc.record_id


def test_chunker_offsets_round_trip_to_source_page() -> None:
    """chunk → offset → original page: every chunk resolves to a real source page."""
    page_a = "Alpha content on the first page about revenue. " * 6
    page_b = "Beta content on the second page about costs. " * 6
    doc = _doc([page_a.strip(), page_b.strip()])
    assert doc.page_map is not None
    page_map = doc.page_map
    index = PageOffsetIndex(page_map)

    chunker = MarkdownChunker({"max_tokens": 30, "overlap_tokens": 0, "min_tokens": 4})
    chunks = list(chunker.chunk([doc]))
    assert len(chunks) >= 2
    for ch in chunks:
        prov = ch.provenance
        assert prov.char_start is not None and prov.char_end is not None
        assert 0 <= prov.char_start <= prov.char_end <= page_map.total_chars
        # The page recorded on the chunk equals the page the offset resolves to.
        expected_start_page = index.page_for_offset(prov.char_start)
        assert prov.page_start == expected_start_page
        # Both source pages are represented across the chunk set.
    pages_seen = {ch.provenance.page_start for ch in chunks}
    assert pages_seen == {1, 2}


def test_chunker_char_span_indexes_original_markdown() -> None:
    doc = _doc([("word alpha beta gamma delta epsilon zeta. " * 10).strip()])
    chunker = MarkdownChunker({"max_tokens": 25, "overlap_tokens": 0, "min_tokens": 2})
    for ch in chunker.chunk([doc]):
        # No markup in this doc, so the original slice equals the chunk text.
        assert doc.content[ch.provenance.char_start : ch.provenance.char_end] == ch.content


def test_chunker_stamps_offset_map_resolving_subchunk_offsets_to_original() -> None:
    """Each chunk carries an offset_map that resolves a sub-chunk offset to the ORIGINAL.

    The chunk's own char span is already correct; the offset_map lets a *sub-chunk*
    offset (an entity mention) resolve exactly even when markdown markup was stripped
    before it — the whole point of the S4 coordinate fix.
    """
    doc = _doc(["The **bold** revenue and the [link](http://x) both grew sharply here."])
    chunker = MarkdownChunker({"max_tokens": 512})
    (chunk,) = list(chunker.chunk([doc]))
    assert chunk.offset_map is not None
    # Markup was stripped, so the map is NOT a single identity segment.
    assert len(chunk.offset_map.segments) > 1
    index = OffsetIndex(chunk.offset_map)
    # Every chunk-local character resolves through the map to the identical original char.
    for k, ch in enumerate(chunk.content):
        assert doc.content[index.resolve(k)] == ch
    # "revenue" follows the stripped ``**bold**``: its resolved span indexes the original.
    local = chunk.content.index("revenue")
    start, end = index.resolve_span(local, local + len("revenue"))
    assert doc.content[start:end] == "revenue"
    assert chunk.provenance.char_start is not None
    assert start > chunk.provenance.char_start + local  # strictly past the lower bound


def test_chunker_indices_are_contiguous_from_zero() -> None:
    doc = _doc([("Sentence content that will split into several chunks. " * 15).strip()])
    chunker = MarkdownChunker({"max_tokens": 30, "overlap_tokens": 5, "min_tokens": 3})
    chunks = list(chunker.chunk([doc]))
    assert [ch.index for ch in chunks] == list(range(len(chunks)))
    assert [ch.record_id for ch in chunks] == [
        f"{doc.record_id}#chunk-{i}" for i in range(len(chunks))
    ]


def test_chunker_skips_parse_error_documents() -> None:
    bad = DocumentRecord(
        record_id="sha256:bad",
        provenance=Provenance(
            source_uri="file:///bad.pdf",
            file_name="bad.pdf",
            file_type="pdf",
            file_size=10,
            document_id="sha256:bad",
        ),
        classification=Classification(language="und"),
        content="",
        page_map=None,
        disposition=Disposition.PARSE_ERROR,
        error="corrupt",
    )
    good = _doc([("Good content to chunk here. " * 10).strip()])
    chunks = list(MarkdownChunker().chunk([bad, good]))
    # No chunks from the failed doc; all chunks belong to the good doc.
    assert chunks
    assert {ch.document_record_id for ch in chunks} == {good.record_id}


def test_chunker_is_deterministic() -> None:
    doc = _doc([("Deterministic output is required by the Baseline bar. " * 20).strip()])
    chunker = MarkdownChunker({"max_tokens": 35, "overlap_tokens": 7, "min_tokens": 3})
    a = [(c.record_id, c.content, c.provenance.char_start) for c in chunker.chunk([doc])]
    b = [(c.record_id, c.content, c.provenance.char_start) for c in chunker.chunk([doc])]
    assert a == b


def test_chunker_empty_content_document_yields_no_chunks() -> None:
    doc = _doc([""])
    assert list(MarkdownChunker().chunk([doc])) == []


# -- a table row is not a hiding place for PII -------------------------------

_SSN = "123 45 6789"
_IBAN = "DE89370400440532013000"
_PROSE = f"Ada Lovelace, SSN {_SSN}, IBAN {_IBAN}."
_TABLE = (
    "| Name | SSN | IBAN |\n"
    "|---|---|---|\n"
    f"| Ada Lovelace | {_SSN} | {_IBAN} |"
)


def _chunk_then_redact(text: str) -> ChunkRecord:
    """One document all the way through ``chunk.markdown`` -> ``redaction.hybrid_rule``."""
    chunks = list(MarkdownChunker().chunk([_doc([text])]))
    assert len(chunks) == 1, f"fixture should be a single chunk, got {len(chunks)}"
    return list(HybridPIIRedactor().redact(chunks))[0]


def test_table_row_pii_is_masked_identically_to_prose() -> None:
    """The same SSN and IBAN must be masked in a table row as in a sentence.

    They were not. The chunker's table-pipe pattern deleted each pipe together with the
    whitespace around it, so the row reached the Redactor as
    ``Ada Lovelace123 45 6789DE89370400440532013000``. Welded to the name, the digits were
    refused by ``ssn_nodash``'s letter-glue guard and the IBAN's leading ``\b`` could not
    fire between ``e`` and ``D`` — the chunk came back with an EMPTY ``pii_spans`` and
    ``masked_content == content``. Prose said ``[SSN]``/``[IBAN]``; the table said nothing,
    for the same data in the same document. This is the end-to-end half of
    ``test_chunking.test_strip_keeps_table_cell_texts_separated``.
    """
    prose = _chunk_then_redact(_PROSE)
    table = _chunk_then_redact(_TABLE)

    assert {span.pii_type for span in prose.pii_spans} == {"ssn", "iban"}
    assert {span.pii_type for span in table.pii_spans} == {
        "ssn",
        "iban",
    }, f"table row PII went undetected: {table.pii_spans}"

    # The values themselves are gone from what a corpus consumer reads.
    for record in (prose, table):
        assert record.masked_content is not None
        assert _SSN not in record.masked_content
        assert _IBAN not in record.masked_content
        assert "[SSN]" in record.masked_content
        assert "[IBAN]" in record.masked_content
        # The clean text is left untouched — masking is additive (W16/ADR-0042).
        assert _SSN in record.content

