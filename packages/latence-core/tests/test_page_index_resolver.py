"""PageIndexResolver: document-scoped offset→page resolution behind one seam (C1).

Two acceptance criteria drive this file:

1. **The document map is no longer duplicated onto every chunk.** A chunk checkpoint
   line must not carry its document's whole page map — only the document's *first*
   chunk does (schema v18). For a many-page document that map was ~92% of every line.
2. **Offset→page resolution stays exact.** Provenance is load-bearing (ADR-0021/0031):
   shedding the payload must not move a single page number. The resolver reproduces,
   byte-for-byte, the pages a per-chunk ``PageOffsetIndex`` produced.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
)
from latence_core.page_index import PageIndexResolver
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.chunk import MarkdownChunker
from latence_core.stages.entity import GazetteerEntityExtractor

# Five pages. The budget below (300 tokens vs ~110 tokens of page) is deliberately wide
# enough that the last chunk STRADDLES a page boundary — the only shape where a mention's
# OWN page differs from the page range it would inherit from its chunk, and therefore the
# only shape that can detect a provenance regression from shedding the per-chunk map.
_PAGES = [
    (f"Acme Corp filed the quarterly statement about revenue growth on page {i}. " * 10)
    for i in range(1, 6)
]
_PAGES[1] = "Globex reported the second quarter costs and Acme Corp restated margins. " * 10
_PAGES[3] = "Initech closed the fourth period while Acme Corp audited the ledger. " * 10

_CONFIG = {
    "labels": {
        "ORG": {"terms": ["Acme Corp", "Globex", "Initech"], "confidence": 0.9},
    }
}


def _document(record_id: str = "doc-1") -> DocumentRecord:
    assembly = PageMap.from_page_texts([p.strip() for p in _PAGES])
    prov = Provenance(
        source_uri="file:///report.pdf",
        file_name="report.pdf",
        file_type="pdf",
        file_size=4096,
        document_id=f"sha256:{record_id}",
        char_start=0,
        char_end=assembly.page_map.total_chars,
        page_start=1,
        page_end=len(assembly.page_map.pages),
    )
    return DocumentRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


def _chunks(doc: DocumentRecord) -> list[ChunkRecord]:
    # A budget wide enough that a chunk spans a page boundary (see _PAGES).
    chunker = MarkdownChunker({"max_tokens": 300, "overlap_tokens": 0, "min_tokens": 4})
    return list(chunker.chunk([doc]))


# -- AC1: the checkpoint line no longer carries the document map ---------------


def test_chunk_checkpoint_line_omits_document_page_map() -> None:
    """Only a document's FIRST chunk carries the page map; the rest ship none.

    The checkpoint is JSONL, so this is measured the way the Runner writes it: the
    serialized line of a non-first chunk must contain no page-map payload at all.
    """
    doc = _document()
    chunks = _chunks(doc)
    assert len(chunks) > 2, "need several chunks for the duplication to be meaningful"

    carriers = [c for c in chunks if c.page_map is not None]
    assert [c.index for c in carriers] == [0]

    for chunk in chunks[1:]:
        line = chunk.model_dump_json()
        assert '"page_map":null' in line or '"page_map": null' in line
        # The map's actual payload — the page spans — must be absent from the line.
        assert '"page_number"' not in line
        assert '"total_chars"' not in line


def test_chunk_stream_bytes_collapse_without_the_duplicated_map() -> None:
    """The per-document map is paid for once, not once per chunk.

    Measured against the pre-C1 stream (the same records with the map re-attached), so
    the saving is the real redundancy: ``per-chunk map bytes × (chunks - 1)``. On the
    measured 519-page document (~399 chunks, ~30 KB of map) that was ~92% of every line.
    """
    doc = _document()
    chunks = _chunks(doc)
    assert doc.page_map is not None
    full = [c.model_copy(update={"page_map": doc.page_map}) for c in chunks]

    bytes_shed = sum(len(c.model_dump_json().encode("utf-8")) for c in chunks)
    bytes_full = sum(len(c.model_dump_json().encode("utf-8")) for c in full)
    per_chunk_map_bytes = len(full[1].model_dump_json().encode("utf-8")) - len(
        chunks[1].model_dump_json().encode("utf-8")
    )
    assert per_chunk_map_bytes > 0
    assert bytes_full - bytes_shed == per_chunk_map_bytes * (len(chunks) - 1)


# -- AC2: offset→page resolution is still exact -------------------------------


def test_resolver_matches_per_chunk_page_offset_index_exactly() -> None:
    """Every offset of a multi-page document resolves to the same page as before.

    Ground truth is a :class:`PageOffsetIndex` built directly from the document's own
    page map — the resolution the pre-C1 code performed per chunk.
    """
    doc = _document()
    assert doc.page_map is not None
    truth = PageOffsetIndex(doc.page_map)
    chunks = _chunks(doc)
    resolver = PageIndexResolver()

    for chunk in chunks:
        start = chunk.provenance.char_start
        end = chunk.provenance.char_end
        assert start is not None and end is not None
        for offset in range(start, end):
            assert resolver.page_for(chunk, offset) == truth.page_for_offset(offset)
        assert resolver.pages_for_span(chunk, start, end) == (
            truth.page_for_offset(start),
            truth.page_for_offset(max(end - 1, start)),
        )
    # Exactly one index was built for the whole document.
    assert resolver.documents == 1
    assert resolver.unresolved == 0


def test_mention_pages_are_identical_with_and_without_per_chunk_maps() -> None:
    """A mention still cites ITS OWN page when its chunk carries no page map.

    This is the regression the shed payload could have caused: a mention on a later page
    of a page-straddling chunk must not fall back to the chunk's inherited ``page_start``.
    """
    doc = _document()
    shed = _chunks(doc)
    # The pre-C1 stream: every chunk carrying its document's whole map.
    assert doc.page_map is not None
    full = [c.model_copy(update={"page_map": doc.page_map}) for c in shed]

    extractor = GazetteerEntityExtractor(_CONFIG)
    pages_full = [
        (m.record_id, m.provenance.page_start, m.provenance.page_end)
        for m in extractor.extract(full)
    ]
    pages_shed = [
        (m.record_id, m.provenance.page_start, m.provenance.page_end)
        for m in extractor.extract(shed)
    ]
    assert pages_full == pages_shed
    assert pages_full, "the fixture must produce mentions"

    # And the guard that makes the assertion above meaningful: at least one mention's own
    # page genuinely differs from the page its chunk would have handed down.
    by_chunk = {c.record_id: c for c in shed}
    differs = [
        m
        for m in extractor.extract(shed)
        if by_chunk[m.chunk_record_id].provenance.page_start != m.provenance.page_start
    ]
    assert differs, "fixture no longer exercises a page-straddling chunk"


def test_entity_stage_resolves_pages_from_the_documents_first_chunk() -> None:
    """The Stage keeps exact pages for chunks that carry NO map of their own.

    The pre-C1 Stage built a ``PageOffsetIndex`` from ``chunk.page_map`` and nothing
    else, so a chunk without one silently inherited its whole page range. The resolver
    memoises the map the document's first chunk carried, so the last chunk — which here
    straddles a page boundary — still cites each mention's own page.
    """
    doc = _document()
    assert doc.page_map is not None
    truth = PageOffsetIndex(doc.page_map)
    # Explicitly the v18 stream shape, independent of what the Chunk Stage stamps.
    stream = [
        c if c.index == 0 else c.model_copy(update={"page_map": None}) for c in _chunks(doc)
    ]

    mentions = list(GazetteerEntityExtractor(_CONFIG).extract(stream))
    assert mentions
    straddling = 0
    by_chunk = {c.record_id: c for c in stream}
    for mention in mentions:
        start = mention.provenance.char_start
        end = mention.provenance.char_end
        assert start is not None and end is not None
        assert mention.provenance.page_start == truth.page_for_offset(start)
        assert mention.provenance.page_end == truth.page_for_offset(max(end - 1, start))
        if by_chunk[mention.chunk_record_id].provenance.page_start != mention.provenance.page_start:
            straddling += 1
    assert straddling, "fixture no longer exercises a page-straddling chunk"


# -- the seam itself ----------------------------------------------------------


def test_resolver_seeded_from_documents_resolves_chunks_carrying_no_map() -> None:
    """The authoritative route: seed from the parsed documents, chunks carry nothing."""
    doc = _document()
    assert doc.page_map is not None
    stripped = [c.model_copy(update={"page_map": None}) for c in _chunks(doc)]

    resolver = PageIndexResolver.from_documents([doc])
    truth = PageOffsetIndex(doc.page_map)
    for chunk in stripped:
        start = chunk.provenance.char_start
        assert start is not None
        assert resolver.page_for(chunk, start) == truth.page_for_offset(start)
    assert resolver.unresolved == 0


def test_resolver_degrades_and_counts_when_no_map_is_available() -> None:
    """No map anywhere ⇒ None (the caller degrades), and the gap is COUNTED, not silent."""
    doc = _document()
    stripped = [c.model_copy(update={"page_map": None}) for c in _chunks(doc)]
    resolver = PageIndexResolver()

    assert resolver.page_for(stripped[0], 0) is None
    assert resolver.pages_for_span(stripped[0], 0, 10) is None
    assert resolver.unresolved == 2
    assert resolver.documents == 0


def test_resolver_memoises_one_index_per_document() -> None:
    """A second document gets its own index; neither is rebuilt per chunk."""
    doc_a = _document("doc-a")
    doc_b = _document("doc-b")
    chunks = _chunks(doc_a) + _chunks(doc_b)
    resolver = PageIndexResolver()
    for chunk in chunks:
        start = chunk.provenance.char_start
        assert start is not None
        assert resolver.page_for(chunk, start) is not None
    assert resolver.documents == 2


def test_resolver_aggregates_drift_across_documents() -> None:
    """Drift is rolled up corpus-wide — impossible while each chunk built its own index."""
    doc_a = _document("doc-a")
    doc_b = _document("doc-b")
    resolver = PageIndexResolver.from_documents([doc_a, doc_b])
    chunk_a = _chunks(doc_a)[0]
    chunk_b = _chunks(doc_b)[0]

    assert doc_a.page_map is not None
    beyond = doc_a.page_map.total_chars + 500
    resolver.page_for(chunk_a, 0)
    resolver.page_for(chunk_a, beyond)  # past the end ⇒ fuzzy recovery, recorded
    resolver.page_for(chunk_b, 0)

    diag = resolver.diagnostics
    assert diag.resolutions == 3
    assert diag.recovered == 1
    assert diag.max_drift_chars > 0
    assert diag.clean is False


def test_resolver_never_raises_on_out_of_range_offsets() -> None:
    """Provenance resolution must never be the thing that fails a run."""
    doc = _document()
    chunk = _chunks(doc)[0]
    resolver = PageIndexResolver.from_documents([doc])
    assert doc.page_map is not None

    assert resolver.page_for(chunk, -1_000) == 1
    assert resolver.page_for(chunk, doc.page_map.total_chars + 1_000) == len(doc.page_map.pages)
    assert resolver.pages_for_span(chunk, 5, 5) is not None


def test_parse_error_document_contributes_no_page_map() -> None:
    doc = _document()
    broken = doc.model_copy(
        update={"page_map": None, "disposition": Disposition.PARSE_ERROR, "content": ""}
    )
    resolver = PageIndexResolver.from_documents([broken])
    assert resolver.documents == 0


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_resolution_is_deterministic(repeats: int) -> None:
    """Identical inputs ⇒ identical outputs, however many times the resolver is used."""
    doc = _document()
    chunks = _chunks(doc)
    expected: list[tuple[int, int] | None] = []
    for _ in range(repeats):
        resolver = PageIndexResolver()
        got = [
            resolver.pages_for_span(
                c, c.provenance.char_start or 0, c.provenance.char_end or 0
            )
            for c in chunks
        ]
        if not expected:
            expected = got
        assert got == expected
