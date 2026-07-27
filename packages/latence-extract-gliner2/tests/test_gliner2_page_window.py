"""A streamed mixed-length corpus must never mis-page a long document (audit R4).

This is the ADR-0021/0031 regression the C1 seam exists to prevent, exercised on the SHIPPED
wiring rather than on a hand-built resolver: the real :func:`_iter_batches` (including the
``min_chars`` drop), a bare ``PageIndexResolver()`` — the exact construction at ``provider.py``
— and the real :func:`_mention_provenance`. No torch, no gliner2: the failure was in the
streaming shape, not in the model.

The shape that broke it is ordinary, not adversarial: a corpus of mostly SHORT documents
(memos, notes, covering letters — each one page, each below the extractor's ``min_chars``
floor) with occasional long ones. Under schema v18 a document's page map rode its FIRST chunk
and the resolver memoised one index per document behind an LRU window, so hundreds of short
documents streaming past while ONE batch of long-document chunks accumulated evicted the long
document the batch was still holding — and its later chunks carried no map of their own, so
their mentions inherited the chunk's whole page range. Not a missing page: a WRONG,
plausible-looking one, in the knowledge graph, with no production reader for the counter that
noticed.

Since v19 every chunk carries its OWN page spans and the resolver holds no document state, so
there is no window, nothing to evict, and no stream position for an answer to depend on. The
corpus-scale property is unchanged and still worth running at full size: every mention cites
the page its own span falls on, at the shipped batch ceiling, over 400 interleaved documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
)
from latence_core.page_index import PageIndexResolver
from latence_core.pagemap import PageOffsetIndex
from latence_core.stages.chunk import MarkdownChunker
from latence_extract_gliner2.provider import (
    _MAX_BATCH_SIZE,
    _iter_batches,
    _mention_provenance,
)

if TYPE_CHECKING:
    from latence_core.contracts import ChunkRecord

_COVER = "Cover sheet."
_BODY = "Acme Corp reported the quarterly revenue growth figures for the period under review. " * 3
#: Long enough to survive the chunker's ``min_tokens``, short enough to be dropped by the
#: Provider's ``min_chars`` floor — the memo/note shape that fills a real corpus.
_STUB = "Short internal note for record."

_MIN_CHARS = 40
_DOCUMENTS = 400
_LONG_EVERY = 9
_BODY_PAGES = 25


def _document(i: int, *, long_form: bool) -> DocumentRecord:
    pages = [_COVER, *([_BODY.strip()] * _BODY_PAGES)] if long_form else [_STUB]
    assembly = PageMap.from_page_texts(pages)
    prov = Provenance(
        source_uri=f"file:///doc{i:04d}.pdf",
        file_name=f"doc{i:04d}.pdf",
        file_type="pdf",
        file_size=4096,
        document_id=f"sha256:doc{i:04d}",
        char_start=0,
        char_end=assembly.page_map.total_chars,
        page_start=1,
        page_end=len(assembly.page_map.pages),
    )
    return DocumentRecord(
        record_id=f"doc{i:04d}",
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


def _corpus() -> tuple[list[ChunkRecord], dict[str, PageOffsetIndex]]:
    docs = [_document(i, long_form=(i % _LONG_EVERY == 0)) for i in range(_DOCUMENTS)]
    truth = {d.record_id: PageOffsetIndex(d.page_map) for d in docs if d.page_map is not None}
    chunker = MarkdownChunker({"max_tokens": 300, "overlap_tokens": 0, "min_tokens": 4})
    return list(chunker.chunk(docs)), truth


def test_a_streamed_short_document_corpus_never_mis_pages_a_long_one() -> None:
    """Every mention cites the page its OWN span falls on, at the shipped batch ceiling.

    ``batch_size: 64`` is in range for an operator — the Provider clamps to ``_MAX_BATCH_SIZE``,
    which IS 64 — so this is a configuration the shipped stack accepts, not an abuse of it.
    Ground truth is a per-document ``PageOffsetIndex`` over the PARENT DOCUMENT's whole map,
    i.e. what pre-C1 code produced when that map rode every chunk — an independent oracle, not
    the mechanism under test.
    """
    chunks, truth = _corpus()
    resolver = PageIndexResolver()  # the bare construction every shipped Provider uses

    straddling = 0
    wrong: list[str] = []
    for batch in _iter_batches(iter(chunks), _MAX_BATCH_SIZE, _MIN_CHARS, resolver):
        for chunk in batch:
            start = chunk.provenance.char_start or 0
            end = chunk.provenance.char_end or start
            # A mention on the LAST page of a page-straddling chunk: the only shape whose own
            # page differs from the range it would inherit, and so the only one that can see
            # this defect at all.
            span_start = max(end - 30, start)
            index = truth[chunk.document_record_id]
            expected = (
                index.page_for_offset(span_start),
                index.page_for_offset(max(end - 1, span_start)),
            )
            if (chunk.provenance.page_start, chunk.provenance.page_end) == expected:
                continue
            straddling += 1
            provenance = _mention_provenance(
                chunk.provenance,
                span_start,
                end,
                resolver.pages_for_span(chunk, span_start, end),
            )
            got = (provenance.page_start, provenance.page_end)
            if got != expected:
                wrong.append(f"{chunk.record_id}: stamped {got}, true {expected}")

    assert straddling > 100, (
        f"only {straddling} mentions could detect the defect — the fixture stopped producing "
        "page-straddling chunks and the assertion below is vacuous"
    )
    assert wrong == [], (
        f"{len(wrong)} of {straddling} mentions were stamped with a WRONG page range because "
        f"the resolver evicted a document the batch was still resolving: {wrong[:5]}"
    )
    # Every resolution was EXACT: the drift roll-up is the corpus-level statement that no
    # offset had to be recovered by the fuzzy fallback (ADR-0034 counts it rather than hiding
    # it, so a zero here is a claim with a reader).
    assert resolver.diagnostics.max_drift_chars == 0
    assert resolver.diagnostics.recovered == 0


def test_the_fixture_produces_the_shape_that_could_detect_the_defect() -> None:
    """Pin the fixture, so the test above cannot pass for free.

    The historical defect needed (a) far more DISTINCT documents streaming past than one batch
    holds, and (b) a long document's chunks straddling a batch boundary, so a chunk consumed in
    one batch was needed by chunks in the next. v19 makes that arrangement harmless, but it is
    still the arrangement worth resolving at scale — and both are properties of the FIXTURE, so
    if either stops holding, the regression test above is no longer testing anything.
    """
    chunks, _ = _corpus()
    resolver = PageIndexResolver()

    batches: list[list[ChunkRecord]] = []
    documents_per_batch: list[int] = []
    for batch in _iter_batches(iter(chunks), _MAX_BATCH_SIZE, _MIN_CHARS, resolver):
        batches.append(batch)
        documents_per_batch.append(len({c.document_record_id for c in batch}))

    assert len(batches) > 1, "a single batch cannot straddle a boundary"
    # Distinct documents streamed past while ONE batch accumulated, counted over the WHOLE
    # stream rather than over the resolver (which no longer knows what a document is).
    distinct_documents = len({c.document_record_id for c in chunks})
    assert distinct_documents > 64, (
        "the corpus must push more distinct documents through the Provider than the old "
        "64-document window held; otherwise nothing would ever have been evicted mid-batch"
    )
    per_batch = [{c.document_record_id for c in b} for b in batches]
    assert max(documents_per_batch) <= _MAX_BATCH_SIZE
    assert any(a & b for a, b in zip(per_batch, per_batch[1:], strict=False)), (
        "no document's chunks straddle a batch boundary — nothing would need page spans first "
        "seen during an earlier batch"
    )
