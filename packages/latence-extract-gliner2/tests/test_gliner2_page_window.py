"""The page-index window must not evict a document the Provider is still resolving (audit R4).

This is the ADR-0021/0031 regression the C1 seam exists to prevent, exercised on the SHIPPED
wiring rather than on a hand-built resolver: the real :func:`_iter_batches` (which observes every
chunk, including the ones ``min_chars`` drops), a bare ``PageIndexResolver()`` — the exact
construction at ``provider.py`` — and the real :func:`_mention_provenance`. No torch, no gliner2:
the failure is in the streaming/observation shape, not in the model.

The shape that breaks it is ordinary, not adversarial: a corpus of mostly SHORT documents
(memos, notes, covering letters — each one page, each below the extractor's ``min_chars`` floor)
with occasional long ones. Every short document must still be observed (its map rides its first
chunk and that chunk is exactly the one the floor drops), so it consumed a slot in a window that
counted DOCUMENTS. Hundreds of them streamed past while ONE batch of long-document chunks
accumulated, evicting the long document whose chunks the batch was still holding — and those
chunks carry no map of their own (schema v18), so their mentions inherited the chunk's whole page
range. Not a missing page: a WRONG, plausible-looking one, in the knowledge graph, with no
production reader for the counter that noticed.
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
    Ground truth is a per-document ``PageOffsetIndex``, i.e. what pre-C1 code produced when the
    map rode every chunk.
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
    assert resolver.evicted_reentries == 0
    assert resolver.unresolved == 0


def test_the_short_documents_are_what_fills_the_window() -> None:
    """Pin the mechanism, so a future change cannot make the test above pass for free.

    The defect needs (a) the Provider to observe far more DISTINCT documents than its batch
    holds, and (b) a long document's chunks to straddle a batch boundary, so the chunk carrying
    its map is consumed one batch before the chunks that need it. Both are properties of the
    fixture; if either stops holding, the regression test is no longer testing anything.
    """
    chunks, _ = _corpus()
    resolver = PageIndexResolver()

    batches: list[list[ChunkRecord]] = []
    observed_per_batch: list[int] = []
    seen = 0
    for batch in _iter_batches(iter(chunks), _MAX_BATCH_SIZE, _MIN_CHARS, resolver):
        batches.append(batch)
        # The trailing partial batch is whatever the corpus had left; only FULL ones say
        # anything about how many documents a batch's worth of stream carries.
        if len(batch) == _MAX_BATCH_SIZE:
            observed_per_batch.append(resolver.documents - seen)
        seen = resolver.documents

    assert len(batches) > 1, "a single batch cannot straddle a boundary"
    assert min(observed_per_batch) > 64, (
        "accumulating ONE batch must push more distinct documents through the resolver than "
        "the old 64-document window held; otherwise nothing is ever evicted mid-batch"
    )
    per_batch = [{c.document_record_id for c in b} for b in batches]
    assert max(len(d) for d in per_batch) <= _MAX_BATCH_SIZE
    assert any(a & b for a, b in zip(per_batch, per_batch[1:], strict=False)), (
        "no document's chunks straddle a batch boundary — nothing would need a map that was "
        "absorbed during an earlier batch"
    )
