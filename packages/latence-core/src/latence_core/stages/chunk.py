"""Built-in reference Chunker Provider (offset-preserving, page-aligned).

The S3 Chunk reference Provider (ADR-0004; ships in core like Source/Export so the
demo runs end to end, ADR-0007 CPU-first, zero extra deps). It wraps the ported
chunking engine (:mod:`latence_core.chunking`, from the pipeline-worker
``orchestrator.py`` asset: 18-pattern markup stripping + token budgeting) and turns
each parsed :class:`DocumentRecord` into retrieval-sized :class:`ChunkRecord`s that
preserve **offsets, page alignment, Provenance and Classification losslessly** (the
CONTEXT definition of Chunk):

* each chunk's Provenance ``char_start``/``char_end`` is the chunk's half-open span
  in the parent document's assembled markdown (from the engine's offset map), so a
  chunk resolves back to its exact source offsets — the S3 round-trip acceptance
  criterion (chunk → offset → original page);
* each chunk's Provenance ``page_start``/``page_end`` is resolved from the parent's
  page map via the core :class:`~latence_core.pagemap.PageOffsetIndex`, so a chunk
  spanning a page boundary records the real source pages;
* each chunk carries its own :class:`~latence_core.contracts.PageSlice` — the document
  page spans overlapping that chunk, and only those (schema v19) — so a downstream
  Stage resolves a sub-chunk offset through
  :class:`~latence_core.page_index.PageIndexResolver` from the record in hand, in any
  order, with any other chunk missing;
* Classification is inherited from the parent document unchanged;
* the parent document's Provenance identity (source_uri, file_name, document_id, …)
  is carried onto every chunk.

A ``PARSE_ERROR`` DocumentRecord (no usable content) yields no chunks — there is
nothing to split, and it must not enter the corpus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.chunking import (
    Chunk,
    ChunkBudget,
    chunk_document,
    chunk_document_by_sentence,
)
from latence_core.contracts import (
    ChunkRecord,
    Disposition,
    OffsetMap,
    OffsetSegment,
    PageSlice,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord


class MarkdownChunker:
    """Chunker Provider: parsed markdown → offset-preserving ChunkRecords.

    Config:
        max_tokens: int  — per-chunk token ceiling (default 512).
        overlap_tokens: int  — tokens of the previous chunk repeated at the start
            of the next, for retrieval recall across a boundary (default 64).
        min_tokens: int  — a trailing remnant below this merges into the previous
            chunk instead of emitting a tiny fragment (default 16).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0.

        The offset-preserving Chunker is a pure-Python part of ``latence-core``
        (PolyForm-Noncommercial-1.0.0, verified), ships no weights (``model_id=None``), is
        deterministic, CPU-only, and has no per-call cost. It processes a stream of documents
        (``batch=True``: it accepts an iterable and folds chunks per document).
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._budget = ChunkBudget(
            max_tokens=int(config.get("max_tokens", 512)),
            overlap_tokens=int(config.get("overlap_tokens", 64)),
            min_tokens=int(config.get("min_tokens", 16)),
        )

    def chunk(self, documents: Iterable[DocumentRecord]) -> Iterator[ChunkRecord]:
        for doc in documents:
            # A document that failed to parse (no content, no page map) is not
            # chunked: nothing to split, and it must not reach the corpus.
            if doc.disposition is Disposition.PARSE_ERROR or not doc.content:
                continue
            yield from self._chunk_one(doc)

    def _chunk_one(self, doc: DocumentRecord) -> Iterator[ChunkRecord]:
        yield from _emit_chunk_records(doc, chunk_document(doc.content, self._budget))


class SentenceWindowChunker:
    """Second reference Chunker Provider: whole-sentence packing (offset-preserving).

    An alternative CPU chunking strategy behind the same :class:`~latence_core.capability.Chunker`
    seam as :class:`MarkdownChunker`, satisfying the identical typed contract — every emitted
    :class:`ChunkRecord` preserves offsets, page alignment, Provenance and Classification
    losslessly (it shares the same :func:`~latence_core.chunking.strip_markup` offset map, so
    the S3 round-trip chunk → offset → original page holds identically). Pure Python, zero extra
    deps, deterministic (ADR-0007 CPU-first, ADR-0016 thin core); no model, so no license note
    (ADR-0012).

    Where :class:`MarkdownChunker` grows a token window to the ceiling then backs up to the best
    break (paragraph → sentence → whitespace, which can land mid-sentence when no earlier break
    exists) with a configurable token overlap, this **packs whole sentences** greedily up to the
    token ceiling with **no overlap**, so a chunk never ends mid-sentence unless a single sentence
    alone exceeds the budget. That trades retrieval recall across a boundary (no overlap) for
    coherent, sentence-aligned passages — a genuinely different segmentation, benchmarked against
    the reference in ``test_chunk_second_provider``.

    Config:
        max_tokens: int  — per-chunk token ceiling (default 512).
        min_tokens: int  — a trailing remnant below this merges into the previous chunk
            (default 16).
        overlap_tokens: int  — accepted for config parity with :class:`MarkdownChunker` but
            IGNORED by this strategy (sentence-aligned boundaries need no overlap). Default 0.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0.

        The sentence-window Chunker is a pure-Python part of ``latence-core``
        (PolyForm-Noncommercial-1.0.0, verified), ships no weights (``model_id=None``), is
        deterministic (greedy sentence packing), CPU-only, does not batch, and has no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._budget = ChunkBudget(
            max_tokens=int(config.get("max_tokens", 512)),
            # Sentence packing has no overlap; keep it 0 regardless of a supplied value so
            # the ChunkBudget invariant (0 <= overlap < max_tokens) always holds.
            overlap_tokens=0,
            min_tokens=int(config.get("min_tokens", 16)),
        )

    def chunk(self, documents: Iterable[DocumentRecord]) -> Iterator[ChunkRecord]:
        for doc in documents:
            if doc.disposition is Disposition.PARSE_ERROR or not doc.content:
                continue
            yield from _emit_chunk_records(
                doc, chunk_document_by_sentence(doc.content, self._budget)
            )


def _emit_chunk_records(doc: DocumentRecord, chunks: list[Chunk]) -> Iterator[ChunkRecord]:
    """Turn a document's engine :class:`Chunk`s into contract :class:`ChunkRecord`s.

    Shared by both reference Chunker Providers so the contract-stamping — page resolution
    through the parent :class:`PageOffsetIndex`, the per-chunk page slice + the per-chunk
    stripped→original offset map, inherited Provenance/Classification — is identical
    regardless of which segmentation strategy produced the chunks.
    """
    # Resolve page alignment through the parent's page map (present after a successful
    # Parse). Absent (shouldn't happen for a PARSED doc), fall back to page 1 so a chunk
    # always carries a page span.
    index = PageOffsetIndex(doc.page_map) if doc.page_map is not None else None
    for i, ch in enumerate(chunks):
        if index is not None:
            page_start = index.page_for_offset(ch.char_start)
            # end offset is exclusive; resolve the last included character.
            page_end = index.page_for_offset(max(ch.char_end - 1, ch.char_start))
        else:  # pragma: no cover - PARSED records always carry a page map
            page_start = page_end = 1
        provenance = doc.provenance.model_copy(
            update={
                "char_start": ch.char_start,
                "char_end": ch.char_end,
                "page_start": page_start,
                "page_end": page_end,
            }
        )
        yield ChunkRecord(
            record_id=f"{doc.record_id}#chunk-{i}",
            document_record_id=doc.record_id,
            index=i,
            provenance=provenance,
            classification=doc.classification,
            content=ch.text,
            media_type="text/markdown",
            token_count=ch.token_count,
            # Stamp THIS chunk's own page spans (schema v19): exactly the document page
            # spans overlapping [char_start, char_end), in the document's original
            # coordinates. Usually one; two when the chunk straddles a page break.
            #
            # This is everything a downstream Stage needs, because the only page question
            # asked of a chunk is sub-chunk: which page does an offset INSIDE it fall on.
            # Carrying the whole document map instead — on every chunk (pre-v18), or on the
            # document's first chunk with the resolver adopting it from the stream (v18) —
            # either duplicated ~30 KB of JSON onto each of ~399 records, or made resolution
            # depend on that one record being seen FIRST and SURVIVING every filtering
            # Stage. A slice keeps the payload win (~2 spans, ~60 bytes) and makes both
            # dependencies inexpressible: the chunk answers for itself.
            page_slice=(
                None
                if doc.page_map is None
                else PageSlice.for_span(doc.page_map, ch.char_start, ch.char_end)
            ),
            # Carry the stripped→original offset map so that same sub-chunk offset resolves
            # to its TRUE original offset (content is markup-stripped, so char_start + local
            # is only a lower bound once markup precedes it).
            offset_map=OffsetMap(
                segments=[
                    OffsetSegment(local_start=ls, doc_start=ds) for ls, ds in ch.offset_map
                ],
                length=len(ch.text),
            ),
        )
