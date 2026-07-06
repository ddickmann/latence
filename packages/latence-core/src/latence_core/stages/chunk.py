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
* Classification is inherited from the parent document unchanged;
* the parent document's Provenance identity (source_uri, file_name, document_id, …)
  is carried onto every chunk.

A ``PARSE_ERROR`` DocumentRecord (no usable content) yields no chunks — there is
nothing to split, and it must not enter the corpus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.chunking import ChunkBudget, chunk_document
from latence_core.contracts import ChunkRecord, Disposition
from latence_core.pagemap import PageOffsetIndex

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
        chunks = chunk_document(doc.content, self._budget)
        # Resolve page alignment through the parent's page map (present after a
        # successful Parse). Absent (shouldn't happen for a PARSED doc), fall back
        # to page 1 so a chunk always carries a page span.
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
            )
