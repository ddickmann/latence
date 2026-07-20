"""Presenting a benchmark document to the chunk-consuming Stages.

Entity Extraction, Relation Extraction, Disambiguation and Redaction all consume
:class:`~latence_core.contracts.ChunkRecord`s, while every labelled benchmark corpus is a piece
of plain text with character offsets into *that* text. This module is the one place the two are
reconciled, and it does so by making the mapping the identity: one chunk per document, whose
``content`` **is** the document text, with an identity ``offset_map`` and a single-page
``page_map``.

That is what keeps the scoring honest. Offsets emitted by the Stages
(``EntityMention.provenance.char_start``, ``PIISpan.char_start``) are then directly comparable
with the corpus's gold offsets — no re-alignment heuristic sits between the Provider's output
and the gold standard, so a scoring bug cannot be mistaken for a Provider result. Chunking
strategy is deliberately *not* benchmarked here: it is a separate Stage with its own Providers,
and mixing it in would make an extraction number unattributable.
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSpan,
    Provenance,
)


def as_chunk(
    document_id: str,
    text: str,
    *,
    language: str = "und",
    sensitivity: str = "internal",
    source_system: str = "latence-benchmark",
) -> ChunkRecord:
    """Wrap one benchmark document as a single :class:`ChunkRecord` covering all of ``text``.

    :param document_id: Stable id for the document; the chunk's ``record_id`` derives from it,
        so two runs over the same corpus produce identical ids and identical results.
    :param language: BCP-47-ish tag from the corpus (``"de"``, ``"en"``), carried on the
        Classification so a Provider that varies by language sees the real value.
    """
    provenance = Provenance(
        source_uri=f"benchmark:///{document_id}",
        file_name=f"{document_id}.txt",
        file_type="txt",
        file_size=len(text.encode("utf-8")),
        document_id=document_id,
        source_system=source_system,
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"{document_id}#chunk:0",
        document_record_id=document_id,
        index=0,
        provenance=provenance,
        classification=Classification(language=language, sensitivity=sensitivity),
        content=text,
        token_count=len(text.split()),
        # Identity map: chunk-local offset == document offset, so gold offsets from the corpus
        # and Stage-emitted offsets live in the same coordinate system.
        offset_map=(
            OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text))
            if text
            else None
        ),
        page_map=(
            PageMap(
                pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
                total_chars=len(text),
            )
            if text
            else None
        ),
    )
