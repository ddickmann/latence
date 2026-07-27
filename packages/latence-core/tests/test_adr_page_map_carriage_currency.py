"""R2-invariants_and_adrs-4 — the ADR set may not document a page-map carriage the code abandoned.

ADR-0017 makes the committed files the source of truth, and the ADR set is the design half of
that: a Provider author (first- or third-party) is told to read it before writing a Capability.
C1 (schema v18) moved the parent document's ``PageMap`` off every ``ChunkRecord`` and onto a
document's **first** chunk only, putting offset→page resolution behind the document-scoped
:class:`~latence_core.page_index.PageIndexResolver` seam — but it amended no ADR. ADR-0022 was
left stating, as its *decision*:

    Chunk now carries the parent document's ``page_map`` forward on each ``ChunkRecord``
    (optional field; absent ⇒ the mention degrades to the inherited chunk range)

which is not merely stale — it prescribes the exact wrong contract. An author who follows it writes
``PageOffsetIndex(chunk.page_map)`` per chunk and degrades to the chunk's inherited page range when
it is ``None``; under v18 that is *every* chunk after a document's first, so every mention, relation
and PII span on them silently cites a plausible-looking wrong page. R1 found that code in four
shipped ecosystem Providers and in the PII redactors — reproduced from the record, which is why
fixing the call sites without fixing the record fixes nothing durable.

**Anti-tautology.** Nothing about the carriage is hard-coded here. :func:`_carrying_chunk_indices`
*runs* the reference Chunk Provider over a multi-page document and observes which chunk indices come
back with a ``page_map``. The documentation gates are predicated on that observation: if the
framework ever legitimately returns to per-chunk carriage, the observation changes and the gates
stand down (:func:`_skip_unless_single_carrier`) instead of pinning prose to a constant.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
)
from latence_core.stages.chunk import MarkdownChunker

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADR_DIR = _REPO_ROOT / "docs" / "adr"

#: Any spelling of the page map, in prose or as the field name.
_PAGE_MAP = re.compile(r"page[ _\-]?maps?", re.I)

#: Verbs that make a sentence a claim about the map *travelling on a record*, as opposed to a claim
#: about producing it (ADR-0020) or resolving through it (ADR-0023/0028).
_CARRIAGE_VERB = (
    r"\b(?:carr(?:y|ies|ied|ying)|rid(?:e|es|ing)|rode|stamp(?:s|ed|ing)?|"
    r"duplicat(?:e|es|ed)|copie[sd]|copy)\b"
)

#: The SUPERSEDED shape: carried onto each/every/all chunks, or per chunk.
_MULTI_CHUNK_CARRIAGE = re.compile(
    _CARRIAGE_VERB
    + r"[^.]{0,80}?\b(?:each|every|all)\b[^.]{0,30}?\b(?:chunks?|`?ChunkRecord`?s?)\b"
    r"|" + _CARRIAGE_VERB + r"[^.]{0,30}?\bper[ \-]chunk\b",
    re.I,
)

#: Any carriage claim at all, superseded or not — the trigger for the "names the mechanism" gate.
_ANY_CHUNK_CARRIAGE = re.compile(
    _CARRIAGE_VERB + r"[^.]{0,80}?\b(?:chunks?|`?ChunkRecord`?s?)\b"
    r"|\b(?:chunks?|`?ChunkRecord`?s?)\b[^.]{0,40}?" + _CARRIAGE_VERB,
    re.I,
)

#: Markers that put a sentence in the PAST — describing what the design used to do. A sentence in
#: this voice may (and should) restate the superseded mechanism; that is what an amendment is.
_SUPERSESSION = re.compile(
    r"(?:used to|previously|no longer|supersed\w+|originally|as first shipped|before C1|"
    r"pre-v18|until schema v18|v18|was\b[^.]{0,40}\breplaced)",
    re.I,
)

#: The v18 mechanism, named. An ADR that makes a carriage claim must say which one it means.
_V18_MECHANISM = re.compile(r"v18|PageIndexResolver|first\W{0,4}chunk", re.I)


def _sentences(text: str) -> list[str]:
    """Whitespace-normalised sentences. Markdown wraps hard, so newlines are not boundaries.

    Blockquote markers are stripped, not skipped: an amendment block IS the ADR's record, so its
    prose is held to the same standard as the body it corrects.
    """
    unquoted = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    flat = re.sub(r"\s+", " ", unquoted)
    return [s for s in re.split(r"(?<=[.!?])\s+", flat) if s]


def _adr_files() -> list[Path]:
    files = sorted(p for p in _ADR_DIR.glob("[0-9]*.md"))
    assert files, f"no ADRs found under {_ADR_DIR}"
    return files


def _multi_page_document() -> DocumentRecord:
    """A three-page document long enough that the default budget yields several chunks."""
    pages = [
        ("Quarterly revenue for the Northern division rose sharply this period. " * 12).strip(),
        ("The Southern division reported a matching contraction in the same period. " * 12).strip(),
        (
            "Consolidated guidance for the coming year is unchanged from the prior filing. " * 12
        ).strip(),
    ]
    assembly = PageMap.from_page_texts(pages)
    return DocumentRecord(
        record_id="sha256:carriage",
        provenance=Provenance(
            source_uri="file:///carriage.pdf",
            file_name="carriage.pdf",
            file_type="pdf",
            file_size=4096,
            document_id="sha256:carriage",
            char_start=0,
            char_end=assembly.page_map.total_chars,
            page_start=1,
            page_end=len(assembly.page_map.pages),
        ),
        classification=Classification(language="en", category="finance", sensitivity="internal"),
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


def _carrying_chunk_indices() -> tuple[set[int], int]:
    """OBSERVE the real Chunk Stage: which chunk indices come back carrying a page map?"""
    doc = _multi_page_document()
    chunks = list(
        MarkdownChunker({"max_tokens": 40, "overlap_tokens": 8, "min_tokens": 4}).chunk([doc])
    )
    assert len(chunks) > 3, "fixture must produce several chunks for the carriage question to exist"
    return {c.index for c in chunks if c.page_map is not None}, len(chunks)


def _skip_unless_single_carrier() -> None:
    carriers, total = _carrying_chunk_indices()
    if carriers == set(range(total)):  # pragma: no cover - only if v18 is ever reverted
        pytest.skip(
            "the Chunk Stage carries the page map on every chunk again — the per-chunk prose in "
            "the ADR set would then be CURRENT, and this gate must be rewritten with that change"
        )


def test_the_chunk_stage_stamps_the_page_map_on_exactly_one_chunk_per_document() -> None:
    """The code fact the documentation gates below are derived from (schema v18, C1)."""
    carriers, total = _carrying_chunk_indices()
    assert carriers == {0}, (
        f"expected the page map on the first chunk only; {total} chunks carried it on {sorted(carriers)}"
    )


def test_no_adr_asserts_the_superseded_per_chunk_page_map_carriage() -> None:
    """An ADR may describe the old carriage as history; it may not still PRESCRIBE it.

    Fails pre-fix on ADR-0022 (its decision sentence) and ADR-0031 (the precedent it cites from it).
    """
    _skip_unless_single_carrier()
    offenders: list[str] = []
    for adr in _adr_files():
        for sentence in _sentences(adr.read_text(encoding="utf-8")):
            if not _PAGE_MAP.search(sentence):
                continue
            if not _MULTI_CHUNK_CARRIAGE.search(sentence):
                continue
            if _SUPERSESSION.search(sentence):
                continue
            offenders.append(f"{adr.name}: {sentence.strip()[:220]}")
    assert not offenders, (
        "these ADRs still state, in present-tense normative voice, that the parent page map rides "
        "each/every chunk — the mechanism C1 (schema v18) replaced. A Provider author following "
        "them writes PageOffsetIndex(chunk.page_map) per chunk and silently degrades every chunk "
        "after a document's first to its inherited page range (ADR-0021/0031 violation). Amend the "
        "ADR (past voice + the v18 mechanism), do not delete the history:\n  "
        + "\n  ".join(offenders)
    )


def test_every_adr_that_claims_page_map_carriage_names_the_v18_mechanism() -> None:
    """Carriage is now a *conditional* property of a chunk, so an ADR that mentions it must say so.

    Fails pre-fix on ADR-0022, which discusses the carriage at length without naming the first-chunk
    stamp or the ``PageIndexResolver`` seam that makes it resolvable.
    """
    _skip_unless_single_carrier()
    silent: list[str] = []
    for adr in _adr_files():
        body = adr.read_text(encoding="utf-8")
        claims = [
            s for s in _sentences(body) if _PAGE_MAP.search(s) and _ANY_CHUNK_CARRIAGE.search(s)
        ]
        if claims and not _V18_MECHANISM.search(body):
            silent.append(f"{adr.name}: {claims[0].strip()[:220]}")
    assert not silent, (
        "these ADRs describe the page map riding a ChunkRecord without naming the schema-v18 "
        "mechanism (first chunk only) or the PageIndexResolver seam that resolves it, so a reader "
        "cannot tell which carriage they mean:\n  " + "\n  ".join(silent)
    )
