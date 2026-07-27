"""R2-invariants_and_adrs-4 — the ADR set may not document a page carriage the code abandoned.

ADR-0017 makes the committed files the source of truth, and the ADR set is the design half of
that: a Provider author (first- or third-party) is told to read it before writing a Capability.
The page carriage has been rewritten twice, and each rewrite made the previous ADR prose not
merely stale but *actively wrong*:

* **as first shipped** — the parent document's whole ``PageMap`` rode **every** ``ChunkRecord``,
  and ADR-0022 stated as its decision "Chunk now carries the parent document's ``page_map``
  forward on each ``ChunkRecord`` (optional field; absent ⇒ the mention degrades to the
  inherited chunk range)";
* **schema v18** — the map rode a document's **first** chunk only, so that "absent ⇒ inherit"
  rule silently mis-paged every mention on every later chunk (audit R1 found the pattern
  reproduced from the ADR in four shipped ecosystem Providers and the PII redactors);
* **schema v19** — no map rides any chunk at all. Each chunk carries its OWN
  :class:`~latence_core.contracts.PageSlice`: exactly the page spans overlapping that chunk,
  in document coordinates. Resolution is a property of the record, so it cannot depend on
  stream order or on another chunk surviving.

An author following the superseded prose writes ``PageOffsetIndex(chunk.page_map)`` — which
under v19 does not even exist on the contract, and whose intent (degrade to the chunk's
inherited page range) is the exact ADR-0021/0031 violation. So: **no ADR may state, in
present-tense normative voice, that the parent document's page MAP rides a ``ChunkRecord``**,
and any ADR discussing chunk-level page carriage must name the mechanism that is actually
shipped.

**Anti-tautology.** Nothing about the carriage is hard-coded here. :func:`_observed_carriage`
*runs* the reference Chunk Provider over a multi-page document and observes what comes back on
the records. The documentation gates are predicated on that observation: if the framework ever
legitimately returns to carrying whole page maps on chunks, the observation changes and the
gates stand down (:func:`_skip_unless_slice_carriage`) instead of pinning prose to a constant.
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

#: Any spelling of the page MAP, in prose or as the field name. Deliberately does NOT match
#: ``page_slice``/"page slice" — the slice is the current mechanism, not a superseded one.
_PAGE_MAP = re.compile(r"page[ _\-]?maps?", re.I)

#: Verbs that make a sentence a claim about the map *travelling on a record*, as opposed to a claim
#: about producing it (ADR-0020) or resolving through it (ADR-0023/0028).
_CARRIAGE_VERB = (
    r"\b(?:carr(?:y|ies|ied|ying)|rid(?:e|es|ing)|rode|stamp(?:s|ed|ing)?|"
    r"duplicat(?:e|es|ed)|copie[sd]|copy)\b"
)

_CHUNK = r"\b(?:chunks?|`?ChunkRecord`?s?)\b"
_MAP = r"page[ _\-]?maps?"

#: A carriage claim about a chunk, in any of the three word orders, where the thing CARRIED is
#: the whole page MAP. Under v19 every such claim is superseded — on each chunk (as first
#: shipped) and on the first chunk (v18) alike — so one pattern covers both abandoned shapes.
#: The proximity bounds are what keep it a claim about the map's carriage rather than a sentence
#: that happens to mention a map, a chunk and a verb.
_MAP_RIDES_CHUNK = re.compile(
    _MAP + r"[^.]{0,60}?" + _CARRIAGE_VERB + r"[^.]{0,60}?" + _CHUNK
    + r"|" + _CARRIAGE_VERB + r"[^.]{0,60}?" + _MAP + r"[^.]{0,60}?" + _CHUNK
    + r"|" + _CHUNK + r"[^.]{0,60}?" + _CARRIAGE_VERB + r"[^.]{0,60}?" + _MAP,
    re.I,
)

#: A weaker probe used only by the "name the mechanism" gate: any carriage claim about a chunk,
#: in a sentence that also mentions a page map.
_ANY_CHUNK_CARRIAGE = re.compile(
    _CARRIAGE_VERB + r"[^.]{0,80}?" + _CHUNK + r"|" + _CHUNK + r"[^.]{0,40}?" + _CARRIAGE_VERB,
    re.I,
)

#: Markers that take a sentence out of present-tense normative voice — it describes what the
#: design used to do, or DENIES that the carriage exists. Either way it is not a prescription.
#: A bare "v18" is deliberately NOT such a marker: v18 is now the NAME of a superseded mechanism,
#: so "since schema v18 the map rides the first chunk" is exactly what this gate must catch.
#: A past-tense carriage verb IS one: "ChunkRecord already carried the page map forward" is
#: history by construction.
_HISTORICAL = re.compile(
    r"(?:used to|previously|no longer|supersed\w+|abandon\w+|originally|as first shipped|"
    r"before C1|pre-v18|pre-v19|until v19|until schema v19|\bstopped\b|"
    r"was\b[^.]{0,40}\b(?:replaced|removed|deleted)"
    # denial
    r"|\bno\b|\bnot\b|\bnever\b|\bnothing\b"
    # past tense
    r"|\b(?:carried|rode|stamped|duplicated|copied)\b)",
    re.I,
)

#: The v19 mechanism, named. An ADR that discusses chunk-level page carriage must say which
#: mechanism it means, and the only shipped one is the per-chunk slice.
_V19_MECHANISM = re.compile(r"v19|page[ _\-]?slice", re.I)


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


def _observed_carriage() -> tuple[set[int], set[int], int]:
    """OBSERVE the real Chunk Stage: which chunks carry a SLICE, and which carry a whole MAP?"""
    doc = _multi_page_document()
    chunks = list(
        MarkdownChunker({"max_tokens": 40, "overlap_tokens": 8, "min_tokens": 4}).chunk([doc])
    )
    assert len(chunks) > 3, "fixture must produce several chunks for the carriage question to exist"
    with_slice = {c.index for c in chunks if c.page_slice is not None}
    with_map = {c.index for c in chunks if getattr(c, "page_map", None) is not None}
    return with_slice, with_map, len(chunks)


def _skip_unless_slice_carriage() -> None:
    _, with_map, _ = _observed_carriage()
    if with_map:  # pragma: no cover - only if whole-map carriage is ever reinstated
        pytest.skip(
            "the Chunk Stage carries a whole page map on a chunk again — the page-map carriage "
            "prose in the ADR set would then be CURRENT, and these gates must be rewritten with "
            "that change"
        )


def test_every_chunk_carries_its_own_page_slice_and_no_chunk_carries_the_document_map() -> None:
    """The code fact the documentation gates below are derived from (schema v19)."""
    with_slice, with_map, total = _observed_carriage()
    assert with_slice == set(range(total)), (
        f"expected a page slice on every chunk; {total} chunks carried one on {sorted(with_slice)}"
    )
    assert not with_map, "no chunk may carry the whole document's page map (v19)"


def test_no_adr_asserts_a_page_map_riding_a_chunk() -> None:
    """An ADR may describe the abandoned carriages as history; it may not still PRESCRIBE one.

    Both abandoned shapes are caught by the one gate, because under v19 neither is true: the
    whole page map rides no chunk at all — not the first, not any other.
    """
    _skip_unless_slice_carriage()
    offenders: list[str] = []
    for adr in _adr_files():
        for sentence in _sentences(adr.read_text(encoding="utf-8")):
            if not _MAP_RIDES_CHUNK.search(sentence):
                continue
            if _HISTORICAL.search(sentence):
                continue
            offenders.append(f"{adr.name}: {sentence.strip()[:220]}")
    assert not offenders, (
        "these ADRs still state, in present-tense normative voice, that the parent document's "
        "page MAP rides a ChunkRecord — on each chunk (as first shipped) or on the document's "
        "first chunk (v18). Both are gone: since v19 a chunk carries only its own PageSlice. A "
        "Provider author following them writes PageOffsetIndex(chunk.page_map) and degrades a "
        "mention to its chunk's inherited page range (ADR-0021/0031 violation). Amend the ADR "
        "(past voice + the v19 mechanism), do not delete the history:\n  " + "\n  ".join(offenders)
    )


def test_every_adr_that_claims_chunk_page_carriage_names_the_v19_mechanism() -> None:
    """An ADR that discusses page carriage on a chunk must say WHICH carriage it means.

    History is welcome, but an ADR that only ever describes the superseded shapes leaves a reader
    with no way to find the one that ships.
    """
    _skip_unless_slice_carriage()
    silent: list[str] = []
    for adr in _adr_files():
        body = adr.read_text(encoding="utf-8")
        claims = [
            s for s in _sentences(body) if _PAGE_MAP.search(s) and _ANY_CHUNK_CARRIAGE.search(s)
        ]
        if claims and not _V19_MECHANISM.search(body):
            silent.append(f"{adr.name}: {claims[0].strip()[:220]}")
    assert not silent, (
        "these ADRs describe a page map riding a ChunkRecord without anywhere naming the schema "
        "v19 mechanism (the per-chunk PageSlice) that replaced it, so a reader cannot tell which "
        "carriage they mean:\n  " + "\n  ".join(silent)
    )
