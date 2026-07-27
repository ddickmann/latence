"""InlineReferenceExtractor: deterministic zero-shot INLINE-CITATION cross-reference RE (S4+S5).

Regression for the dogfood measured gap. The header-only ``fused_entity_relation.header_refs``
Provider anchors a chunk's self id + a metadata-header line (``Obsoletes:``/``Updates:``) that must
BOTH sit in the same front-matter chunk, so it recovers header edges beautifully but structurally
CANNOT see a document's *inline body* citations ("RFC 7230", "[RFC9110]") — a body chunk carries no
self declaration, so header_refs yields nothing for it. On a real RFC corpus that is 1039 of 1105
gold reference edges (~14% recall).

``InlineReferenceExtractor`` closes exactly that gap: it resolves each DOCUMENT's self id ONCE (from
a front-matter self declaration, or from the provenance file name like ``rfc9110.txt``) and mines
every chunk's body for inline "RFC <n>" citations, emitting ``references(source_doc, cited_doc)``
edges — including for body chunks that have no self declaration of their own. These tests pin:

* an inline citation in a chunk WITHOUT the self id still produces a ``references`` edge (the whole
  point — the header-only Provider produces nothing here);
* self-citations and ids already carried as ``obsoletes``/``updates`` header edges are skipped (no
  duplicate/contradictory edges);
* the source id resolves from the provenance FILE NAME when a chunk carries no front-matter self;
* byte-deterministic output.

Pre-build (before ``latence_core.stages.inline_refs`` exists) this whole module fails to import;
post-build every case passes.
"""

from __future__ import annotations

from latence_core.capability import FusedEntityRelationExtractor
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    PageSlice,
    PageSpan,
    Provenance,
)
from latence_core.stages.header_refs import HeaderReferenceExtractor
from latence_core.stages.inline_refs import InlineReferenceExtractor

# The zero-shot config a Pipeline supplies for an RFC corpus. Nothing RFC-specific is hardcoded in
# core: the self recogniser, the file-name fallback, the inline citation pattern, the "RFC {id}"
# rendering, and the header relations to exclude are all Stage config.
_RFC_CONFIG = {
    "self_pattern": r"(?im)^Request for Comments:\s*(\d+)",
    "source_id_pattern": r"(?i)rfc[\s_-]?(\d+)",
    "citation_pattern": r"(?i)RFCs?[\s-]?(\d{3,5})",
    "relation_label": "references",
    "id_template": "RFC {id}",
    "entity_label": "rfc",
    "exclude_relations": {
        "obsoletes": {"header": r"(?im)^Obsoletes:\s*([\d,\s]+)"},
        "updates": {"header": r"(?im)^Updates:\s*([\d,\s]+)"},
    },
    "target_pattern": r"(\d+)",
}

# The header-only config that recovers ONLY the front-matter Obsoletes:/Updates: edges.
_HEADER_CONFIG = {
    "self_pattern": r"(?im)^Request for Comments:\s*(\d+)",
    "id_template": "RFC {id}",
    "entity_label": "rfc",
    "relations": {
        "obsoletes": {"header": r"(?im)^Obsoletes:\s*([\d,\s]+)"},
        "updates": {"header": r"(?im)^Updates:\s*([\d,\s]+)"},
    },
}

# A synthetic multi-chunk RFC document:
#   chunk 0 — the front matter: the self declaration + an Obsoletes header line (a header edge);
#   chunk 1 — a BODY chunk with inline citations and NO self declaration of its own — the case the
#             header-only extractor cannot see;
#   chunk 2 — another BODY chunk citing more RFCs, plus a self-citation (RFC 9110) to be skipped and
#             a re-cite of RFC 7230 (already an Obsoletes header id) to be skipped.
_FRONT_MATTER = (
    "Internet Engineering Task Force (IETF)                      R. Fielding, Ed.\n"
    "Request for Comments: 9110\n"
    "Obsoletes: 7230\n"
    "Category: Standards Track\n"
    "\n"
    "                          HTTP Semantics\n"
)
_BODY_ONE = (
    "As described in RFC 3986, a URI reference is resolved against a base. "
    "See also [RFC5234] for the ABNF core rules used throughout this document.\n"
)
_BODY_TWO = (
    "The message framing in RFC 9112 supersedes the older grammar. "
    "This document (RFC 9110) obsoletes RFC 7230, so RFC 7230 is not a plain reference here.\n"
)


def _chunk(text: str, *, index: int, char_start: int, file_name: str = "rfc9110.txt") -> ChunkRecord:
    prov = Provenance(
        source_uri=f"file:///corpus-rfc/{file_name}",
        file_name=file_name,
        file_type="txt",
        file_size=len(text),
        document_id="sha256:rfc9110",
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"rfc9110#chunk-{index}",
        document_record_id="rfc9110",
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        # v19: the chunk carries its own page spans. This fixture is a one-page document,
        # so each chunk's slice is that page, covering the chunk's own extent.
        page_slice=PageSlice(
            pages=[PageSpan(page_number=1, char_start=char_start, char_end=char_start + len(text))],
            total_chars=char_start + len(text),
        ),
    )


def _rfc_document() -> list[ChunkRecord]:
    """The front-matter chunk + two body chunks, at contiguous document offsets."""
    c0 = _chunk(_FRONT_MATTER, index=0, char_start=0)
    c1 = _chunk(_BODY_ONE, index=1, char_start=len(_FRONT_MATTER))
    c2 = _chunk(_BODY_TWO, index=2, char_start=len(_FRONT_MATTER) + len(_BODY_ONE))
    return [c0, c1, c2]


def _edges(provider: InlineReferenceExtractor, chunks: list[ChunkRecord]) -> set[tuple[str, str, str]]:
    fused = provider.extract_fused(chunks)
    return {(r.head_text, r.label, r.tail_text) for r in fused.relations}


def test_header_only_extractor_misses_inline_body_citations() -> None:
    """Documents the gap: the header-only Provider recovers the Obsoletes edge but ZERO inline ones.

    This is why InlineReferenceExtractor exists — driving the SAME multi-chunk document through the
    header-only extractor yields only the front-matter Obsoletes: edge; every inline body citation
    (RFC 3986 / RFC 5234 / RFC 9112) is invisible to it.
    """
    header = HeaderReferenceExtractor(_HEADER_CONFIG)
    edges = {(r.head_text, r.label, r.tail_text) for r in header.extract_fused(_rfc_document()).relations}
    assert edges == {("RFC 9110", "obsoletes", "RFC 7230")}
    assert not any(label == "references" for _, label, _ in edges)


def test_inline_citation_without_self_id_still_produces_edge() -> None:
    """THE POINT: a body chunk carrying no self declaration still yields references edges.

    chunk 1 and chunk 2 have no ``Request for Comments:`` line — the header-only extractor produces
    nothing for them — yet the document's inline citations become source_doc -> cited edges because
    the self id is resolved once at the document level (from chunk 0) and carried forward.
    """
    edges = _edges(InlineReferenceExtractor(_RFC_CONFIG), _rfc_document())
    assert ("RFC 9110", "references", "RFC 3986") in edges  # body chunk 1, no self id present
    assert ("RFC 9110", "references", "RFC 5234") in edges  # body chunk 1, bracket form [RFC5234]
    assert ("RFC 9110", "references", "RFC 9112") in edges  # body chunk 2


def test_self_and_header_ids_are_skipped() -> None:
    """Self-citations (RFC 9110) and header-dup ids (RFC 7230, an Obsoletes id) are NOT references."""
    edges = _edges(InlineReferenceExtractor(_RFC_CONFIG), _rfc_document())
    cited = {tail for _, label, tail in edges if label == "references"}
    assert "RFC 9110" not in cited  # the document's own id — no self-loop
    assert "RFC 7230" not in cited  # already an Obsoletes header edge — no contradictory reference
    assert cited == {"RFC 3986", "RFC 5234", "RFC 9112"}


def test_source_id_resolves_from_file_name_when_no_front_matter() -> None:
    """With no front-matter self declaration anywhere, the source id resolves from the file name.

    A body-only document (all chunks lack ``Request for Comments:``) still gets its source id from
    the ``rfc9110.txt`` provenance file name via ``source_id_pattern`` — so its inline citations are
    still recovered rather than silently dropped.
    """
    body_only = [_chunk(_BODY_ONE, index=0, char_start=0)]  # no front matter anywhere in the doc
    edges = _edges(InlineReferenceExtractor(_RFC_CONFIG), body_only)
    cited = {tail for _, label, tail in edges if label == "references"}
    assert cited == {"RFC 3986", "RFC 5234"}
    assert all(head == "RFC 9110" for head, _, _ in edges)  # id came from rfc9110.txt


def test_unresolvable_source_yields_nothing() -> None:
    """No front-matter self AND a file name the source_id_pattern cannot parse ⇒ no edges, no error."""
    chunks = [_chunk(_BODY_ONE, index=0, char_start=0, file_name="notes.txt")]
    fused = InlineReferenceExtractor(_RFC_CONFIG).extract_fused(chunks)
    assert fused.mentions == []
    assert fused.relations == []


def test_satisfies_fused_seam_and_synthesizes_nodes() -> None:
    """It fulfils FusedEntityRelationExtractor; every edge resolves to real synthesized mentions."""
    provider = InlineReferenceExtractor(_RFC_CONFIG)
    assert isinstance(provider, FusedEntityRelationExtractor)
    fused = provider.extract_fused(_rfc_document())
    ids = {m.record_id for m in fused.mentions}
    assert fused.mentions[0].text == "RFC 9110"  # the source-doc node is created first (index 0)
    assert all(m.label == "rfc" for m in fused.mentions)
    for r in fused.relations:
        assert r.head_mention_id in ids
        assert r.tail_mention_id in ids
        assert r.head_mention_id != r.tail_mention_id
    # The NER half alone returns the same synthesized nodes.
    assert {m.text for m in provider.extract(_rfc_document())} == {m.text for m in fused.mentions}


def test_evidence_offset_points_at_the_real_inline_citation() -> None:
    """A cited node's Provenance span covers the exact digits of its first inline occurrence."""
    fused = InlineReferenceExtractor(_RFC_CONFIG).extract_fused(_rfc_document())
    tail = next(m for m in fused.mentions if m.text == "RFC 3986")
    # chunk 1 begins at len(_FRONT_MATTER); the "3986" sits at its body offset.
    doc_text = _FRONT_MATTER + _BODY_ONE + _BODY_TWO
    assert doc_text[tail.provenance.char_start : tail.provenance.char_end] == "3986"


def test_deterministic_byte_identical_output() -> None:
    """Identical input + config ⇒ byte-identical records (the Baseline bar)."""
    chunks = _rfc_document()
    a = InlineReferenceExtractor(_RFC_CONFIG).extract_fused(chunks)
    b = InlineReferenceExtractor(_RFC_CONFIG).extract_fused(chunks)
    assert [m.model_dump() for m in a.mentions] == [m.model_dump() for m in b.mentions]
    assert [r.model_dump() for r in a.relations] == [r.model_dump() for r in b.relations]


def test_distinct_cited_id_is_one_node_per_document() -> None:
    """A cited id that appears in multiple body chunks folds to ONE node + ONE edge (deduped)."""
    repeated = [
        _chunk(_FRONT_MATTER, index=0, char_start=0),
        _chunk("See RFC 3986 for details.\n", index=1, char_start=len(_FRONT_MATTER)),
        _chunk("RFC 3986 again, and RFC 3986 once more.\n", index=2, char_start=len(_FRONT_MATTER) + 30),
    ]
    fused = InlineReferenceExtractor(_RFC_CONFIG).extract_fused(repeated)
    tails = [r.tail_text for r in fused.relations if r.tail_text == "RFC 3986"]
    assert tails == ["RFC 3986"]  # exactly one edge despite three occurrences
    nodes = [m for m in fused.mentions if m.text == "RFC 3986"]
    assert len(nodes) == 1
