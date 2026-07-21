"""HeaderReferenceExtractor: deterministic zero-shot structured-header cross-reference RE (S4+S5).

Regression for dogfood-3-inspect: an entity-anchored RelationExtractor (learned gliner2 or the
in-core ``relation.pattern``) recovers ZERO of an RFC corpus's explicit ``Obsoletes:``/``Updates:``
header edges, because the targets are bare identifiers (``7230``) never tagged as entities and the
subject (``RFC 9110``) is declared once in the front matter far from the header lines. These tests
pin that the new fused Provider recovers exactly those edges — with real synthesized nodes, real
Evidence offsets/pages, and byte-deterministic output — plus the config-validation and degradation
contracts. Pre-fix (before ``latence_core.stages.header_refs`` exists) this whole module fails to
import; post-fix every case passes.
"""

from __future__ import annotations

import pytest
from latence_core.capability import FusedEntityRelationExtractor
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSpan,
    Provenance,
)
from latence_core.stages.header_refs import HeaderReferenceExtractor

# The RFC front-matter shape build_corpus.py mines for the independent _refs.jsonl gold graph:
# a "Request for Comments: N" self declaration plus bare-number Obsoletes:/Updates: header lines.
_RFC_FRONT_MATTER = (
    "Internet Engineering Task Force (IETF)                      R. Fielding, Ed.\n"
    "Request for Comments: 9110\n"
    "STD: 97\n"
    "Obsoletes: 2818, 7230, 7231, 7232, 7233, 7235\n"
    "Updates: 3864\n"
    "Category: Standards Track\n"
    "\n"
    "                          HTTP Semantics\n"
)

# The zero-shot config a Pipeline supplies for an RFC corpus: nothing RFC-specific is hardcoded in
# core — the self recogniser, the "RFC {id}" rendering, and each header relation are all config.
_RFC_CONFIG = {
    "self_pattern": r"Request for Comments:\s*(\d+)",
    "id_template": "RFC {id}",
    "entity_label": "rfc",
    "relations": {
        "obsoletes": {"header": r"(?im)^Obsoletes:\s*([0-9,\s]+)"},
        "updates": {"header": r"(?im)^Updates:\s*([0-9,\s]+)"},
    },
}


def _chunk(
    text: str,
    *,
    char_start: int = 0,
    record_id: str = "rfc9110#chunk-0",
    document_record_id: str = "rfc9110",
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///corpus-rfc/rfc9110.txt",
        file_name="rfc9110.txt",
        file_type="txt",
        file_size=len(text),
        document_id="sha256:rfc9110",
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id=document_record_id,
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        page_map=page_map,
        offset_map=offset_map,
    )


def _typed_edges(provider: HeaderReferenceExtractor, chunk: ChunkRecord) -> set[tuple[str, str, str]]:
    """The (head_text, label, tail_text) edges the provider emits over one chunk."""
    fused = provider.extract_fused([chunk])
    return {(r.head_text, r.label, r.tail_text) for r in fused.relations}


def test_recovers_obsoletes_and_updates_header_edges() -> None:
    """The core gap: bare-number header cross-references become real self -> target edges."""
    chunk = _chunk(_RFC_FRONT_MATTER)
    edges = _typed_edges(HeaderReferenceExtractor(_RFC_CONFIG), chunk)
    assert edges == {
        ("RFC 9110", "obsoletes", "RFC 2818"),
        ("RFC 9110", "obsoletes", "RFC 7230"),
        ("RFC 9110", "obsoletes", "RFC 7231"),
        ("RFC 9110", "obsoletes", "RFC 7232"),
        ("RFC 9110", "obsoletes", "RFC 7233"),
        ("RFC 9110", "obsoletes", "RFC 7235"),
        ("RFC 9110", "updates", "RFC 3864"),
    }


def test_satisfies_fused_seam_and_synthesizes_nodes() -> None:
    """It fulfils FusedEntityRelationExtractor and emits the self + target nodes the edges need."""
    provider = HeaderReferenceExtractor(_RFC_CONFIG)
    assert isinstance(provider, FusedEntityRelationExtractor)
    fused = provider.extract_fused([_chunk(_RFC_FRONT_MATTER)])
    node_texts = {m.text for m in fused.mentions}
    assert node_texts == {
        "RFC 9110",
        "RFC 2818",
        "RFC 7230",
        "RFC 7231",
        "RFC 7232",
        "RFC 7233",
        "RFC 7235",
        "RFC 3864",
    }
    # Every mention is the configured entity label, and the self mention is created first (index 0).
    assert all(m.label == "rfc" for m in fused.mentions)
    assert fused.mentions[0].text == "RFC 9110"
    # Every relation resolves to real synthesized mentions (no dangling endpoint).
    ids = {m.record_id for m in fused.mentions}
    for r in fused.relations:
        assert r.head_mention_id in ids
        assert r.tail_mention_id in ids
        assert r.head_mention_id != r.tail_mention_id
    # The NER half alone returns the same synthesized nodes.
    assert {m.text for m in provider.extract([_chunk(_RFC_FRONT_MATTER)])} == node_texts


def test_evidence_offsets_point_at_the_real_header_line() -> None:
    """A target's Provenance span covers the exact bare id in the Obsoletes line (real Evidence)."""
    chunk = _chunk(_RFC_FRONT_MATTER)
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([chunk])
    by_text = {m.text: m for m in fused.mentions}
    tail = by_text["RFC 7230"]
    span = _RFC_FRONT_MATTER[tail.provenance.char_start : tail.provenance.char_end]
    assert span == "7230"
    # The subject span covers the self id in the "Request for Comments:" line.
    head = by_text["RFC 9110"]
    assert _RFC_FRONT_MATTER[head.provenance.char_start : head.provenance.char_end] == "9110"


def test_offset_map_lifts_bare_id_to_true_original_offset() -> None:
    """With a stripped-content offset_map, a target resolves to its TRUE original-markdown offset."""
    # content is markup-stripped: the original doc had a 5-char prefix removed before this run, so
    # every char sits 5 later in the original than its chunk-local index (one shifted segment).
    content = "Request for Comments: 9110\nObsoletes: 7230\n"
    offset_map = OffsetMap(
        segments=[OffsetSegment(local_start=0, doc_start=5)], length=len(content)
    )
    chunk = _chunk(content, char_start=5, offset_map=offset_map)
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([chunk])
    tail = next(m for m in fused.mentions if m.text == "RFC 7230")
    local = content.index("7230")
    assert tail.provenance.char_start == local + 5
    assert tail.provenance.char_end == local + 5 + len("7230")


def test_target_page_resolves_through_page_map() -> None:
    """A target's page is resolved from the parent page map at the target's own offset."""
    # A two-page document: the Obsoletes line falls on page 2 (offset >= its char_start).
    text = _RFC_FRONT_MATTER
    split = text.index("Obsoletes:")
    page_map = PageMap(
        pages=[
            PageSpan(page_number=1, char_start=0, char_end=split),
            PageSpan(page_number=2, char_start=split, char_end=len(text)),
        ],
        total_chars=len(text),
    )
    chunk = _chunk(text, page_map=page_map)
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([chunk])
    tail = next(m for m in fused.mentions if m.text == "RFC 7230")
    assert tail.provenance.page_start == 2
    head = next(m for m in fused.mentions if m.text == "RFC 9110")
    assert head.provenance.page_start == 1


def test_deterministic_byte_identical_output() -> None:
    """Identical input + config ⇒ byte-identical records (the Baseline bar)."""
    chunk = _chunk(_RFC_FRONT_MATTER)
    a = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([chunk])
    b = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([chunk])
    assert [m.model_dump() for m in a.mentions] == [m.model_dump() for m in b.mentions]
    assert [r.model_dump() for r in a.relations] == [r.model_dump() for r in b.relations]


def test_no_self_declaration_yields_nothing() -> None:
    """A chunk with header lines but no self declaration has no subject ⇒ no edges, no error."""
    text = "Obsoletes: 7230, 7231\nUpdates: 3864\n"
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([_chunk(text)])
    assert fused.mentions == []
    assert fused.relations == []


def test_self_referential_target_is_dropped() -> None:
    """A header listing the document's own id emits no self-loop edge."""
    text = "Request for Comments: 9110\nObsoletes: 9110, 7230\n"
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([_chunk(text)])
    edges = {(r.head_text, r.label, r.tail_text) for r in fused.relations}
    assert edges == {("RFC 9110", "obsoletes", "RFC 7230")}


def test_non_matching_corpus_is_byte_safe() -> None:
    """A corpus whose text carries none of the configured headers yields nothing (byte-safe)."""
    text = "Alice works for Acme Corp. Bob visited Berlin last week.\n"
    fused = HeaderReferenceExtractor(_RFC_CONFIG).extract_fused([_chunk(text)])
    assert fused.mentions == []
    assert fused.relations == []


def test_target_fan_out_cap_bounds_synthesized_records() -> None:
    """The per-relation target cap bounds a pathological comma-list header."""
    ids = ", ".join(str(n) for n in range(1000, 1050))  # 50 targets
    text = f"Request for Comments: 9110\nObsoletes: {ids}\n"
    cfg = {**_RFC_CONFIG, "max_targets_per_relation": 10}
    fused = HeaderReferenceExtractor(cfg).extract_fused([_chunk(text)])
    assert len(fused.relations) == 10


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"relations": {"obsoletes": {"header": r"(.+)"}}}, "self_pattern"),
        ({"self_pattern": r"(\d+)"}, "relations"),
        ({"self_pattern": r"(\d+)", "relations": {"x": {}}}, "header"),
        ({"self_pattern": r"\d+", "relations": {"x": {"header": r"(.+)"}}}, "capturing group"),
        (
            {"self_pattern": r"(\d+)", "id_template": "no-placeholder", "relations": {"x": {"header": r"(.+)"}}},
            "{id}",
        ),
    ],
)
def test_config_validation(config: dict[str, object], message: str) -> None:
    """Missing/invalid config is a loud construction error, not a silent no-op run."""
    with pytest.raises((ValueError, TypeError)) as exc:
        HeaderReferenceExtractor(config)
    assert message in str(exc.value)
