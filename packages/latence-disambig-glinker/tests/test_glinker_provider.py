"""GLinkerDisambiguator: the enterprise headline — acronym/expansion merges, context-distinct stays.

The two cases the embedding-only ER got wrong, now correct through the neural GLinker linking + the
audited resolver (with the heavy stack faked offline):

* **BAFA ↔ Bundesamt für Wirtschaft und Ausfuhrkontrolle MERGES** — the neural biencoder
  context-links the acronym mention and its full-name mention to ONE canonical entity (the primary
  GLinker mechanism an embedding-only ER, which embeds surfaces, misses).
* **IBM ↔ International Business Machines MERGES with an AUDITED MergeDecision** — the resolver's
  acronym rung folds them (reason ``acronym_expansion``, applied), surfaced in the record's
  ``merge_decisions`` (no silent over-merge, §19).
* **IBM ↔ Microsoft-in-similar-context does NOT merge** — the neural linker keeps distinct entities
  apart even when their surrounding text is near-identical (the over-merge an embedding-only ER
  produces), so they land in SEPARATE canonical entities.
"""

from __future__ import annotations

from typing import Any

import latence_disambig_glinker.provider as provider_mod
import pytest
from conftest import FakeGLinkerBackend, chunk, mention
from latence_core.capability import Disambiguator
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    DisambiguationScope,
    EntityMention,
    MergeStrategy,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSlice,
    Provenance,
)
from latence_core.disambiguation import normalize_text
from latence_disambig_glinker.backends import InferenceBackend, LinkedEntity
from latence_disambig_glinker.provider import GLinkerDisambiguator

# Two documents with NEAR-IDENTICAL context around IBM (d3) and Microsoft (d4) — the exact
# similar-context trap that over-merges an embedding-only ER; the neural linker keeps them apart.
_C1 = "BAFA reviewed the export licence application this quarter."
_C2 = "The Bundesamt für Wirtschaft und Ausfuhrkontrolle regulates German dual-use exports."
_C3 = "IBM builds enterprise software and cloud services for large customers."
_C4 = "Microsoft builds enterprise software and cloud services for large customers."


def _fixture() -> tuple[list[EntityMention], list[ChunkRecord]]:
    chunks = [
        chunk("ch1", "d1", _C1),
        chunk("ch2", "d2", _C2),
        chunk("ch3", "d3", _C3),
        chunk("ch4", "d4", _C4),
    ]
    mentions = [
        mention("m1", "d1", "ch1", "BAFA", _C1.index("BAFA")),
        mention(
            "m2", "d2", "ch2",
            "Bundesamt für Wirtschaft und Ausfuhrkontrolle",
            _C2.index("Bundesamt"),
        ),
        mention("m3", "d3", "ch3", "IBM", _C3.index("IBM")),
        mention("m4", "d3", "ch3", "International Business Machines", 0),
        mention("m5", "d4", "ch4", "Microsoft", _C4.index("Microsoft")),
    ]
    return mentions, chunks


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the neural GLinkerBackend for the faithful offline fake with the linking decisions."""
    FakeGLinkerBackend.context_rules = []  # reset the shared class-level rule table
    FakeGLinkerBackend.link_map = {
        # Context-links the acronym AND its full name to ONE entry (the GLinker mechanism).
        "bafa": ("grp_bafa", "organization"),
        "bundesamt für wirtschaft und ausfuhrkontrolle": ("grp_bafa", "organization"),
        # IBM and its expansion link to SEPARATE entries -> the RESOLVER acronym rung merges them
        # (an audited MergeDecision), proving the audited-fusion path.
        "ibm": ("grp_ibm", "organization"),
        "international business machines": ("grp_ibm_full", "organization"),
        # Microsoft links to its OWN entry -> stays distinct from IBM despite the identical context.
        "microsoft": ("grp_ms", "organization"),
    }
    monkeypatch.setattr(provider_mod, "GLinkerBackend", FakeGLinkerBackend)


def _config() -> dict[str, object]:
    # Isolate the linking + string-cascade path (embedding rung off) for a crisp, deterministic
    # headline; the resolver embedding rung is exercised separately in test_entity_resolver.
    return {"use_embedding_merge": False}


def test_satisfies_disambiguator_protocol() -> None:
    assert isinstance(GLinkerDisambiguator(_config()), Disambiguator)


def test_headline_acronym_merges_and_context_distinct_stays_apart() -> None:
    mentions, chunks = _fixture()
    disambiguator = GLinkerDisambiguator(_config())
    records = list(disambiguator.disambiguate(mentions, [], chunks))

    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity]
    # 3 canonical entities: {BAFA + Bundesamt…}, {IBM + International Business Machines}, {Microsoft}.
    assert len(entities) == 3

    members_by_entity = {e.record_id: set(e.member_mention_ids) for e in entities}

    # BAFA ↔ Bundesamt… merged (neural context-linking).
    bafa = _entity_containing(members_by_entity, "m1")
    assert bafa is not None and "m2" in members_by_entity[bafa]

    # IBM ↔ International Business Machines merged (resolver acronym rung).
    ibm = _entity_containing(members_by_entity, "m3")
    assert ibm is not None and "m4" in members_by_entity[ibm]

    # Microsoft is a SEPARATE entity — NOT swept in with IBM despite identical context.
    ms = _entity_containing(members_by_entity, "m5")
    assert ms is not None and ms != ibm
    assert "m3" not in members_by_entity[ms] and "m5" not in members_by_entity[ibm]


def test_audited_merge_decision_is_surfaced_no_silent_over_merge() -> None:
    mentions, chunks = _fixture()
    disambiguator = GLinkerDisambiguator(_config())
    records = list(disambiguator.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.entity]

    ibm_entity = next(
        e for e in entities if "m3" in set(e.member_mention_ids) and "m4" in set(e.member_mention_ids)
    )
    # The acronym fold is AUDITED (reason + confidence + provenance-bearing evidence), applied.
    acronym_decisions = [
        d for d in ibm_entity.merge_decisions if d.strategy is MergeStrategy.ACRONYM
    ]
    assert acronym_decisions, "acronym merge must be surfaced in merge_decisions (no silent merge)"
    d = acronym_decisions[0]
    assert d.applied is True
    assert 0.0 <= d.confidence <= 1.0
    assert d.evidence.mention_ids  # provenance-bearing


def test_fuzzy_ocr_variant_merge_is_surfaced_in_quality_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX-3: two OCR-variant person mentions fuse via the fuzzy rung, as an AUDITED MergeDecision.

    Neither surface links neurally (empty link_map) so each unique text becomes its own direct-KB
    canonical entity; the resolver's fuzzy rung then folds them (ratio 0.90 ≥ default 0.84) into ONE
    person. The merge is surfaced in the record's ``merge_decisions`` (reason ``fuzzy_edit_distance``,
    strategy SUBSTRING — the lexical bucket), so nothing is silently over-merged (§19).
    """
    FakeGLinkerBackend.context_rules = []
    FakeGLinkerBackend.link_map = {}  # neither links neurally → per-unique-text direct-KB entities
    monkeypatch.setattr(provider_mod, "GLinkerBackend", FakeGLinkerBackend)

    c1 = "Kai Kölsch signed the shareholder resolution before the notary."
    c2 = "Kai Kolsch signed the shareholder resolution before the notary."
    chunks = [chunk("ch1", "d1", c1), chunk("ch2", "d2", c2)]
    mentions = [
        mention("m1", "d1", "ch1", "Kai Kölsch", c1.index("Kai"), label="person"),
        mention("m2", "d2", "ch2", "Kai Kolsch", c2.index("Kai"), label="person"),
    ]
    disambiguator = GLinkerDisambiguator({"use_embedding_merge": False})  # fuzzy ON by default
    records = list(disambiguator.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity]

    assert len(entities) == 1  # the two OCR variants fused into one canonical person
    entity = entities[0]
    assert set(entity.member_mention_ids) == {"m1", "m2"}  # no dropped provenance
    fuzzy = [d for d in entity.merge_decisions if d.reason == "fuzzy_edit_distance"]
    assert fuzzy, "the fuzzy merge must be surfaced in merge_decisions (no silent over-merge)"
    d = fuzzy[0]
    assert d.strategy is MergeStrategy.SUBSTRING  # the lexical bucket the fuzzy rung maps to
    assert d.applied is True
    assert 0.0 <= d.confidence <= 1.0
    assert d.evidence.mention_ids  # provenance-bearing


def test_records_are_contract_valid_and_deterministically_ordered() -> None:
    mentions, chunks = _fixture()
    disambiguator = GLinkerDisambiguator(_config())
    records = list(disambiguator.disambiguate(mentions, [], chunks))
    # Every ENTITY record carries a canonical entity with page-accurate Provenance + Evidence.
    for r in records:
        if r.scope is DisambiguationScope.ENTITY:
            assert r.entity is not None
            assert r.provenance.document_id
            assert r.entity.evidence.mention_ids
    # Deterministic ENTITY order (canonical-text sorted): re-running yields the identical stream.
    again = list(disambiguator.disambiguate(mentions, [], chunks))
    assert [r.record_id for r in records] == [r.record_id for r in again]


def test_empty_mentions_is_graceful_empty() -> None:
    disambiguator = GLinkerDisambiguator(_config())
    assert list(disambiguator.disambiguate([], [], [])) == []


def _entity_containing(members: dict[str, set[str]], mention_id: str) -> str | None:
    for eid, ids in members.items():
        if mention_id in ids:
            return eid
    return None


# --------------------------------------------------------------------------- #
# C-F2 / C-F3 / A-F5: context is genuinely READ + fed — anti-false-green.
# --------------------------------------------------------------------------- #


def test_backend_link_is_context_conditional() -> None:
    """The faithful fake links the SAME surface to DIFFERENT entities by context (reads m['context']).

    Mirrors the real backend's ``m.get("context", m["text"])``: a polysemous surface disambiguates on
    its surrounding text. If the backend stopped consulting context, both would link identically and
    this fails — the property that makes the context claim non-vacuous.
    """
    FakeGLinkerBackend.context_rules = [
        ("mercury", "planet", ("grp_planet", "location")),
        ("mercury", "element", ("grp_element", "substance")),
    ]
    backend = FakeGLinkerBackend()
    results = backend.link_mentions(
        [
            {"text": "Mercury", "context": "Mercury is the smallest planet orbiting the sun."},
            {"text": "Mercury", "context": "Mercury is a heavy metallic element, liquid at room temp."},
        ],
        entity_kb=[],
    )
    assert results[0] is not None and results[0].entity_id == "grp_planet"
    assert results[1] is not None and results[1].entity_id == "grp_element"


def test_provider_feeds_window_context_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Provider passes the ±window chunk context (not the bare surface) to the backend.

    A capturing backend records the ``context`` it received; asserting it carries the surrounding
    sentence proves the context wiring is live end-to-end, so a regression that fed the bare surface
    would be caught (A-F5).
    """
    captured: dict[str, str] = {}

    class _CapturingBackend(InferenceBackend):
        LINKER_MODEL = "fake/linker"
        RERANKER_MODEL = "fake/rerank"

        def __init__(self, **_kw: Any) -> None:
            pass

        @property
        def name(self) -> str:
            return "capturing"

        def warmup(self, entity_kb: list[dict[str, Any]]) -> None:
            return None

        def link_mentions(
            self,
            mentions: list[dict[str, Any]],
            entity_kb: list[dict[str, Any]],
            threshold: float = 0.3,
            batch_size: int = 32,
        ) -> list[LinkedEntity | None]:
            for m in mentions:
                captured[str(m["text"]).lower()] = str(m.get("context", ""))
            return [None] * len(mentions)

    ctx = "BAFA reviewed the export licence application this quarter."
    chunks = [chunk("ch1", "d1", ctx)]
    mentions = [mention("m1", "d1", "ch1", "BAFA", ctx.index("BAFA"))]

    monkeypatch.setattr(provider_mod, "GLinkerBackend", _CapturingBackend)
    disambiguator = GLinkerDisambiguator({"use_embedding_merge": False})
    list(disambiguator.disambiguate(mentions, [], chunks))

    assert "export licence" in captured["bafa"]  # the ±window, not just "BAFA"


# --------------------------------------------------------------------------- #
# A-F1: a confusable surface does not collide onto an existing framework record_id.
# --------------------------------------------------------------------------- #


def test_confusable_surface_no_record_id_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    """'ﬁle' and 'file' resolve to ONE canonical entity — one record_id, no dropped provenance (A-F1).

    Both surfaces normalise (NFKC) to 'file', so the framework derives ``entity:file`` for each. The
    resolver now groups them into one cluster upstream, so exactly one ENTITY record is emitted and
    both mentions survive on it.
    """
    FakeGLinkerBackend.context_rules = []
    FakeGLinkerBackend.link_map = {}  # neither links neurally -> direct-KB clusters by surface
    monkeypatch.setattr(provider_mod, "GLinkerBackend", FakeGLinkerBackend)

    c1 = "The ﬁle was archived."  # U+FB01 ligature
    c2 = "The file was archived."
    chunks = [chunk("ch1", "d1", c1), chunk("ch2", "d2", c2)]
    mentions = [
        mention("m1", "d1", "ch1", "ﬁle", c1.index("ﬁle"), label="artifact"),
        mention("m2", "d2", "ch2", "file", c2.index("file"), label="artifact"),
    ]
    disambiguator = GLinkerDisambiguator({"use_embedding_merge": False})
    records = list(disambiguator.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity]

    assert len(entities) == 1  # not two clusters colliding on entity:file
    entity = entities[0]
    assert entity.record_id == f"entity:{normalize_text('file')}"
    assert set(entity.member_mention_ids) == {"m1", "m2"}  # no dropped provenance
    # No two ENTITY records share a record_id (the collision the audit found).
    ids = [r.record_id for r in records if r.scope is DisambiguationScope.ENTITY]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# A-F5: the offset_map inverse-resolution branch of _mention_context.
# --------------------------------------------------------------------------- #


def test_mention_context_uses_offset_map_inverse_resolution() -> None:
    """A markup-stripping OffsetMap inverts the mention's original offset to a content-local one.

    Original markdown ``**BAFA** reviewed…`` strips to content ``BAFA reviewed…`` — the leading
    ``**`` (2 chars) is gone. The mention's ``char_start`` is in ORIGINAL coords (2, after ``**``);
    the naive ``char_start - chunk_start`` (=2) would index content[2:] = 'FA reviewed…' and drop
    'BA'. The offset_map inverts 2 -> local 0, so the window opens at 'BAFA'. This exercises the
    inverse-resolution branch (all other fixtures use offset_map=None).
    """
    cls = Classification(language="en", category="news", sensitivity="internal")
    content = "BAFA reviewed the export licence application in detail this quarter."
    # content[0]='B' corresponds to original-markdown offset 2 (the '**' before it was stripped).
    offset_map = OffsetMap(
        segments=[OffsetSegment(local_start=0, doc_start=2)], length=len(content)
    )
    prov = Provenance(
        source_uri="file://d1.txt", file_name="d1.txt", file_type="txt", file_size=1000,
        document_id="d1", char_start=0, char_end=len(content) + 2, page_start=1, page_end=1,
    )
    chunk_rec = ChunkRecord(
        record_id="ch1", document_record_id="d1:doc", index=0, content=content, token_count=20,
        provenance=prov, classification=cls,
        page_slice=PageSlice.for_span(
            PageMap.from_page_texts([content]).page_map, 0, len(content)
        ),
        offset_map=offset_map,
    )
    m = EntityMention(
        record_id="m1", document_record_id="d1:doc", chunk_record_id="ch1", index=0,
        label="organization", text="BAFA", confidence=0.9,
        provenance=Provenance(
            source_uri="file://d1.txt", file_name="d1.txt", file_type="txt", file_size=1000,
            document_id="d1", char_start=2, char_end=6, page_start=1, page_end=1,
        ),
        classification=cls,
    )
    disambiguator = GLinkerDisambiguator({"use_embedding_merge": False})
    ctx = disambiguator._mention_context(m, {"ch1": chunk_rec})
    # Coordinate-correct: the window opens at the full 'BAFA' surface (not the naive 'FA').
    assert ctx.startswith("BAFA")
    assert "export licence" in ctx


# -- v20 (ADR-0057): link confidence is persisted, not dropped -------------------


def test_apply_linking_results_carries_the_link_score() -> None:
    """The linker has always computed a link score and then dropped it (the ticket-01
    finding); v20 carries it onto the mention so Hyperedge gamma has a data source."""
    from latence_disambig_glinker.backends import LinkedEntity
    from latence_disambig_glinker.linker import _apply_linking_results
    from latence_disambig_glinker.models import MentionRecord

    mentions = [
        MentionRecord(mention_id="m1", dataset_id="ds", document_id="d1",
                      text="Acme", label="ORG"),
        MentionRecord(mention_id="m2", dataset_id="ds", document_id="d1",
                      text="Beta", label="ORG"),
    ]
    linked = [
        LinkedEntity(entity_id="e1", label="Acme", description="", score=0.87,
                     entity_type="ORG", aliases=[], source="glinker"),
        None,
    ]
    updated = _apply_linking_results(mentions, linked)
    assert updated[0].canonical_entity_id == "e1"
    assert updated[0].link_confidence == 0.87
    # unresolved mention: confidence stays None (unknown), never 0.0
    assert updated[1].canonical_entity_id is None
    assert updated[1].link_confidence is None


def test_build_canonical_entities_collects_member_link_confidences() -> None:
    from latence_disambig_glinker.linker import EntityLinker
    from latence_disambig_glinker.models import MentionRecord

    linker = EntityLinker.__new__(EntityLinker)  # engine method under test, no backend needed
    mentions = [
        MentionRecord(mention_id="m1", dataset_id="ds", document_id="d1", text="Acme",
                      label="ORG", score=0.9, canonical_entity_id="e1", link_confidence=0.87),
        MentionRecord(mention_id="m2", dataset_id="ds", document_id="d2", text="Acme",
                      label="ORG", score=0.8, canonical_entity_id="e1", link_confidence=0.75),
        MentionRecord(mention_id="m3", dataset_id="ds", document_id="d1", text="Gamma",
                      label="ORG", score=0.7),  # unresolved: provisional entity, no link score
    ]
    entities = EntityLinker._build_canonical_entities(linker, mentions, "ds")
    by_name = {e.canonical_name: e for e in entities}
    assert by_name["Acme"].link_confidences == {"m1": 0.87, "m2": 0.75}
    assert by_name["Gamma"].link_confidences == {}
