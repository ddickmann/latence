"""End-to-end GLinkerDisambiguator: entities + external-KB link + normalized relations, offline.

Exercises the inherited record-building machinery through the enterprise path: canonical entities
with an external-KB ``kb_id`` link, RELATION-scope records with a normalised + remapped label, the
Granite-STS embedding merge rung ON (via the in-core deterministic hashing embedder) proving it
neither crashes nor over-merges, and byte-stable determinism over two runs.
"""

from __future__ import annotations

import latence_disambig_glinker.provider as provider_mod
import pytest
from conftest import FakeGLinkerBackend, chunk, mention, relation
from latence_core.contracts import ChunkRecord, DisambiguationScope, EntityMention, RelationMention
from latence_disambig_glinker.provider import GLinkerDisambiguator

_C1 = "IBM builds enterprise software. International Business Machines partners widely."
_C2 = "Microsoft builds enterprise software for large customers worldwide."


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeGLinkerBackend.context_rules = []
    FakeGLinkerBackend.link_map = {
        "ibm": ("grp_ibm", "organization"),
        "international business machines": ("grp_ibm_full", "organization"),
        "microsoft": ("grp_ms", "organization"),
    }
    monkeypatch.setattr(provider_mod, "GLinkerBackend", FakeGLinkerBackend)


def _config() -> dict[str, object]:
    return {
        # Embedding merge ON via the in-core deterministic hashing embedder (offline default): proves
        # it runs + does not over-merge the distinct IBM/Microsoft entities.
        "use_embedding_merge": True,
        "embedding_merge_threshold": 0.9,
        # External KB for the canonical entities' kb_id link (graceful fallback for unlinked).
        "kb": [
            {"kb_id": "Q2", "name": "International Business Machines", "entity_type": "organization",
             "aliases": ["IBM"]},
        ],
        "kb_threshold": 0.4,
        # Relation-label normalisation + type filter.
        "label_aliases": {"partners with": "partner_of"},
        "allowed_labels": ["partner_of"],
    }


def _fixture() -> tuple[list[EntityMention], list[ChunkRecord], list[RelationMention]]:
    chunks = [chunk("ch1", "d1", _C1), chunk("ch2", "d2", _C2)]
    m_ibm = mention("m1", "d1", "ch1", "IBM", _C1.index("IBM"))
    m_ibm_full = mention("m2", "d1", "ch1", "International Business Machines", _C1.index("International"))
    m_ms = mention("m3", "d2", "ch2", "Microsoft", _C2.index("Microsoft"))
    rels = [relation("r1", "d1", "partners with", m_ibm, m_ms)]
    return [m_ibm, m_ibm_full, m_ms], chunks, rels


def test_entities_link_to_external_kb_and_relations_normalize() -> None:
    mentions, chunks, rels = _fixture()
    disambiguator = GLinkerDisambiguator(_config())
    records = list(disambiguator.disambiguate(mentions, rels, chunks))

    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity]
    # IBM + expansion merged (acronym), Microsoft separate => 2 entities (no over-merge on hashing).
    assert len(entities) == 2
    # The IBM entity links to the external KB entry Q2 (via its 'IBM' alias).
    ibm_entity = next(e for e in entities if "m1" in set(e.member_mention_ids))
    assert ibm_entity.kb_id == "Q2"

    relations = [r.relation for r in records if r.scope is DisambiguationScope.RELATION and r.relation]
    assert len(relations) == 1
    rel = relations[0]
    assert rel.label == "partner_of"  # normalised via label_aliases + allowed_labels type filter
    assert rel.head_entity_id != rel.tail_entity_id  # remapped onto DISTINCT canonical entities


def test_e2e_is_deterministic_over_two_runs() -> None:
    mentions, chunks, rels = _fixture()
    disambiguator = GLinkerDisambiguator(_config())
    first = [r.model_dump_json() for r in disambiguator.disambiguate(mentions, rels, chunks)]
    second = [r.model_dump_json() for r in disambiguator.disambiguate(mentions, rels, chunks)]
    assert first == second
