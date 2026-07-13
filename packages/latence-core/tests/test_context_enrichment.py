"""W13 context enrichment: the reference ContextHeaderEnricher over a faithful fake KG.

The unit tests for the OPTIONAL ``context.kg_header`` Provider (ADR-0039): it projects the run's
already-assembled knowledge graph back onto each chunk as a compact, CAPPED, confidence-gated
:class:`~latence_core.contracts.ContextHeader`. The crux is the bloat discipline — the top-k
per-entity cap + the overall triple cap must BITE (a hub entity contributes only its TOP relations),
the confidence gate must drop weak entities/edges, and the build must be byte-identical under seed.

The fake KG is hand-built (not the pipeline) so a hub entity, weak edges, and a low-confidence
mention are all present and the caps/gates are provable.
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.stages.context_enrichment import ContextHeaderEnricher

_CLS = Classification(language="en", sensitivity="public")


def _prov(char_start: int = 0, char_end: int = 10) -> Provenance:
    return Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=char_start,
        char_end=char_end,
        page_start=1,
        page_end=1,
    )


def _chunk(record_id: str, index: int, content: str) -> ChunkRecord:
    return ChunkRecord(
        record_id=record_id,
        provenance=_prov(0, len(content)),
        classification=_CLS,
        document_record_id="d1",
        index=index,
        content=content,
        token_count=max(1, len(content.split())),
    )


def _mention(record_id: str, chunk_id: str, index: int, text: str, confidence: float) -> EntityMention:
    return EntityMention(
        record_id=record_id,
        provenance=_prov(0, len(text)),
        classification=_CLS,
        document_record_id="d1",
        chunk_record_id=chunk_id,
        index=index,
        label="ORG",
        text=text,
        confidence=confidence,
    )


def _node(node_id: str, name: str, label: str, members: list[str], confidence: float = 0.9) -> GraphRecord:
    node = GraphNode(
        record_id=f"node:{node_id}",
        node_id=node_id,
        entity_id=f"ent:{node_id}",
        label=label,
        canonical_name=name,
        member_mention_ids=members,
        confidence=confidence,
        evidence=Evidence(mention_ids=members, confidence=confidence),
        provenance=_prov(),
        classification=_CLS,
    )
    return GraphRecord(
        record_id=f"gr:node:{node_id}",
        scope=GraphScope.NODE,
        node=node,
        provenance=_prov(),
        classification=_CLS,
    )


def _edge(edge_id: str, src: str, tgt: str, label: str, confidence: float) -> GraphRecord:
    edge = GraphEdge(
        record_id=f"edge:{edge_id}",
        edge_id=edge_id,
        relation_id=f"rel:{edge_id}",
        source_node_id=src,
        target_node_id=tgt,
        label=label,
        confidence=confidence,
        evidence=Evidence(mention_ids=[], confidence=confidence),
        provenance=_prov(),
        classification=_CLS,
    )
    return GraphRecord(
        record_id=f"gr:edge:{edge_id}",
        scope=GraphScope.EDGE,
        edge=edge,
        provenance=_prov(),
        classification=_CLS,
    )


def _small_kg() -> tuple[list[ChunkRecord], list[EntityMention], list[GraphRecord]]:
    """Two chunks; Alice—works_for→Acme and Acme—located_in→Berlin, plus Globex."""
    chunks = [
        _chunk("c1", 0, "Alice works for Acme in Berlin."),
        _chunk("c2", 1, "Globex partnered with Acme."),
    ]
    mentions = [
        _mention("m-alice", "c1", 0, "Alice", 0.9),
        _mention("m-acme1", "c1", 1, "Acme", 0.9),
        _mention("m-berlin", "c1", 2, "Berlin", 0.9),
        _mention("m-globex", "c2", 0, "Globex", 0.9),
        _mention("m-acme2", "c2", 1, "Acme", 0.9),
    ]
    graph = [
        _node("n-alice", "Alice", "PERSON", ["m-alice"]),
        _node("n-acme", "Acme", "ORG", ["m-acme1", "m-acme2"]),
        _node("n-berlin", "Berlin", "PLACE", ["m-berlin"]),
        _node("n-globex", "Globex", "ORG", ["m-globex"]),
        _edge("e-works", "n-alice", "n-acme", "works_for", 0.95),
        _edge("e-located", "n-acme", "n-berlin", "located_in", 0.80),
        _edge("e-partner", "n-globex", "n-acme", "partner_of", 0.70),
    ]
    return chunks, mentions, graph


def _hub_kg(
    n_neighbors: int = 8,
) -> tuple[list[ChunkRecord], list[EntityMention], list[GraphRecord]]:
    """A hub entity 'Acme' present in MANY chunks with MANY neighbor relations (the cap target)."""
    chunks = [_chunk(f"c{i}", i, f"Doc {i} mentions Acme.") for i in range(3)]
    mentions = [_mention(f"m-acme-{i}", f"c{i}", 0, "Acme", 0.9) for i in range(3)]
    graph: list[GraphRecord] = [_node("n-acme", "Acme", "ORG", [f"m-acme-{i}" for i in range(3)])]
    # n_neighbors distinct partner orgs of the hub, each an edge Acme —partner_of→ nK with a
    # DISTINCT, descending confidence so the top-k selection is unambiguous.
    for k in range(n_neighbors):
        graph.append(_node(f"n-p{k:02d}", f"Partner{k:02d}", "ORG", [f"m-p{k:02d}"]))
        # confidence 0.99, 0.98, ... strictly descending so the top-k is deterministic + provable.
        graph.append(
            _edge(f"e-p{k:02d}", "n-acme", f"n-p{k:02d}", "partner_of", round(0.99 - 0.01 * k, 4))
        )
    return chunks, mentions, graph


def test_builds_correct_entities_and_triples_per_chunk() -> None:
    """Each chunk's header carries exactly its own canonical entities + their KG-neighbor triples."""
    chunks, mentions, graph = _small_kg()
    out = list(ContextHeaderEnricher({}).enrich(chunks, mentions, graph))
    assert [c.record_id for c in out] == ["c1", "c2"]  # 1:1, order-stable

    h1 = out[0].context_header
    assert h1 is not None
    assert h1.entities == ["Acme", "Alice", "Berlin"]
    assert h1.entity_types == ["ORG", "PERSON", "PLACE"]
    assert h1.kg_node_ids == ["n-acme", "n-alice", "n-berlin"]
    # c1's entities reach all three relations (works_for, located_in, partner_of via Acme).
    assert "Alice —works_for→ Acme" in h1.neighbor_triples
    assert "Acme —located_in→ Berlin" in h1.neighbor_triples
    assert "Globex —partner_of→ Acme" in h1.neighbor_triples
    # Neighbor node ids = the endpoints reached that are NOT c1's own entities (Globex).
    assert h1.neighbor_node_ids == ["n-globex"]

    h2 = out[1].context_header
    assert h2 is not None
    assert h2.entities == ["Acme", "Globex"]
    # c2's entities (Acme, Globex) reach works_for, located_in, partner_of too (all touch Acme).
    assert "Globex —partner_of→ Acme" in h2.neighbor_triples
    # situating_sentence / model_id stay None in the deterministic default (no LLM).
    assert h2.situating_sentence is None
    assert h2.model_id is None


def test_top_k_per_entity_cap_bites_for_a_hub_entity() -> None:
    """A hub entity in many chunks contributes only its TOP-k relations — the hub-entity guard."""
    chunks, mentions, graph = _hub_kg(n_neighbors=8)
    cap = 3
    prov = ContextHeaderEnricher({"max_neighbors_per_entity": cap, "max_triples": 50})
    out = list(prov.enrich(chunks, mentions, graph))

    for chunk in out:
        header = chunk.context_header
        assert header is not None
        # The hub 'Acme' has 8 partner relations but only its TOP-k survive in EVERY chunk.
        assert len(header.neighbor_triples) == cap, header.neighbor_triples
        # The kept triples are the HIGHEST-confidence ones: Partner00..Partner02 (0.99, 0.98, 0.97).
        assert header.neighbor_triples == [
            "Acme —partner_of→ Partner00",
            "Acme —partner_of→ Partner01",
            "Acme —partner_of→ Partner02",
        ]
    # The hub was capped once per chunk it appears in (3 chunks) — the anti-bloat witness fires.
    assert prov.hub_entities_capped == 3


def test_overall_max_triples_cap_bites() -> None:
    """The overall ``max_triples`` cap bounds the whole header, on top of the per-entity cap."""
    chunks, mentions, graph = _hub_kg(n_neighbors=8)
    prov = ContextHeaderEnricher({"max_neighbors_per_entity": 8, "max_triples": 4})
    out = list(prov.enrich(chunks, mentions, graph))
    for chunk in out:
        header = chunk.context_header
        assert header is not None
        assert len(header.neighbor_triples) == 4  # overall cap wins over the 8 per-entity relations
        # Still the top-4 by confidence.
        assert header.neighbor_triples == [
            "Acme —partner_of→ Partner00",
            "Acme —partner_of→ Partner01",
            "Acme —partner_of→ Partner02",
            "Acme —partner_of→ Partner03",
        ]


def test_confidence_gate_drops_low_confidence_entities_and_edges() -> None:
    """The min_entity/min_edge confidence gates drop weak mentions/entities + weak edges."""
    chunks, mentions, graph = _small_kg()
    # Make Globex's mention weak (below the entity gate) and the located_in edge weak (below the
    # edge gate). Then the Globex entity + the Berlin relation must be gone.
    mentions = [m if m.record_id != "m-globex" else _mention("m-globex", "c2", 0, "Globex", 0.3) for m in mentions]
    prov = ContextHeaderEnricher({"min_entity_confidence": 0.5, "min_edge_confidence": 0.85})
    out = list(prov.enrich(chunks, mentions, graph))

    h2 = out[1].context_header
    # c2 had only Globex (weak, dropped) + Acme; Acme survives so the header is present with Acme only.
    assert h2 is not None
    assert h2.entities == ["Acme"]
    # The located_in (0.80) and partner_of (0.70) edges are below the 0.85 edge gate → dropped;
    # only works_for (0.95) survives corpus-wide.
    h1 = out[0].context_header
    assert h1 is not None
    assert h1.neighbor_triples == ["Alice —works_for→ Acme"]


def test_chunk_with_no_gated_entity_gets_none_header() -> None:
    """A chunk whose only mention is gated out is left context_header=None (nothing to add)."""
    chunks, mentions, graph = _small_kg()
    # Every mention weak; entity gate above them all → no entities anywhere → all headers None.
    weak = [_mention(m.record_id, m.chunk_record_id, m.index, m.text, 0.2) for m in mentions]
    out = list(ContextHeaderEnricher({"min_entity_confidence": 0.5}).enrich(chunks, weak, graph))
    assert all(c.context_header is None for c in out)
    # Byte-identical to the untouched inputs (the additive-field / no-op guarantee, per chunk).
    assert [c.model_dump_json() for c in out] == [c.model_dump_json() for c in chunks]


def test_byte_identical_across_two_seeded_runs() -> None:
    """Determinism: two runs over the same fake KG produce byte-identical enriched chunks."""
    chunks, mentions, graph = _hub_kg(n_neighbors=8)
    a = [c.model_dump_json() for c in ContextHeaderEnricher({}).enrich(chunks, mentions, graph)]
    b = [c.model_dump_json() for c in ContextHeaderEnricher({}).enrich(chunks, mentions, graph)]
    assert a == b


def test_provider_never_reorders_or_drops_the_chunk_stream() -> None:
    """The chunk stream is threaded 1:1 and order-stable — offsets/content untouched."""
    chunks, mentions, graph = _small_kg()
    out = list(ContextHeaderEnricher({}).enrich(chunks, mentions, graph))
    assert [c.record_id for c in out] == [c.record_id for c in chunks]
    for original, enriched in zip(chunks, out, strict=True):
        # content + provenance byte-unchanged — only context_header is added.
        assert enriched.content == original.content
        assert enriched.provenance == original.provenance
        assert enriched.model_copy(update={"context_header": None}) == original
