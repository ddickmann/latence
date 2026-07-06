"""S9 GraphBuilder engine tests: deterministic ids, endpoint validation, Evidence, sorting."""

from __future__ import annotations

from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    Evidence,
    NormalizedRelation,
    Provenance,
)
from latence_core.graph import GraphBuilder, make_graph_id


def _prov(doc: str = "doc-1") -> Provenance:
    return Provenance(
        source_uri=f"file:///c/{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=10,
        document_id=doc,
        char_start=0,
        char_end=4,
        page_start=1,
        page_end=1,
    )


def _cls() -> Classification:
    return Classification(language="en", sensitivity="internal")


def _entity(rid: str, name: str, docs: list[str], mentions: list[str]) -> CanonicalEntity:
    return CanonicalEntity(
        record_id=rid,
        canonical_text=name,
        entity_type="ORG",
        member_mention_ids=mentions,
        source_document_ids=docs,
        evidence=Evidence(mention_ids=mentions, document_ids=docs, confidence=0.9),
        provenance=_prov(docs[0]),
        classification=_cls(),
    )


def _relation(rid: str, head: str, tail: str) -> NormalizedRelation:
    return NormalizedRelation(
        record_id=rid,
        source_relation_id=rid.replace("#normalized", ""),
        label="works_for",
        head_entity_id=head,
        tail_entity_id=tail,
        head_text="Alice",
        tail_text="Acme",
        confidence=0.8,
        evidence=Evidence(mention_ids=["m1", "m2"], document_ids=["doc-1"], confidence=0.8),
        provenance=_prov(),
        classification=_cls(),
    )


def test_make_graph_id_is_deterministic_and_content_addressed() -> None:
    a = make_graph_id("corpus", "node", "entity:acme")
    b = make_graph_id("corpus", "node", "entity:acme")
    c = make_graph_id("corpus", "node", "entity:globex")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_node_id_for_matches_builder_edge_resolution() -> None:
    builder = GraphBuilder(corpus_id="corpus")
    assert builder.node_id_for("entity:acme") == make_graph_id("corpus", "node", "entity:acme")


def test_build_nodes_and_edges_happy_path() -> None:
    e_alice = _entity("entity:alice", "Alice", ["doc-1"], ["m1"])
    e_acme = _entity("entity:acme", "Acme", ["doc-1", "doc-2"], ["m2", "m3"])
    rel = _relation("rel-1#normalized", "entity:alice", "entity:acme")
    builder = GraphBuilder(corpus_id="corpus")
    nodes, edges = builder.build([e_alice, e_acme], [rel])

    assert len(nodes) == 2
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_node_id == builder.node_id_for("entity:alice")
    assert edge.target_node_id == builder.node_id_for("entity:acme")
    assert edge.label == "works_for"
    # Every edge carries Evidence (S9 AC).
    assert edge.evidence_mention_ids == ("m1", "m2")
    # Node evidence links back to its member mentions.
    acme_node = next(n for n in nodes if n.canonical_name == "Acme")
    assert acme_node.evidence_mention_ids == ("m2", "m3")
    assert acme_node.properties["document_count"] == 2
    # v1 has no link prediction: every edge is extracted, not predicted.
    assert edge.is_predicted is False
    assert edge.properties["is_predicted"] is False


def test_edge_dropped_when_endpoint_missing() -> None:
    # A relation whose tail entity has no node must be dropped (no dangling edge).
    e_alice = _entity("entity:alice", "Alice", ["doc-1"], ["m1"])
    rel = _relation("rel-1#normalized", "entity:alice", "entity:ghost")
    builder = GraphBuilder(corpus_id="corpus")
    nodes, edges = builder.build([e_alice], [rel])
    assert len(nodes) == 1
    assert edges == []


def test_edge_dropped_when_endpoints_collapse_to_same_node() -> None:
    # The engine's second edge-drop branch (graph.py: source_node == target_node) is
    # defense-in-depth: a NormalizedRelation whose head and tail resolve to the SAME canonical
    # entity would collapse to a self-loop node. The upstream contract validator
    # (NormalizedRelation._endpoints_distinct) normally makes this state unconstructable, so we
    # bypass validation with model_construct to feed the engine exactly the malformed input the
    # guard defends against, and assert it is DROPPED — never emitted as a GraphEdge with equal
    # endpoints, which the GraphEdge._endpoints_distinct validator would reject, crashing the
    # run mid-assembly. Distinct from the missing-endpoint branch (both endpoints exist here;
    # they are simply identical).
    e_acme = _entity("entity:acme", "Acme", ["doc-1"], ["m1"])
    valid = _relation("rel-self#normalized", "entity:acme", "entity:other")
    self_rel = valid.model_copy(update={"tail_entity_id": "entity:acme"})  # head == tail
    assert self_rel.head_entity_id == self_rel.tail_entity_id  # collapse precondition
    builder = GraphBuilder(corpus_id="corpus")
    assert builder.node_id_for("entity:acme") == builder.node_id_for("entity:acme")
    nodes, edges = builder.build([e_acme], [self_rel])
    assert len(nodes) == 1
    assert edges == []


def test_outputs_are_sorted_by_id_for_determinism() -> None:
    entities = [
        _entity(f"entity:e{i}", f"E{i}", ["doc-1"], [f"m{i}"]) for i in range(5)
    ]
    builder = GraphBuilder(corpus_id="corpus")
    nodes, _edges = builder.build(entities, [])
    ids = [n.node_id for n in nodes]
    assert ids == sorted(ids)


def test_max_evidence_per_node_caps_mention_evidence() -> None:
    big = _entity("entity:big", "Big", ["doc-1"], [f"m{i}" for i in range(10)])
    builder = GraphBuilder(corpus_id="corpus", max_evidence_per_node=3)
    nodes, _edges = builder.build([big], [])
    assert len(nodes[0].evidence_mention_ids) == 3
    # The full membership is still recorded on the node itself.
    assert len(nodes[0].member_mention_ids) == 10


def test_corpus_id_changes_ids() -> None:
    e = _entity("entity:acme", "Acme", ["doc-1"], ["m1"])
    n1, _ = GraphBuilder(corpus_id="a").build([e], [])
    n2, _ = GraphBuilder(corpus_id="b").build([e], [])
    assert n1[0].node_id != n2[0].node_id
