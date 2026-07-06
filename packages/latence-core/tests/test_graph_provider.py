"""S9 CanonicalGraphAssembler Provider tests: DisambiguationRecords -> GraphRecords."""

from __future__ import annotations

from latence_core.capability import GraphAssembler
from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    GraphScope,
    NormalizedRelation,
    Provenance,
)
from latence_core.stages.graph_assembly import CanonicalGraphAssembler


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


def _entity_record(rid: str, name: str, mentions: list[str], kb: str | None = None) -> DisambiguationRecord:
    entity = CanonicalEntity(
        record_id=rid,
        canonical_text=name,
        entity_type="ORG",
        member_mention_ids=mentions,
        source_document_ids=["doc-1"],
        kb_id=kb,
        evidence=Evidence(mention_ids=mentions, document_ids=["doc-1"], confidence=0.9),
        provenance=_prov(),
        classification=_cls(),
    )
    return DisambiguationRecord(
        record_id=f"{rid}#canonical",
        scope=DisambiguationScope.ENTITY,
        entity=entity,
        provenance=_prov(),
        classification=_cls(),
    )


def _relation_record(rid: str, head: str, tail: str) -> DisambiguationRecord:
    relation = NormalizedRelation(
        record_id=rid,
        source_relation_id="rel-1",
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
    return DisambiguationRecord(
        record_id=f"{rid}#record",
        scope=DisambiguationScope.RELATION,
        relation=relation,
        provenance=_prov(),
        classification=_cls(),
    )


def test_provider_satisfies_capability() -> None:
    assert isinstance(CanonicalGraphAssembler(), GraphAssembler)


def test_assemble_emits_nodes_then_edges() -> None:
    records = [
        _entity_record("entity:alice", "Alice", ["m1"]),
        _entity_record("entity:acme", "Acme", ["m2"], kb="Q1"),
        _relation_record("rel-1#normalized", "entity:alice", "entity:acme"),
    ]
    out = list(CanonicalGraphAssembler().assemble(records))
    nodes = [r for r in out if r.scope is GraphScope.NODE]
    edges = [r for r in out if r.scope is GraphScope.EDGE]
    assert len(nodes) == 2
    assert len(edges) == 1
    # Nodes come before edges in the emitted stream (deterministic ordering).
    scopes = [r.scope for r in out]
    assert scopes.index(GraphScope.EDGE) > max(
        i for i, s in enumerate(scopes) if s is GraphScope.NODE
    )
    # KB link carried onto the node.
    acme = next(r.node for r in nodes if r.node and r.node.canonical_name == "Acme")
    assert acme.kb_id == "Q1"
    assert acme.linked is True
    # Edge endpoints point at the deterministic node ids.
    edge = edges[0].edge
    assert edge is not None
    node_ids = {r.node.node_id for r in nodes if r.node}
    assert edge.source_node_id in node_ids
    assert edge.target_node_id in node_ids
    assert edge.evidence.mention_ids  # every edge carries Evidence


def test_node_inherits_source_provenance_and_classification() -> None:
    rec = _entity_record("entity:alice", "Alice", ["m1"])
    out = list(CanonicalGraphAssembler().assemble([rec]))
    node = out[0].node
    assert node is not None
    assert node.provenance.document_id == "doc-1"
    assert node.classification.sensitivity == "internal"


def test_relation_with_unresolved_endpoint_is_dropped() -> None:
    # A relation pointing at an entity that has no node yields no edge.
    records = [
        _entity_record("entity:alice", "Alice", ["m1"]),
        _relation_record("rel-1#normalized", "entity:alice", "entity:missing"),
    ]
    out = list(CanonicalGraphAssembler().assemble(records))
    assert [r for r in out if r.scope is GraphScope.EDGE] == []


def test_assemble_is_deterministic() -> None:
    records = [
        _entity_record("entity:acme", "Acme", ["m2"]),
        _entity_record("entity:alice", "Alice", ["m1"]),
        _relation_record("rel-1#normalized", "entity:alice", "entity:acme"),
    ]
    a = [r.model_dump_json() for r in CanonicalGraphAssembler().assemble(records)]
    b = [r.model_dump_json() for r in CanonicalGraphAssembler().assemble(records)]
    assert a == b


def test_empty_input_yields_empty_graph() -> None:
    assert list(CanonicalGraphAssembler().assemble([])) == []
