"""Second reference GraphAssembler Provider (#77): WeightedGraphAssembler parity + benchmark.

The 'providers' hardening bar requires ≥2 reference providers per Capability with parity
tests (same typed contracts) + a benchmark vs the existing one.
:class:`WeightedGraphAssembler` is the second CPU Graph Assembler behind the same seam as
:class:`CanonicalGraphAssembler`.

* **Parity**: both satisfy the ``GraphAssembler`` protocol and emit the same typed contract
  — NODE + EDGE :class:`GraphRecord`s with deterministic ids, per-edge Evidence, inherited
  Provenance/Classification, endpoint-validated (no self-loop). Node assembly is identical.
* **Benchmark / difference**: parallel same-``(head, tail, label)`` relations fold into ONE
  weighted edge (weight = multiplicity, confidence = max, Evidence = union) rather than the
  reference's one-edge-per-relation multigraph. The fold assertions are the reverting
  regression guard.
"""

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
from latence_core.stages.graph_assembly import (
    CanonicalGraphAssembler,
    WeightedGraphAssembler,
)


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


def _entity_record(rid: str, name: str, mentions: list[str]) -> DisambiguationRecord:
    entity = CanonicalEntity(
        record_id=rid,
        canonical_text=name,
        entity_type="ORG",
        member_mention_ids=mentions,
        source_document_ids=["doc-1", "doc-2"],
        kb_id=None,
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


def _relation_record(
    rid: str, head: str, tail: str, label: str, confidence: float, doc: str
) -> DisambiguationRecord:
    relation = NormalizedRelation(
        record_id=rid,
        source_relation_id=rid,
        label=label,
        head_entity_id=head,
        tail_entity_id=tail,
        head_text="Acme",
        tail_text="Beta",
        confidence=confidence,
        evidence=Evidence(mention_ids=[f"{rid}-m"], document_ids=[doc], confidence=confidence),
        provenance=_prov(doc),
        classification=_cls(),
    )
    return DisambiguationRecord(
        record_id=f"{rid}#record",
        scope=DisambiguationScope.RELATION,
        relation=relation,
        provenance=_prov(doc),
        classification=_cls(),
    )


def _corpus_with_parallel_edges() -> list[DisambiguationRecord]:
    """Two documents both assert Acme —acquired→ Beta (a parallel-edge multigraph)."""
    return [
        _entity_record("entity:acme", "Acme", ["ma1"]),
        _entity_record("entity:beta", "Beta", ["mb1"]),
        _relation_record("r1", "entity:acme", "entity:beta", "acquired", 0.80, "doc-1"),
        _relation_record("r2", "entity:acme", "entity:beta", "acquired", 0.95, "doc-2"),
    ]


def test_second_graph_assembler_satisfies_the_same_seam() -> None:
    assert isinstance(WeightedGraphAssembler(), GraphAssembler)


def test_second_graph_assembler_node_parity_with_reference() -> None:
    """Parity: node assembly is identical to the reference (same ported builder)."""
    records = _corpus_with_parallel_edges()
    weighted_nodes = {
        r.node.node_id: r.node.canonical_name
        for r in WeightedGraphAssembler().assemble(records)
        if r.scope is GraphScope.NODE and r.node is not None
    }
    ref_nodes = {
        r.node.node_id: r.node.canonical_name
        for r in CanonicalGraphAssembler().assemble(records)
        if r.scope is GraphScope.NODE and r.node is not None
    }
    assert weighted_nodes == ref_nodes and weighted_nodes


def test_benchmark_folds_parallel_edges_into_one_weighted_edge() -> None:
    """Benchmark: the reference emits a multigraph; the weighted assembler folds it."""
    records = _corpus_with_parallel_edges()
    ref_edges = [
        r.edge for r in CanonicalGraphAssembler().assemble(records)
        if r.scope is GraphScope.EDGE and r.edge is not None
    ]
    weighted_edges = [
        r.edge for r in WeightedGraphAssembler().assemble(records)
        if r.scope is GraphScope.EDGE and r.edge is not None
    ]
    # Reference: one edge per relation (two parallel edges).
    assert len(ref_edges) == 2
    # Weighted: the two parallel relations fold into exactly one edge.
    assert len(weighted_edges) == 1
    edge = weighted_edges[0]
    assert edge.properties["weight"] == 2  # multiplicity recorded
    assert edge.properties["folded_relation_ids"] == ["r1", "r2"]
    # Strongest assertion wins.
    assert edge.confidence == 0.95
    # Per-edge Evidence is the UNION over both folded relations (S9: every edge has Evidence).
    assert edge.evidence.mention_ids == ["r1-m", "r2-m"]
    assert edge.evidence.document_ids == ["doc-1", "doc-2"]


def test_second_graph_assembler_deterministic_edge_id() -> None:
    """The weighted edge id is content-addressed over endpoints+label, stable across runs."""
    records = _corpus_with_parallel_edges()
    first = [
        r.edge.edge_id for r in WeightedGraphAssembler().assemble(records)
        if r.scope is GraphScope.EDGE and r.edge is not None
    ]
    second = [
        r.edge.edge_id for r in WeightedGraphAssembler().assemble(records)
        if r.scope is GraphScope.EDGE and r.edge is not None
    ]
    assert first == second and first
    # Distinct labels are NOT folded together (weighting is per (head, tail, label)).
    records.append(
        _relation_record("r3", "entity:acme", "entity:beta", "partnered_with", 0.7, "doc-1")
    )
    labels = sorted(
        r.edge.label for r in WeightedGraphAssembler().assemble(records)
        if r.scope is GraphScope.EDGE and r.edge is not None
    )
    assert labels == ["acquired", "partnered_with"]


def test_second_graph_assembler_drops_dangling_edges() -> None:
    """Contract parity: an edge to a missing node (endpoint that resolved to no node) is dropped.

    (A self-loop relation cannot even be constructed — the NormalizedRelation contract forbids
    head == tail — so endpoint validation for self-loops is enforced upstream at the contract;
    the assembler's remaining job is dropping edges whose endpoint has no node.)
    """
    records = [
        _entity_record("entity:acme", "Acme", ["ma1"]),
        # tail entity:ghost has no node → dangling; must be dropped.
        _relation_record("r1", "entity:acme", "entity:ghost", "acquired", 0.9, "doc-1"),
    ]
    edges = [
        r for r in WeightedGraphAssembler().assemble(records) if r.scope is GraphScope.EDGE
    ]
    assert edges == []
