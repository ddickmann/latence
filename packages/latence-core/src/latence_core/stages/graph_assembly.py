"""Built-in reference GraphAssembler Provider (deterministic, CPU-viable, zero-dep).

The S9 Graph Assembly reference Provider (ADR-0004/0017; ships in core like every other
reference Provider so the demo runs end to end, ADR-0007 CPU-first, ADR-0016 no extra deps).
It wraps the ported graph builder (:mod:`latence_core.graph` — the deterministic,
content-addressed ``_make_id`` id scheme, endpoint validation, and Evidence links) and turns
the run's :class:`~latence_core.contracts.DisambiguationRecord`s (canonical entities +
normalized relations) into typed :class:`~latence_core.contracts.GraphRecord`s:

* one ``NODE``-scope record per :class:`~latence_core.contracts.CanonicalEntity` — a graph
  node with a deterministic id, its member mentions + source-document reach, an optional
  external-KB link, page-accurate Provenance inherited from the source entity, and
  :class:`~latence_core.contracts.Evidence` back to the source mentions;
* one ``EDGE``-scope record per :class:`~latence_core.contracts.NormalizedRelation` — a graph
  edge with a deterministic id, its head/tail node ids (the deterministic ids of the entity
  nodes), the normalized label, its confidence, and **per-edge Evidence** (every edge carries
  Evidence — the S9 acceptance criterion). An edge is emitted only when both endpoints
  resolved to a node.

Determinism: nodes are emitted in node-id order, then edges in edge-id order; every member/
document list is engine-sorted — so a seeded run is byte-identical and two runs over the same
corpus produce the same graph (S9 "deterministic IDs" + Baseline bar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
)
from latence_core.graph import BuiltEdge, BuiltNode, GraphBuilder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import CanonicalEntity, NormalizedRelation


class CanonicalGraphAssembler:
    """GraphAssembler Provider: DisambiguationRecords -> graph nodes + edges with Evidence.

    Config:
        corpus_id: str  — the stable corpus identity mixed into every deterministic node/edge
            id (so a run and a re-run over the same corpus share ids). Default ``"corpus"``.
        max_evidence_per_node: int  — cap on mention ids folded into a node's Evidence
            (0 = unlimited; the ported F8 bound for hub entities). Default 0.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._max_evidence_per_node = int(config.get("max_evidence_per_node", 0))

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        record_list = list(records)
        # Partition the scoped Disambiguation carrier back into its two payload streams, and
        # keep the source records so a node/edge inherits its source's Provenance/Classification.
        entities: list[CanonicalEntity] = []
        entity_src: dict[str, DisambiguationRecord] = {}
        relations: list[NormalizedRelation] = []
        relation_src: dict[str, DisambiguationRecord] = {}
        for record in record_list:
            if record.scope is DisambiguationScope.ENTITY and record.entity is not None:
                entities.append(record.entity)
                entity_src[record.entity.record_id] = record
            elif record.scope is DisambiguationScope.RELATION and record.relation is not None:
                relations.append(record.relation)
                relation_src[record.relation.record_id] = record

        builder = GraphBuilder(
            corpus_id=self._corpus_id, max_evidence_per_node=self._max_evidence_per_node
        )
        nodes, edges = builder.build(entities, relations)

        for node in nodes:
            yield self._node_record(node, entity_src[node.entity_id])
        for edge in edges:
            yield self._edge_record(edge, relation_src[edge.relation_id])

    @staticmethod
    def _node_record(node: BuiltNode, source: DisambiguationRecord) -> GraphRecord:
        evidence = Evidence(
            mention_ids=list(node.evidence_mention_ids),
            document_ids=list(node.evidence_document_ids),
            snippet=f"node {node.canonical_name!r} ({node.label}) "
            f"from {len(node.member_mention_ids)} mention(s)",
            confidence=node.evidence_confidence,
        )
        graph_node = GraphNode(
            record_id=f"node:{node.node_id}",
            node_id=node.node_id,
            entity_id=node.entity_id,
            label=node.label,
            canonical_name=node.canonical_name,
            member_mention_ids=list(node.member_mention_ids),
            source_document_ids=list(node.source_document_ids),
            kb_id=node.kb_id,
            confidence=node.confidence,
            properties=dict(node.properties),
            evidence=evidence,
            provenance=source.provenance,
            classification=source.classification,
        )
        return GraphRecord(
            record_id=f"node:{node.node_id}#graph",
            scope=GraphScope.NODE,
            node=graph_node,
            provenance=source.provenance,
            classification=source.classification,
        )

    @staticmethod
    def _edge_record(edge: BuiltEdge, source: DisambiguationRecord) -> GraphRecord:
        evidence = Evidence(
            mention_ids=list(edge.evidence_mention_ids),
            document_ids=list(edge.evidence_document_ids),
            snippet=f"edge {edge.label!r} "
            f"{edge.source_node_id} -> {edge.target_node_id}",
            confidence=edge.evidence_confidence,
        )
        graph_edge = GraphEdge(
            record_id=f"edge:{edge.edge_id}",
            edge_id=edge.edge_id,
            relation_id=edge.relation_id,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            label=edge.label,
            confidence=edge.confidence,
            properties=dict(edge.properties),
            evidence=evidence,
            provenance=source.provenance,
            classification=source.classification,
        )
        return GraphRecord(
            record_id=f"edge:{edge.edge_id}#graph",
            scope=GraphScope.EDGE,
            edge=graph_edge,
            provenance=source.provenance,
            classification=source.classification,
        )
