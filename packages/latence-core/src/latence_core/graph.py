"""GraphBuilder — the canonical knowledge-graph assembler (ported PORT-AS-IS).

The pure, dependency-free engine behind the S9 Graph Assembly Stage: it takes the
disambiguated canonical entities and normalized relations and produces the canonical
knowledge graph — nodes (one per entity), edges (one per relation), and per-edge/-node
:class:`~latence_core.contracts.Evidence` back to the source mentions/documents.

Ported as-is from ``dataset-intelligence/graph/builder.py``, re-homed behind the
framework's contracts. The algorithmic value is preserved verbatim:

* **Deterministic, content-addressed IDs** — a node/edge id is a sha256 over a stable key
  (the corpus id + the entity/relation id), truncated to 16 hex chars, so the same entity
  always maps to the same node and two runs over the same corpus produce the same graph (the
  S9 "deterministic IDs" acceptance criterion). This is the ported ``_make_id`` helper.
* **Endpoint validation** — an edge is emitted only when *both* its head and tail entity
  resolved to a node; a relation pointing at a missing endpoint is dropped, never emitted as
  a dangling edge (the ported ``_build_edges`` guard).
* **Provenance links (Evidence)** — every node carries Evidence back to its source mentions,
  and every edge carries Evidence back to the source relation's mentions/documents (the
  ported ``_build_evidence`` step, folded onto the contract's per-record ``evidence`` field).

The private stack's ``§19`` predicted-vs-extracted split is preserved as an
``is_predicted`` edge property: v1 has no link-prediction Stage (that is the post-v1 P3
plugin), so every v1 edge is extracted (``is_predicted=False``); the
attribute is carried so a future predicted edge is never silently mixed with an extracted
one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latence_core.contracts import CanonicalEntity, NormalizedRelation

_ID_LEN = 16


def make_graph_id(corpus_id: str, *parts: str) -> str:
    """Deterministic content-addressed id: sha256 over ``corpus_id | parts`` (ported ``_make_id``).

    Truncated to 16 hex chars — collision-safe at corpus scale and short enough to read in a
    GraphML/TTL export. The same inputs always yield the same id (byte-stable), so the graph
    is reproducible across runs (S9 determinism).
    """
    raw = "|".join([corpus_id, *parts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_ID_LEN]


@dataclass(frozen=True)
class BuiltNode:
    """One assembled graph node (engine-level; the Provider maps it to a GraphNode contract)."""

    node_id: str
    entity_id: str
    label: str
    canonical_name: str
    member_mention_ids: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    kb_id: str | None
    confidence: float
    evidence_mention_ids: tuple[str, ...]
    evidence_document_ids: tuple[str, ...]
    evidence_confidence: float
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltEdge:
    """One assembled graph edge (engine-level; the Provider maps it to a GraphEdge contract)."""

    edge_id: str
    relation_id: str
    source_node_id: str
    target_node_id: str
    label: str
    confidence: float
    inverted: bool
    is_predicted: bool
    evidence_mention_ids: tuple[str, ...]
    evidence_document_ids: tuple[str, ...]
    evidence_confidence: float
    properties: dict[str, object] = field(default_factory=dict)


class GraphBuilder:
    """Builds the canonical knowledge graph from resolved entities + normalized relations.

    Produces one :class:`BuiltNode` per canonical entity and one :class:`BuiltEdge` per
    normalized relation whose endpoints both resolved to a node, plus the Evidence links. All
    ids are deterministic (``make_graph_id``) and every output collection is sorted, so the
    assembled graph is byte-identical across runs over the same corpus.

    ``max_evidence_per_node`` caps the mention ids folded into a node's Evidence (0 =
    unlimited) — the ported ``max_evidence_per_entity`` F8 fix, so a hub entity with tens of
    thousands of mentions does not produce an unbounded Evidence blob.
    """

    def __init__(self, corpus_id: str, max_evidence_per_node: int = 0) -> None:
        self._corpus_id = corpus_id
        self._max_evidence_per_node = max_evidence_per_node

    def node_id_for(self, entity_id: str) -> str:
        """The deterministic node id an entity maps to (shared by edge endpoint resolution)."""
        return make_graph_id(self._corpus_id, "node", entity_id)

    def build(
        self,
        entities: list[CanonicalEntity],
        relations: list[NormalizedRelation],
    ) -> tuple[list[BuiltNode], list[BuiltEdge]]:
        """Assemble the graph: return ``(nodes, edges)``, each sorted by its deterministic id."""
        nodes = self._build_nodes(entities)
        node_ids = {n.node_id for n in nodes}
        edges = self._build_edges(relations, node_ids)
        nodes.sort(key=lambda n: n.node_id)
        edges.sort(key=lambda e: e.edge_id)
        return nodes, edges

    def _build_nodes(self, entities: list[CanonicalEntity]) -> list[BuiltNode]:
        """One node per canonical entity, with Evidence back to its source mentions."""
        nodes: list[BuiltNode] = []
        for entity in entities:
            node_id = self.node_id_for(entity.record_id)
            mention_ids = list(entity.member_mention_ids)
            if self._max_evidence_per_node > 0:
                mention_ids = mention_ids[: self._max_evidence_per_node]
            nodes.append(
                BuiltNode(
                    node_id=node_id,
                    entity_id=entity.record_id,
                    label=entity.entity_type,
                    canonical_name=entity.canonical_text,
                    member_mention_ids=tuple(entity.member_mention_ids),
                    source_document_ids=tuple(entity.source_document_ids),
                    kb_id=entity.kb_id,
                    confidence=entity.evidence.confidence,
                    evidence_mention_ids=tuple(mention_ids),
                    evidence_document_ids=tuple(entity.source_document_ids),
                    evidence_confidence=entity.evidence.confidence,
                    properties={
                        "mention_count": entity.mention_count,
                        "document_count": len(entity.source_document_ids),
                        "confidence": entity.evidence.confidence,
                        "linked": entity.linked,
                    },
                )
            )
        return nodes

    def _build_edges(
        self, relations: list[NormalizedRelation], node_ids: set[str]
    ) -> list[BuiltEdge]:
        """One edge per relation whose head AND tail resolved to a node (endpoint validation).

        The relation's endpoints are canonical *entity* record_ids; each maps deterministically
        to its node id. An edge is emitted only when both node ids exist and are distinct — a
        relation pointing at a missing or collapsed endpoint is dropped (never a dangling or
        self-loop edge), the ported ``_build_edges`` guard.
        """
        edges: list[BuiltEdge] = []
        for relation in relations:
            source_node = self.node_id_for(relation.head_entity_id)
            target_node = self.node_id_for(relation.tail_entity_id)
            if source_node not in node_ids or target_node not in node_ids:
                continue
            if source_node == target_node:
                continue
            edge_id = make_graph_id(self._corpus_id, "edge", relation.record_id)
            edges.append(
                BuiltEdge(
                    edge_id=edge_id,
                    relation_id=relation.record_id,
                    source_node_id=source_node,
                    target_node_id=target_node,
                    label=relation.label,
                    confidence=relation.confidence,
                    inverted=relation.inverted,
                    is_predicted=False,  # v1 has no link-prediction Stage; every edge is extracted.
                    evidence_mention_ids=tuple(relation.evidence.mention_ids),
                    evidence_document_ids=tuple(relation.evidence.document_ids),
                    evidence_confidence=relation.evidence.confidence,
                    properties={
                        "inverted": relation.inverted,
                        "is_predicted": False,
                        "confidence": relation.confidence,
                    },
                )
            )
        return edges
