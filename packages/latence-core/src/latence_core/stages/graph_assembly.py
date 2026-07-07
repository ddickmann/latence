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
from latence_core.graph import BuiltEdge, BuiltNode, GraphBuilder, make_graph_id
from latence_core.providers.profile import ProviderProfile

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

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The canonical Graph Assembler builds content-addressed nodes/edges + Evidence.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the
        package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
        cost, so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._max_evidence_per_node = int(config.get("max_evidence_per_node", 0))

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        entities, entity_src, relations, relation_src = _partition(records)
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


def _partition(
    records: Iterable[DisambiguationRecord],
) -> tuple[
    list[CanonicalEntity],
    dict[str, DisambiguationRecord],
    list[NormalizedRelation],
    dict[str, DisambiguationRecord],
]:
    """Split the scoped Disambiguation carrier back into its entity + relation streams.

    Returns the entities and relations plus, for each, a ``record_id -> source record`` map
    so an assembled node/edge inherits its source's Provenance + Classification. Shared by
    both reference Graph Assembler Providers.
    """
    entities: list[CanonicalEntity] = []
    entity_src: dict[str, DisambiguationRecord] = {}
    relations: list[NormalizedRelation] = []
    relation_src: dict[str, DisambiguationRecord] = {}
    for record in records:
        if record.scope is DisambiguationScope.ENTITY and record.entity is not None:
            entities.append(record.entity)
            entity_src[record.entity.record_id] = record
        elif record.scope is DisambiguationScope.RELATION and record.relation is not None:
            relations.append(record.relation)
            relation_src[record.relation.record_id] = record
    return entities, entity_src, relations, relation_src


class WeightedGraphAssembler:
    """Second reference GraphAssembler Provider (#77): parallel edges folded into weighted edges.

    An alternative CPU graph-assembly strategy behind the same
    :class:`~latence_core.capability.GraphAssembler` seam as :class:`CanonicalGraphAssembler`,
    satisfying the identical typed contract — one ``NODE`` record per canonical entity and one
    ``EDGE`` record per surviving edge, every record carrying a **deterministic** id, its
    :class:`~latence_core.contracts.Evidence`, and inherited Provenance + Classification
    (endpoint-validated, no dangling/self-loop edge). Pure Python, deterministic, zero extra
    deps (ADR-0007 CPU-first, ADR-0016 thin core, ADR-0017 files-not-a-live-DB); no model, so
    no license note (ADR-0012).

    The difference vs the reference is the graph model: where
    :class:`CanonicalGraphAssembler` emits **one edge per relation** (a multigraph — two
    documents both asserting ``Acme —acquired→ Beta`` produce two parallel edges), this folds
    every parallel same-``(head, tail, label)`` relation into **one weighted edge** (a simple
    weighted graph) whose:

    * ``edge_id`` is deterministic and content-addressed over the *endpoint pair + label* (so a
      re-run over the same corpus yields the same edge id, S9 determinism), NOT over the source
      relation id (which differs per parallel relation);
    * ``confidence`` is the maximum over the folded relations (the strongest assertion wins);
    * per-edge :class:`~latence_core.contracts.Evidence` is the **union** of the folded
      relations' mention + document ids (deduplicated, sorted) — so the merged edge still cites
      every source that justifies it (per-edge Evidence, the S9 acceptance criterion);
    * ``properties["weight"]`` records how many source relations were folded (the multiplicity),
      the feature a downstream ranker/KG consumer uses that the reference multigraph does not
      expose directly.

    Nodes are assembled by the same ported :class:`GraphBuilder` as the reference (node parity).
    Deterministic: nodes in node-id order, then edges in edge-id order; every Evidence list is
    sorted. Benchmarked against the reference in ``test_graph_second_provider``.

    Config:
        corpus_id: str  — the stable corpus identity mixed into every deterministic id.
            Default ``"corpus"``.
        max_evidence_per_node: int  — cap on mention ids folded into a node's Evidence
            (0 = unlimited). Default 0.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The weighted Graph Assembler folds parallel same-typed relations into weighted edges.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the
        package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
        cost, so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._max_evidence_per_node = int(config.get("max_evidence_per_node", 0))

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        entities, entity_src, relations, relation_src = _partition(records)
        builder = GraphBuilder(
            corpus_id=self._corpus_id, max_evidence_per_node=self._max_evidence_per_node
        )
        nodes, edges = builder.build(entities, relations)

        # Nodes are assembled identically to the reference (node parity).
        for node in nodes:
            yield CanonicalGraphAssembler._node_record(node, entity_src[node.entity_id])

        # Fold parallel edges (same source, target, label) into one weighted edge.
        yield from self._weighted_edge_records(edges, relation_src)

    def _weighted_edge_records(
        self, edges: list[BuiltEdge], relation_src: dict[str, DisambiguationRecord]
    ) -> Iterator[GraphRecord]:
        groups: dict[tuple[str, str, str], list[BuiltEdge]] = {}
        for edge in edges:
            key = (edge.source_node_id, edge.target_node_id, edge.label)
            groups.setdefault(key, []).append(edge)

        records: list[GraphRecord] = []
        for (source_node, target_node, label), members in groups.items():
            edge_id = make_graph_id(self._corpus_id, "wedge", source_node, target_node, label)
            # Union the Evidence over every folded relation (dedup + sort for determinism).
            mention_ids = sorted(
                {mid for m in members for mid in m.evidence_mention_ids}
            )
            document_ids = sorted(
                {did for m in members for did in m.evidence_document_ids}
            )
            # The strongest assertion wins; Evidence confidence tracks it too.
            confidence = max(m.confidence for m in members)
            evidence_confidence = max(m.evidence_confidence for m in members)
            # A representative source relation supplies the inherited Provenance/Classification
            # (deterministic: the lexicographically-first source relation id in the group).
            representative = min(members, key=lambda m: m.relation_id)
            source = relation_src[representative.relation_id]
            evidence = Evidence(
                mention_ids=mention_ids,
                document_ids=document_ids,
                snippet=f"weighted edge {label!r} {source_node} -> {target_node} "
                f"(folded {len(members)} relation(s))",
                confidence=evidence_confidence,
            )
            graph_edge = GraphEdge(
                record_id=f"edge:{edge_id}",
                edge_id=edge_id,
                relation_id=representative.relation_id,
                source_node_id=source_node,
                target_node_id=target_node,
                label=label,
                confidence=confidence,
                properties={
                    "weight": len(members),
                    "folded_relation_ids": sorted(m.relation_id for m in members),
                    "confidence": confidence,
                },
                evidence=evidence,
                provenance=source.provenance,
                classification=source.classification,
            )
            records.append(
                GraphRecord(
                    record_id=f"edge:{edge_id}#graph",
                    scope=GraphScope.EDGE,
                    edge=graph_edge,
                    provenance=source.provenance,
                    classification=source.classification,
                )
            )
        # Edge-id order for determinism (matches the reference's edge ordering discipline).
        records.sort(key=lambda r: r.edge.edge_id if r.edge is not None else "")
        yield from records
