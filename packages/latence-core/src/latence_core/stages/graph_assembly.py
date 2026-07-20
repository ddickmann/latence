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

from latence_core.capability import (
    GraphFeatureComputer,
    ProviderRegistry,
    satisfies_capability,
)
from latence_core.contracts import (
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
)
from latence_core.errors import ConfigError, ContractError
from latence_core.graph import BuiltEdge, BuiltNode, GraphBuilder, make_graph_id
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import CanonicalEntity, NormalizedRelation


def _resolve_feature_computer(
    spec: Any, registry: ProviderRegistry
) -> GraphFeatureComputer | None:
    """Resolve the OPTIONAL ``graph_features`` computer from a ``{"provider", "config"}`` block.

    The same composition Export uses for its Embedder (ADR-0017): ``spec`` is ``None`` (enrichment
    off — the graph is emitted byte-identically to today, ADR-0051) or a ``{"provider", "config"}``
    map naming a :class:`~latence_core.capability.GraphFeatureComputer` Provider (the in-core
    ``graph_features.degree`` / ``graph_features.pagerank`` references, or a pluggable
    Louvain/Leiden package). A mis-named Provider fails **loudly at wiring time** — a
    :class:`~latence_core.errors.ContractError` — rather than silently emitting no features; a
    malformed ``spec`` shape is a :class:`~latence_core.errors.ConfigError`.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        msg = (
            "Graph Assembly config 'graph_features' must be a mapping "
            "{'provider': <name>, 'config': {...}} or absent; "
            f"got {type(spec).__name__}."
        )
        raise ConfigError(msg)
    provider_name = spec.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        msg = (
            "Graph Assembly config 'graph_features' requires a non-empty string 'provider' "
            "entry-point name."
        )
        raise ConfigError(msg)
    provider_config = spec.get("config", {})
    if not isinstance(provider_config, dict):
        msg = "Graph Assembly config 'graph_features.config' must be a mapping if present."
        raise ConfigError(msg)
    provider_cls = registry.load(provider_name)
    provider = provider_cls(provider_config)
    # satisfies_capability, not isinstance: a runtime_checkable isinstance INVOKES a Protocol
    # property via hasattr, which would run Provider work at wiring time (audit R6).
    if not satisfies_capability(provider, GraphFeatureComputer):
        msg = (
            f"Provider {provider_name!r} named in Graph Assembly config 'graph_features' does not "
            "satisfy the GraphFeatureComputer capability."
        )
        raise ContractError(msg)
    return provider


def _stamp_features(
    node_records: list[GraphRecord],
    edge_records: list[GraphRecord],
    computer: GraphFeatureComputer | None,
) -> list[GraphRecord]:
    """Stamp centrality + community onto each NODE record's ``properties`` (ADR-0051), additively.

    When no computer is wired the node records are returned UNCHANGED (the graph is byte-identical
    to today — the genuinely-optional guarantee). Otherwise the computer sees the whole assembled
    graph (all nodes + all edges) and returns a :class:`~latence_core.capability.GraphFeatures`
    covering every node; each node gets ``properties['centrality']`` (float) +
    ``properties['community']`` (str) added via ``model_copy`` (offsets / Evidence / Provenance
    untouched). A node the computer failed to score is a loud
    :class:`~latence_core.errors.ContractError` (the GraphFeatures contract already forbids a
    partial map — this is belt-and-braces), never a silently-unenriched node.
    """
    if computer is None:
        return node_records
    nodes = [r.node for r in node_records if r.node is not None]
    edges = [r.edge for r in edge_records if r.edge is not None]
    features = computer.compute_features(nodes, edges)
    stamped: list[GraphRecord] = []
    for record in node_records:
        node = record.node
        if node is None:  # pragma: no cover - node_records are all NODE-scope by construction
            stamped.append(record)
            continue
        if node.node_id not in features.centrality or node.node_id not in features.community:
            msg = (
                f"GraphFeatureComputer returned no features for node {node.node_id!r}; the "
                "GraphFeatures contract covers every node."
            )
            raise ContractError(msg)
        new_properties = {
            **node.properties,
            "centrality": features.centrality[node.node_id],
            "community": features.community[node.node_id],
        }
        new_node = node.model_copy(update={"properties": new_properties})
        stamped.append(record.model_copy(update={"node": new_node}))
    return stamped


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
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._max_evidence_per_node = int(config.get("max_evidence_per_node", 0))
        # ADR-0051: the OPTIONAL graph-feature enrichment. Absent ⇒ no computer ⇒ the graph is
        # emitted byte-identically to today. Resolved from config like Export resolves its Embedder.
        registry = registry or ProviderRegistry()
        self._feature_computer = _resolve_feature_computer(
            config.get("graph_features"), registry
        )

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        entities, entity_src, relations, relation_src = _partition(records)
        builder = GraphBuilder(
            corpus_id=self._corpus_id, max_evidence_per_node=self._max_evidence_per_node
        )
        nodes, edges = builder.build(entities, relations)

        node_records = [self._node_record(node, entity_src[node.entity_id]) for node in nodes]
        edge_records = [self._edge_record(edge, relation_src[edge.relation_id]) for edge in edges]
        # ADR-0051: stamp centrality + community onto each node (a no-op when unconfigured, so the
        # emitted stream — nodes then edges, in id order — is byte-identical to today).
        node_records = _stamp_features(node_records, edge_records, self._feature_computer)
        yield from node_records
        yield from edge_records

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
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._max_evidence_per_node = int(config.get("max_evidence_per_node", 0))
        # ADR-0051: the OPTIONAL graph-feature enrichment (parity with the reference assembler).
        registry = registry or ProviderRegistry()
        self._feature_computer = _resolve_feature_computer(
            config.get("graph_features"), registry
        )

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        entities, entity_src, relations, relation_src = _partition(records)
        builder = GraphBuilder(
            corpus_id=self._corpus_id, max_evidence_per_node=self._max_evidence_per_node
        )
        nodes, edges = builder.build(entities, relations)

        # Nodes are assembled identically to the reference (node parity).
        node_records = [
            CanonicalGraphAssembler._node_record(node, entity_src[node.entity_id])
            for node in nodes
        ]
        # Fold parallel edges (same source, target, label) into one weighted edge.
        edge_records = list(self._weighted_edge_records(edges, relation_src))
        # ADR-0051: stamp centrality + community over the WEIGHTED graph (its folded edges), so the
        # features reflect this assembler's edge model. A no-op when unconfigured (byte-identical).
        node_records = _stamp_features(node_records, edge_records, self._feature_computer)
        yield from node_records
        yield from edge_records

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
