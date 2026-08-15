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

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from latence_core.capability import (
    GraphFeatureComputer,
    ProviderRegistry,
)
from latence_core.contracts import (
    ChunkRecord,
    DisambiguationRecord,
    DisambiguationScope,
    EntityMention,
    Evidence,
    GraphEdge,
    GraphHyperedge,
    GraphNode,
    GraphRecord,
    GraphScope,
    RelationMention,
)
from latence_core.errors import ConfigError, ContractError
from latence_core.graph import BuiltEdge, BuiltNode, GraphBuilder, make_graph_id
from latence_core.hyperedge import HyperedgeBuilder, HyperedgeConfig
from latence_core.pipeline import CapabilityKind
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.subprovider import owner_label, resolve_sub_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from latence_core.contracts import CanonicalEntity, NormalizedRelation

# The message prefix this Stage's nested-block errors carry, from the one slot declaration.
_OWNER = owner_label(CapabilityKind.GRAPH_ASSEMBLY)


def _resolve_feature_computer(
    spec: Any, registry: ProviderRegistry
) -> GraphFeatureComputer | None:
    """Resolve the OPTIONAL ``graph_features`` computer from a ``{"provider", "config"}`` block.

    The same nested sub-Provider block the Export uses for its Embedder (ADR-0017), resolved
    through the one module that owns that block (:mod:`latence_core.subprovider`): ``spec`` is
    ``None`` (enrichment off — the graph is emitted byte-identically to today, ADR-0051) or a
    ``{"provider", "config"}`` map naming a
    :class:`~latence_core.capability.GraphFeatureComputer` Provider (the in-core
    ``graph_features.degree`` / ``graph_features.pagerank`` references, or a pluggable
    Louvain/Leiden package).

    Failure modes are unchanged — a mis-named Provider is a loud wiring-time
    :class:`~latence_core.errors.ContractError`, a malformed shape a
    :class:`~latence_core.errors.ConfigError` — but the *declaration* that ``GRAPH_ASSEMBLY``
    owns a ``graph_features`` slot demanding ``GraphFeatureComputer`` now lives in one place, so
    the offline ``latence stack check`` verifies this block instead of it reaching run time
    unexamined (the blessed SOTA stack wires it, and nothing static used to look at it).
    """
    return resolve_sub_provider(
        spec,
        capability=CapabilityKind.GRAPH_ASSEMBLY,
        key="graph_features",
        registry=registry,
        owner=_OWNER,
    )


def _resolve_hyperedge_config(spec: Any) -> HyperedgeConfig | None:
    """Resolve the OPTIONAL ``hyperedges`` block (ADR-0057): None = projection off.

    ``spec`` is ``None`` (off — the emitted stream is byte-identical to today and the Runner
    never streams document-level records to this Stage), ``{}``/``True`` (on, spec-default
    policy), or a mapping overriding :class:`~latence_core.hyperedge.HyperedgeConfig` fields
    (``weak_hyperedges``, ``max_arity``, ``min_tokens``, ``max_tokens``,
    ``max_window_sentences``). A malformed shape is a loud wiring-time ConfigError, never a
    silent default.
    """
    if spec is None:
        return None
    if spec is True:
        return HyperedgeConfig()
    if not isinstance(spec, dict):
        msg = (
            "graph_assembly config 'hyperedges' must be true or a mapping of "
            f"HyperedgeConfig overrides, got {type(spec).__name__}."
        )
        raise ConfigError(msg)
    allowed = {
        "weak_hyperedges", "max_arity", "min_tokens", "max_tokens", "max_window_sentences",
        "token_counter",
    }
    unknown = set(spec) - allowed
    if unknown:
        msg = f"graph_assembly config 'hyperedges' has unknown key(s): {sorted(unknown)}."
        raise ConfigError(msg)
    counter, counter_id = _resolve_token_counter(spec.get("token_counter"))
    try:
        config = HyperedgeConfig(
            weak_hyperedges=bool(spec.get("weak_hyperedges", True)),
            max_arity=int(spec.get("max_arity", 12)),
            min_tokens=int(spec.get("min_tokens", 25)),
            max_tokens=int(spec.get("max_tokens", 180)),
            max_window_sentences=int(spec.get("max_window_sentences", 3)),
        )
    except (TypeError, ValueError) as exc:
        msg = f"graph_assembly config 'hyperedges' is malformed: {exc}"
        raise ConfigError(msg) from exc
    if counter is not None and counter_id is not None:
        config = replace(config, token_counter=counter, token_counter_id=counter_id)
    return config


def _resolve_token_counter(
    spec: Any,
) -> tuple[Callable[[str], int] | None, str | None]:
    """Resolve the OPTIONAL operator-named exact token counter (ADR-0058, index side).

    ``None`` (the default) keeps the deterministic in-core heuristic — nothing is pinned and
    nothing downloads. ``"tiktoken:<encoding>"`` names a license-verified encoding the
    operator cleared separately (the same discipline as the retrieval Packer's
    ``TiktokenCounter``): ``tiktoken`` is imported lazily and ONLY here, a missing dep or an
    unknown encoding is a loud wiring-time ConfigError, and the full spec string becomes the
    stored counter identity so a query-time mismatch stays detectable (ADR-0058).
    """
    if spec is None:
        return (None, None)
    if not isinstance(spec, str) or not spec.startswith("tiktoken:") or spec == "tiktoken:":
        msg = (
            "graph_assembly config 'hyperedges.token_counter' must be "
            f"'tiktoken:<encoding>', got {spec!r}."
        )
        raise ConfigError(msg)
    encoding_name = spec.split(":", 1)[1]
    try:
        import tiktoken
    except ImportError as exc:
        msg = (
            f"hyperedges.token_counter {spec!r} needs the 'tiktoken' package, which is not "
            "installed (it is never a core dependency — install it explicitly)."
        )
        raise ConfigError(msg) from exc
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except (KeyError, ValueError) as exc:
        msg = f"hyperedges.token_counter names unknown tiktoken encoding {encoding_name!r}."
        raise ConfigError(msg) from exc

    def count(text: str) -> int:
        return len(encoding.encode(text))

    return (count, spec)


def _stamp_hyperedge_descriptions(
    node_records: list[GraphRecord],
    hyperedges: list[GraphHyperedge],
) -> list[GraphRecord]:
    """Stamp ``description`` + ``hyperedge_degree`` onto each node's properties (spec 1.4).

    The description is composed DETERMINISTICALLY — canonical name, entity type, and the
    top-3 most frequent relation contexts drawn from the node's Hyperedges (head side
    ``label: other``, tail side ``label by: other``; ties broken lexicographically) — so the
    entity index is query-matchable without any LLM summarisation (ADR-0057/0059's tier-0
    embedding input). Runs ONLY on the hyperedge-configured path, so an unconfigured stack's
    node stream stays byte-identical. Nodes untouched by any Hyperedge get degree 0 and the
    name+type description — never a missing key on one node and not another.
    """
    name_of = {
        r.node.node_id: r.node.canonical_name for r in node_records if r.node is not None
    }
    degree: dict[str, int] = {}
    contexts: dict[str, Counter[str]] = {}
    for h in hyperedges:
        for node_id in h.entity_node_ids:
            degree[node_id] = degree.get(node_id, 0) + 1
        for rel in h.relations:
            head_name = name_of.get(rel.head_node_id, rel.head_node_id)
            tail_name = name_of.get(rel.tail_node_id, rel.tail_node_id)
            contexts.setdefault(rel.head_node_id, Counter())[
                f"{rel.label}: {tail_name}"
            ] += 1
            contexts.setdefault(rel.tail_node_id, Counter())[
                f"{rel.label} by: {head_name}"
            ] += 1

    stamped: list[GraphRecord] = []
    for record in node_records:
        node = record.node
        if node is None:  # pragma: no cover - node_records are all NODE-scope by construction
            stamped.append(record)
            continue
        top = sorted(
            contexts.get(node.node_id, Counter()).items(), key=lambda kv: (-kv[1], kv[0])
        )[:3]
        description = f"{node.canonical_name} — {node.label}"
        if top:
            description += " — " + ", ".join(text for text, _ in top)
        properties = {
            **node.properties,
            "description": description,
            "hyperedge_degree": degree.get(node.node_id, 0),
        }
        new_node = node.model_copy(update={"properties": properties})
        stamped.append(record.model_copy(update={"node": new_node}))
    return stamped


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
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The canonical Graph Assembler builds content-addressed nodes/edges + Evidence.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
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
        # ADR-0057: the OPTIONAL Hyperedge projection, following the graph_features pattern
        # exactly — absent ⇒ no builder ⇒ the emitted stream is byte-identical to today AND the
        # Runner never streams the document-level records this projection alone needs (the
        # ``wants_document_streams`` gate below).
        self._hyperedge_config = _resolve_hyperedge_config(config.get("hyperedges"))

    @property
    def wants_document_streams(self) -> bool:
        """True when the Runner should feed chunks/mentions/relations (hyperedges on).

        The Runner checks this attribute (default absent/False on every other Provider) before
        choosing :meth:`assemble_with_documents` over :meth:`assemble`, so a stack that never
        configures hyperedges never pays the extra stream reads — additive at the seam, no
        Capability-protocol change.
        """
        return self._hyperedge_config is not None

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        yield from self.assemble_with_documents(records, chunks=(), mentions=(), relations=())

    def assemble_with_documents(
        self,
        records: Iterable[DisambiguationRecord],
        *,
        chunks: Iterable[ChunkRecord] = (),
        mentions: Iterable[EntityMention] = (),
        relations: Iterable[RelationMention] = (),
    ) -> Iterator[GraphRecord]:
        """Assemble nodes + edges, then (when configured, ADR-0057) the Hyperedge stream.

        The document-level streams are consumed only when ``hyperedges`` is configured; with
        the projection off this is exactly :meth:`assemble` — same records, same order, same
        bytes.
        """
        entities, entity_src, relation_list, relation_src = _partition(records)
        builder = GraphBuilder(
            corpus_id=self._corpus_id, max_evidence_per_node=self._max_evidence_per_node
        )
        nodes, edges = builder.build(entities, relation_list)

        node_records = [self._node_record(node, entity_src[node.entity_id]) for node in nodes]
        edge_records = [self._edge_record(edge, relation_src[edge.relation_id]) for edge in edges]
        # ADR-0051: stamp centrality + community onto each node (a no-op when unconfigured, so the
        # emitted stream — nodes then edges, in id order — is byte-identical to today).
        node_records = _stamp_features(node_records, edge_records, self._feature_computer)

        if self._hyperedge_config is None:
            yield from node_records
            yield from edge_records
            return

        # ADR-0057: the m-ary projection of the SAME inputs. Built BEFORE the nodes are
        # yielded so each node can carry its Hyperedge-derived ``description`` (spec 1.4 —
        # the deterministic embedding input that makes the entity index query-matchable
        # without any LLM summarisation) and ``hyperedge_degree``. Emission order is
        # unchanged: nodes, edges, then Hyperedges in hyperedge-id order. Only canonicalised
        # entities can join (node_id_for is the same deterministic mapping the NODE stream
        # used), so Hyperedges never reference a node the graph does not carry.
        hyperedge_builder = HyperedgeBuilder(self._corpus_id, self._hyperedge_config)
        hyperedges = hyperedge_builder.build(
            chunks, mentions, relations, entities, builder.node_id_for
        )
        node_records = _stamp_hyperedge_descriptions(node_records, hyperedges)
        yield from node_records
        yield from edge_records
        for hyperedge in hyperedges:
            yield GraphRecord(
                record_id=f"{hyperedge.record_id}#graph",
                scope=GraphScope.HYPEREDGE,
                hyperedge=hyperedge,
                provenance=hyperedge.provenance,
                classification=hyperedge.classification,
            )

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
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The weighted Graph Assembler folds parallel same-typed relations into weighted edges.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
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
