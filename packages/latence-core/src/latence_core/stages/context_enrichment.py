"""Built-in reference ContextEnricher Provider (deterministic, CPU-viable, zero-dep; W13).

The OPTIONAL post-Assembly context-enrichment reference Provider (ADR-0039; ships in core like every
other reference Provider so the seam runs end to end offline, ADR-0007 CPU-first, ADR-0016 no extra
deps). It projects the run's ALREADY-computed knowledge graph back onto each
:class:`~latence_core.contracts.ChunkRecord` as a compact
:class:`~latence_core.contracts.ContextHeader` — the chunk's canonical entities + their top-k
KG-neighbor triples — which the Export prepends to the EMBEDDING input ONLY. Coherence goes into the
vector + a metadata column; the stored chunk text stays the clean, byte-unchanged chunk.

The build is pure-Python and DETERMINISTIC (the Baseline/G1 bar): every collection is sorted, the
neighbor triples are confidence-sorted then CAPPED, and the node ids are the graph's content-
addressed ids — so a seeded run is byte-identical. The bloat discipline (the crux of W13, ADR-0039)
is enforced structurally:

1. **Compact triples, not neighbor prose** — each neighbor is rendered ``A —label→ B``.
2. **Top-k neighbor cap + confidence sort (the hub-entity guard, mandatory)** — per entity at most
   ``max_neighbors_per_entity`` of its highest-confidence relations survive, and an overall
   ``max_triples`` caps the whole header, so a corpus-wide HUB entity present in many chunks
   contributes only its TOP relations, never all of them.
3. **Keys not raw text** — ``kg_node_ids`` (the chunk's own entities) + ``neighbor_node_ids`` (the
   entities the triples reach) are join keys a downstream retriever expands on (query-time KG
   expansion is the RETRIEVER's job — ADR-0017 files-not-a-DB; W13 only emits the keys).
4. **Embed-but-don't-store-in-content** — the Export owns the prepend (embedding input only), so the
   header never inflates the exported chunk (:mod:`latence_core.stages.export`).

The optional LLM situating-sentence mode (Contextual-Retrieval style, deterministic=False) is a
DOCUMENTED follow-on (ADR-0039), not built here: ``situating_sentence``/``model_id`` stay ``None``
so
W13 is byte-identical + offline (no model, no key in the default path). A chunk with no
confidence-gated canonical entity gets ``context_header=None`` (nothing to add — byte-identical to a
run without the Stage for that chunk).

**Memory shape.** A ContextHeader is a handful of strings per chunk, but the inputs it is derived
from are the two biggest streams in the run. The S3 UDA dataset (220 129 chunks / 2 603 887
EntityMentions / 516 512 canonical entities) was OOM-killed HERE, inside a container with a hard
29 GB cgroup cap (``memory.limit_in_bytes`` 30999998464; the cgroup ``oom_kill`` counter went
3 → 4) — the same shape bug the Hyperedge builder had, in the next Stage along. The cause was that
the run's mentions were grouped per chunk as **pydantic records** (a measured ~3.35 kB per
:class:`~latence_core.contracts.EntityMention`, each dragging its own ``Provenance`` +
``Classification`` submodel: 8.1 GiB of mentions before a single header was built), even though
:meth:`ContextHeaderEnricher._build_header` reads exactly three fields off a mention and keeps only
a per-(chunk, entity) MAXIMUM of one of them. So:

* the mention stream is **folded on arrival** into :class:`_ChunkEntities` — per chunk, the
  ``node_id -> strongest gated mention confidence`` map that ``_build_header`` actually consumes.
  No mention record and no per-mention projection is ever retained: 2.6 M pydantic records collapse
  to ~57 MiB of floats keyed by the chunk's own (far fewer) canonical entities;
* ``GraphNode`` / ``GraphEdge`` are projected to :class:`_NodeView` / :class:`_EdgeMeta` on arrival
  too — the ~6 fields the header build and the triple rendering read, instead of the full record
  with its ``properties`` bag, ``evidence``, ``member_mention_ids`` and Provenance;
* the mention→node join, the single genuinely mention-sized table, is scoped to the fold and
  **dropped before the chunk loop starts**, so it is not resident while headers are being built.

Folding on arrival only bounds the Stage if nothing UPSTREAM is holding the records instead. It was
not: the Runner gathered this Stage's ``depends_on`` inputs into one list and partitioned it by
carrier, so every ChunkRecord and every GraphRecord stayed resident for the Stage's whole execution
regardless of what this module retained. Because ``enrich`` consumes each of its three inputs
exactly once (pinned by ``test_context_enrichment_memory_shape``), the Runner now hands it one
independent lazy stream per carrier instead (the declared ``PER_CARRIER`` input shape, ADR-0033)
— which is what makes the bound below real end to end rather than only inside this file.

What is deliberately NOT done: the folded per-chunk map is *not* popped as each chunk is emitted.
Unlike the Hyperedge builder — which dedups its chunk views by ``record_id`` — this Provider threads
the chunk stream 1:1 and cannot dedup, and a Pipeline may legitimately wire it to more than one
chunk-carrying parent (Chunk *and* Redaction both re-emit the same ``record_id``, ADR-0042). Popping
would silently strip the header from every repeat. The map is the folded aggregate, bounded by
distinct (chunk, entity) pairs rather than by mentions, so keeping it whole costs ~57 MiB at UDA
scale instead of the 8.1 GiB it replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    ContextHeader,
    GraphScope,
)
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import (
        ChunkRecord,
        EntityMention,
        GraphEdge,
        GraphNode,
        GraphRecord,
    )

#: Rendered-triple arrow (``A —label→ B``) — a compact, deterministic relation surface.
_ARROW_L = " —"
_ARROW_R = "→ "


@dataclass
class _NodeView:
    """The ONLY fields this module reads off a :class:`~latence_core.contracts.GraphNode`.

    Projected on arrival (module docstring, *Memory shape*) so the KG index costs ~7 slots per
    entity instead of a full pydantic node with its ``properties`` bag, ``evidence``,
    ``member_mention_ids``, ``source_document_ids``, Provenance and Classification.

    ``centrality``/``community`` are the ADR-0051 ``graph_features`` stamps read out of
    ``properties`` ONCE, with the exact acceptance rule :func:`_graph_features` used to apply per
    call: both must be present and well-typed, otherwise BOTH are ``None`` and the node counts as
    unstamped (a malformed value is treated as absent, so a hand-built graph never crashes).

    ``__slots__`` is spelled out (rather than ``@dataclass(slots=True)``) so ``__weakref__``
    survives: the memory-shape regression test counts live instances of this class.
    """

    __slots__ = (
        "__weakref__",
        "canonical_name",
        "centrality",
        "community",
        "confidence",
        "label",
        "node_id",
    )

    node_id: str
    canonical_name: str
    label: str
    confidence: float
    centrality: float | None
    community: str | None


@dataclass
class _EdgeMeta:
    """The ONLY fields read off a :class:`~latence_core.contracts.GraphEdge` (see
    :class:`_NodeView`)."""

    __slots__ = (
        "__weakref__",
        "confidence",
        "edge_id",
        "label",
        "source_node_id",
        "target_node_id",
    )

    edge_id: str
    source_node_id: str
    target_node_id: str
    label: str
    confidence: float


@dataclass
class _ChunkEntities:
    """One chunk's gated canonical entities, folded out of the corpus-wide mention stream.

    ``best_mention_confidence`` maps ``node_id -> the STRONGEST gated mention confidence this chunk
    has for that entity`` — the only thing :meth:`ContextHeaderEnricher._build_header` ever derived
    from a chunk's mentions. Folding on arrival (a maximum, so order-independent and identical to
    the per-chunk pass it replaces) is what keeps the run's 2.6 M mentions from being resident as
    pydantic records; see the module docstring.

    Non-empty by construction: an instance is created only when a gated (mention, node) pair is
    actually admitted, so a chunk whose every mention is rejected has NO entry at all — which is the
    same ``context_header=None`` outcome as the empty-map check it replaces.
    """

    __slots__ = ("__weakref__", "best_mention_confidence", "chunk_record_id")

    chunk_record_id: str
    best_mention_confidence: dict[str, float]


def _node_view(node: GraphNode) -> _NodeView:
    raw_centrality = node.properties.get("centrality")
    raw_community = node.properties.get("community")
    centrality: float | None = None
    community: str | None = None
    if isinstance(raw_centrality, (int, float)) and isinstance(raw_community, str):
        centrality = float(raw_centrality)
        community = raw_community
    return _NodeView(
        node_id=node.node_id,
        canonical_name=node.canonical_name,
        label=node.label,
        confidence=node.confidence,
        centrality=centrality,
        community=community,
    )


def _edge_meta(edge: GraphEdge) -> _EdgeMeta:
    return _EdgeMeta(
        edge_id=edge.edge_id,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        label=edge.label,
        confidence=edge.confidence,
    )


class _Neighbor:
    """One gated KG edge as seen from an entity: the rendered triple + its ranking keys.

    Pure ranking state (not a contract): ``confidence`` + the rendered ``triple`` + the
    deterministic
    ``edge_id`` tiebreak give a total order so the top-k cap is byte-stable, and
    ``neighbor_node_id``
    is the OTHER endpoint (the query-time KG-expansion key).
    """

    __slots__ = ("confidence", "edge_id", "neighbor_node_id", "triple")

    def __init__(
        self, confidence: float, triple: str, neighbor_node_id: str, edge_id: str
    ) -> None:
        self.confidence = confidence
        self.triple = triple
        self.neighbor_node_id = neighbor_node_id
        self.edge_id = edge_id

    def sort_key(self) -> tuple[float, str, str]:
        # Highest confidence first (negated), then the rendered triple, then the edge id — a total,
        # content-addressed order so ties never depend on iteration order (the Baseline bar).
        return (-self.confidence, self.triple, self.edge_id)


def _graph_features(
    ordered_entities: list[_NodeView],
) -> tuple[list[float], list[str]]:
    """The chunk's entities' centrality + community, aligned with ``ordered_entities`` (ADR-0051).

    Reads the ``centrality`` / ``community`` the ``graph_features`` enrichment stamped onto each
    :class:`~latence_core.contracts.GraphNode`'s ``properties`` bag — resolved once, on arrival, by
    :func:`_node_view` (a malformed value is treated as absent there, so a hand-built graph never
    crashes enrichment). Returns aligned lists ONLY when EVERY entity carries both (the assembler
    stamps all-or-none, so an enriched graph has them on every node); otherwise returns ``([], [])``
    — byte-identical to a run without it (the header's fields stay empty).
    """
    centrality: list[float] = []
    community: list[str] = []
    for node in ordered_entities:
        if node.centrality is None or node.community is None:
            return [], []
        centrality.append(node.centrality)
        community.append(node.community)
    return centrality, community


class ContextHeaderEnricher:
    """ContextEnricher Provider: chunks + mentions + assembled graph -> the SAME chunks, enriched.

    Pure Python, deterministic, zero extra deps (ADR-0007/0016/0039). It never drops or reorders the
    chunk stream (threaded 1:1) and never touches a chunk's offsets/provenance/content — it only
    attaches ``context_header`` via ``chunk.model_copy(update=...)`` (exactly the
    additive-annotation
    shape the schema inducer uses), so a chunk with no gated canonical entity flows through
    unchanged
    (``context_header`` stays ``None``, byte-identical to a run without the Stage).

    Config:
        max_neighbors_per_entity: int  — top-k cap on the relations ONE entity contributes (the
            hub-entity guard). Default 4. 0 = unbounded (not recommended — defeats the anti-bloat).
        max_triples: int  — overall cap on the header's neighbor triples. Default 6. 0 = unbounded.
        min_entity_confidence: float  — drop a chunk's mentions (and their canonical entities) below
            this 0..1 confidence. Default 0.0.
        min_edge_confidence: float  — drop KG edges below this 0..1 confidence before ranking
        neighbor
            triples. Default 0.0.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The reference Context Enricher projects the KG onto chunks by a deterministic, capped
        adjacency read. A pure-Python part of ``latence-core`` (Apache-2.0,
        verified per ADR-0012 — the package's own code, weights-and-code trivially the same), it
        ships no model weights (``model_id=None``), is deterministic, CPU-only, does not batch, and
        has no per-call cost, so a bake-off/conformance run reads honest, non-fabricated columns off
        it. (The optional LLM situating-sentence mode is a separate, non-default Provider —
        ADR-0039.)
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-10",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._max_neighbors = int(config.get("max_neighbors_per_entity", 4))
        self._max_triples = int(config.get("max_triples", 6))
        self._min_entity_confidence = float(config.get("min_entity_confidence", 0.0))
        self._min_edge_confidence = float(config.get("min_edge_confidence", 0.0))
        # The corpus-level anti-bloat witness (counts-only, no content): how many (chunk, entity)
        # pairs had MORE gated neighbor relations than the per-entity cap, so only the entity's
        # TOP-k
        # survived — the hub-entity guard biting. Drained by tests + surfaced honestly.
        self.hub_entities_capped = 0

    def enrich(
        self,
        chunks: Iterable[ChunkRecord],
        mentions: Iterable[EntityMention],
        graph_records: Iterable[GraphRecord],
    ) -> Iterator[ChunkRecord]:
        self.hub_entities_capped = 0
        # Materialise the assembled KG (bounded by the KG size, like GraphCompleter's input) once:
        # node lookup, mention->node join, and per-node gated adjacency (each edge seen from BOTH
        # endpoints, rendered in its true asserted direction). Every graph record is projected to
        # its compact view on arrival and dropped (module docstring, *Memory shape*).
        nodes, mention_to_node, adjacency = self._index_graph(graph_records)
        # Fold each chunk's mentions into its gated canonical entities as they stream past (a
        # chunk's canonical entities are the nodes its mentions belong to). The mention stream is
        # corpus-wide; a chunk with no gated mention gets no entry at all.
        by_chunk = self._fold_mentions(mentions, nodes, mention_to_node)
        # The mention->node join is the one mention-sized table in the Stage, and nothing past the
        # fold reads it — drop it before a single header is built, so it is not resident for the
        # whole chunk loop.
        del mention_to_node

        for chunk in chunks:
            entities = by_chunk.get(chunk.record_id)
            if entities is None:
                # No gated canonical entity: nothing to add — pass the chunk through byte-identical.
                yield chunk
            else:
                header = self._build_header(entities, nodes, adjacency)
                yield chunk.model_copy(update={"context_header": header})

    def _index_graph(
        self, graph_records: Iterable[GraphRecord]
    ) -> tuple[dict[str, _NodeView], dict[str, str], dict[str, list[_Neighbor]]]:
        """Build the node lookup, mention->node join, and per-node gated adjacency from the KG."""
        nodes: dict[str, _NodeView] = {}
        mention_to_node: dict[str, str] = {}
        edges: list[_EdgeMeta] = []
        for record in graph_records:
            if record.scope is GraphScope.NODE and record.node is not None:
                node = record.node
                nodes[node.node_id] = _node_view(node)
                for mention_id in node.member_mention_ids:
                    mention_to_node[mention_id] = node.node_id
            elif record.scope is GraphScope.EDGE and record.edge is not None:
                edge = record.edge
                if edge.confidence >= self._min_edge_confidence:
                    edges.append(_edge_meta(edge))

        adjacency: dict[str, list[_Neighbor]] = {}
        for meta in edges:
            src = nodes.get(meta.source_node_id)
            tgt = nodes.get(meta.target_node_id)
            if src is None or tgt is None:
                # A dangling edge (endpoint not in the node set) has no renderable canonical name —
                # skip it (the assembler already validates endpoints, so this is belt-and-braces).
                continue
            triple = f"{src.canonical_name}{_ARROW_L}{meta.label}{_ARROW_R}{tgt.canonical_name}"
            # Seen from the source node, the neighbor is the target; from the target it is the
            # source. The rendered triple is identical (the edge's asserted direction) either way.
            adjacency.setdefault(meta.source_node_id, []).append(
                _Neighbor(meta.confidence, triple, meta.target_node_id, meta.edge_id)
            )
            adjacency.setdefault(meta.target_node_id, []).append(
                _Neighbor(meta.confidence, triple, meta.source_node_id, meta.edge_id)
            )
        return nodes, mention_to_node, adjacency

    def _fold_mentions(
        self,
        mentions: Iterable[EntityMention],
        nodes: dict[str, _NodeView],
        mention_to_node: dict[str, str],
    ) -> dict[str, _ChunkEntities]:
        """Fold the corpus-wide mention stream into per-chunk gated canonical entities.

        Each chunk's canonical entities are its gated mentions' canonical nodes (BOTH the mention
        confidence and the node confidence must clear ``min_entity_confidence``), carrying the
        STRONGEST gated mention confidence per node — this chunk's best NER evidence for the entity,
        so the header can report a real per-entity confidence downstream (ADR-0053).

        Folding here rather than in :meth:`_build_header` is the memory fix: a maximum is
        order-independent, so this is exactly the per-chunk pass it replaces, but no mention record
        (nor any per-mention projection of one) is ever retained — see the module docstring.
        """
        by_chunk: dict[str, _ChunkEntities] = {}
        for mention in mentions:
            confidence = mention.confidence
            if confidence < self._min_entity_confidence:
                continue
            node_id = mention_to_node.get(mention.record_id)
            if node_id is None:
                continue
            node = nodes.get(node_id)
            if node is None or node.confidence < self._min_entity_confidence:
                continue
            chunk_id = mention.chunk_record_id
            entities = by_chunk.get(chunk_id)
            if entities is None:
                entities = _ChunkEntities(chunk_record_id=chunk_id, best_mention_confidence={})
                by_chunk[chunk_id] = entities
            best = entities.best_mention_confidence
            prev = best.get(node_id)
            if prev is None or confidence > prev:
                best[node_id] = confidence
        return by_chunk

    def _build_header(
        self,
        chunk_entities: _ChunkEntities,
        nodes: dict[str, _NodeView],
        adjacency: dict[str, list[_Neighbor]],
    ) -> ContextHeader:
        """Build one chunk's capped, confidence-gated ContextHeader from its folded entities.

        ``chunk_entities`` is non-empty by construction (:class:`_ChunkEntities`); a chunk with no
        gated canonical entity never reaches here and keeps ``context_header=None``.
        """
        # 1. The chunk's canonical entities, already folded out of its mentions by
        #    :meth:`_fold_mentions`: node_id -> the strongest gated mention confidence.
        best_mention_conf = chunk_entities.best_mention_confidence
        own_ids = sorted(best_mention_conf)
        # Entities/types aligned + sorted by canonical name (then node id) — deterministic surface.
        ordered_entities = sorted(
            (nodes[nid] for nid in own_ids), key=lambda n: (n.canonical_name, n.node_id)
        )
        entities = [n.canonical_name for n in ordered_entities]
        entity_types = [n.label for n in ordered_entities]
        kg_node_ids = [n.node_id for n in ordered_entities]
        # Per-entity ADMISSION confidence, aligned with ``entities``: min(best gated mention conf,
        # node conf) — the exact value v s.t. this gate at threshold t admits the entity iff v >= t.
        # Carrying it lets a composing SparseEmbedder's ``confidence_threshold`` apply the identical
        # rule at index time (ADR-0053; #162 AC-04), so weak NER never pollutes the sparse signal.
        entity_confidences = [
            min(best_mention_conf[n.node_id], n.confidence) for n in ordered_entities
        ]
        # ADR-0051: project the index-time graph features onto the header, aligned with kg_node_ids.
        # All-or-nothing: only when EVERY one of the chunk's own entities carries the stamped
        # centrality + community (the assembler stamps every node, so an enriched graph has them on
        # all) — otherwise both stay empty, byte-identical to a run without graph_features.
        centrality, community = _graph_features(ordered_entities)

        # 2. The chunk's KG-neighbor triples: per-entity top-k capped (the hub-entity guard), then
        #    an overall cap. Iterate entities in node-id order for determinism.
        kept: list[_Neighbor] = []
        for node_id in own_ids:
            neighbors = sorted(adjacency.get(node_id, []), key=lambda n: n.sort_key())
            cap = self._max_neighbors if self._max_neighbors > 0 else len(neighbors)
            if len(neighbors) > cap:
                # This entity had more gated relations than the per-entity cap: only its TOP-k
                # survive — the hub-entity guard biting. Counted (counts-only, no content).
                self.hub_entities_capped += 1
            kept.extend(neighbors[:cap])

        # Dedup identical triples (an internal relation between two of the chunk's own entities is
        # reachable from both endpoints), keeping the highest-confidence witness, then apply the
        # overall cap on the confidence-sorted, content-tiebroken order.
        best_by_triple: dict[str, _Neighbor] = {}
        for neigh in kept:
            existing = best_by_triple.get(neigh.triple)
            if existing is None or neigh.sort_key() < existing.sort_key():
                best_by_triple[neigh.triple] = neigh
        ranked = sorted(best_by_triple.values(), key=lambda n: n.sort_key())
        overall_cap = self._max_triples if self._max_triples > 0 else len(ranked)
        capped = ranked[:overall_cap]

        neighbor_triples = [n.triple for n in capped]
        own_id_set = set(kg_node_ids)
        # Query-time expansion keys: the neighbor endpoints the kept triples reach, EXCLUDING the
        # chunk's own entities (an internal-only relation contributes no external neighbor).
        neighbor_node_ids = sorted(
            {n.neighbor_node_id for n in capped} - own_id_set
        )

        return ContextHeader(
            entities=entities,
            entity_types=entity_types,
            entity_confidences=entity_confidences,
            neighbor_triples=neighbor_triples,
            kg_node_ids=kg_node_ids,
            neighbor_node_ids=neighbor_node_ids,
            kg_node_centrality=centrality,
            kg_node_community=community,
            situating_sentence=None,
            model_id=None,
        )
