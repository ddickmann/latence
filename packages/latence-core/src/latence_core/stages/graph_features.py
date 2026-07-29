"""Built-in reference GraphFeatureComputer Providers (deterministic, CPU-viable, zero-dep; #161).

The two in-core reference :class:`~latence_core.capability.GraphFeatureComputer` Providers behind
the OPTIONAL index-time graph-feature enrichment (ADR-0051). Each consumes the assembled graph's
:class:`~latence_core.contracts.GraphNode`s + :class:`~latence_core.contracts.GraphEdge`s and
returns one :class:`~latence_core.capability.GraphFeatures` — a **centrality** score and a
**community** id per node — which the Graph-Assembly Stage stamps onto every node's ``properties``
bag (``centrality`` / ``community``). Those flow into ``graph-nodes.parquet`` columns and, via
Context Enrichment, onto each chunk's :class:`~latence_core.contracts.ContextHeader` — the two
consumers ADR-0051 names (graph-augmented retrieval and the Knapsack packer's ``centrality`` /
``cluster_ids`` inputs).

Two families, one seam (behaviour VARIES → two adapters, the mattpocock bar):

* :class:`DegreeComponentGraphFeatureComputer` (``graph_features.degree``) — normalised undirected
  **degree centrality** + **connected-components** community (Union-Find). Cheap, exact, the natural
  default.
* :class:`PageRankLabelPropGraphFeatureComputer` (``graph_features.pagerank``) — **PageRank**
  centrality (directed power iteration) + **label-propagation** community. A flow-based centrality
  and a soft clustering that can split a component that connected-components merges.

Both wrap the pure, dependency-free :mod:`latence_core.graph_features` engine and are DETERMINISTIC
(fixed iteration order, rounded scores, content-addressed community ids), so a seeded run is
byte-reproducible (ADR-0007 CPU-first, ADR-0016 no extra deps). They ship no model weights and name
no model (ADR-0012). They are *references*, not the blessed Louvain/Leiden community detection
(ADR-0051) — that ships as a pluggable, license-verified package, exactly as SPLADE/MUVERA do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.capability import GraphFeatures
from latence_core.graph_features import (
    connected_component_communities,
    degree_centrality,
    label_propagation_communities,
    pagerank_centrality,
)
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from latence_core.contracts import GraphEdge, GraphNode


def _reference_profile() -> ProviderProfile:
    """The shared descriptor: pure-code, CPU, deterministic, PolyForm-Noncommercial-1.0.0 (ADR-0036
    §2 / ADR-0012).

    A reference graph-feature computer is a pure-Python part of ``latence-core``
    (PolyForm-Noncommercial-1.0.0, verified per ADR-0012 — the package's own code, weights+code
    trivially the same): it ships no model weights (``model_id=None``), is deterministic, CPU-only,
    does not batch, and has no per-call cost, so a bake-off / conformance run reads honest,
    non-fabricated columns off it.
    """
    return ProviderProfile(
        compute="cpu",
        memory_mb=32,
        model_id=None,
        license=REPO_LICENSE_CODE,
        license_verified=True,
        license_code=REPO_LICENSE_CODE,
        license_source=REPO_LICENSE_SOURCE,
        license_verified_on="2026-07-16",
        deterministic=True,
        batch=False,
        cost_per_1k=None,
    )


def _node_ids_and_edges(
    nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
) -> tuple[list[str], list[tuple[str, str]]]:
    """The plain ``node_id`` list + directed ``(source, target)`` tuples the pure engine reads."""
    node_ids = [node.node_id for node in nodes]
    edge_pairs = [(edge.source_node_id, edge.target_node_id) for edge in edges]
    return node_ids, edge_pairs


class DegreeComponentGraphFeatureComputer:
    """GraphFeatureComputer: degree centrality + connected-components community (degree family).

    Registered ``graph_features.degree``. Config: none (the algorithms are parameter-free). Pure
    Python, deterministic, zero extra deps.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # No configurable knobs (parameter-free algorithms); accept + ignore a config for the
        # uniform Provider constructor shape the runner/registry instantiate with.
        _ = config

    def compute_features(
        self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
    ) -> GraphFeatures:
        node_ids, edge_pairs = _node_ids_and_edges(nodes, edges)
        # The engine dedupes/sorts nodes and drops edges to unknown/self endpoints internally; pass
        # the raw lists so the two families see the identical cleaned graph.
        from latence_core.graph_features import _clean_edges, _sorted_nodes

        ordered = _sorted_nodes(node_ids)
        clean = _clean_edges(set(ordered), edge_pairs)
        centrality = degree_centrality(ordered, clean)
        community = connected_component_communities(ordered, clean)
        return GraphFeatures(centrality=centrality, community=community)


class PageRankLabelPropGraphFeatureComputer:
    """GraphFeatureComputer: PageRank centrality + label-propagation community (pagerank family).

    Registered ``graph_features.pagerank``. Config: none (the power-iteration + propagation bounds
    are fixed for determinism). Pure Python, deterministic, zero extra deps.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        _ = config

    def compute_features(
        self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
    ) -> GraphFeatures:
        node_ids, edge_pairs = _node_ids_and_edges(nodes, edges)
        from latence_core.graph_features import _clean_edges, _sorted_nodes

        ordered = _sorted_nodes(node_ids)
        clean = _clean_edges(set(ordered), edge_pairs)
        centrality = pagerank_centrality(ordered, clean)
        community = label_propagation_communities(ordered, clean)
        return GraphFeatures(centrality=centrality, community=community)
