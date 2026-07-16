"""Graph-feature enrichment (ADR-0051, #161): centrality + community engines, Providers, wiring.

The index-time graph-feature enrichment computes a centrality measure + a community id per node and
projects them onto (1) graph-nodes.parquet columns and (2) each chunk's context_header — additively,
byte-identical when the enrichment is off. These tests exercise the pure engines, the two reference
Providers (through the GraphFeatureComputer seam), the assembler wiring, the KG-export columns, and
the context-header projection — asserting BEHAVIOUR (values, alignment, determinism), never a flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from latence_core.capability import GraphFeatureComputer, GraphFeatures
from latence_core.conformance import fixtures
from latence_core.contracts import ChunkRecord, GraphScope
from latence_core.errors import ConfigError, ContractError
from latence_core.graph_features import (
    connected_component_communities,
    degree_centrality,
    label_propagation_communities,
    pagerank_centrality,
)
from latence_core.stages.context_enrichment import ContextHeaderEnricher
from latence_core.stages.graph_assembly import (
    CanonicalGraphAssembler,
    WeightedGraphAssembler,
)
from latence_core.stages.graph_export import KnowledgeGraphExport
from latence_core.stages.graph_features import (
    DegreeComponentGraphFeatureComputer,
    PageRankLabelPropGraphFeatureComputer,
)
from latence_core.storage import Storage

_DEGREE = {"provider": "graph_features.degree", "config": {}}
_PAGERANK = {"provider": "graph_features.pagerank", "config": {}}


# ---------------------------------------------------------------------------
# Pure engines — plain string/tuple inputs (no Pydantic)
# ---------------------------------------------------------------------------


def test_degree_centrality_is_normalised_neighbour_fraction() -> None:
    # A star: hub 'a' adjacent to b, c, d; each leaf adjacent only to 'a'. n=4 → denom 3.
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("a", "c"), ("a", "d")]
    centrality = degree_centrality(nodes, edges)
    assert centrality["a"] == pytest.approx(3 / 3)  # adjacent to all 3 others
    assert centrality["b"] == pytest.approx(1 / 3)
    assert centrality["c"] == pytest.approx(1 / 3)


def test_degree_centrality_single_node_is_zero() -> None:
    assert degree_centrality(["only"], []) == {"only": 0.0}


def test_connected_components_split_into_min_id_communities() -> None:
    # Two disjoint components: {a,b} and {c,d}. Community id = the min member id.
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("c", "d")]
    community = connected_component_communities(nodes, edges)
    assert community["a"] == community["b"] == "a"
    assert community["c"] == community["d"] == "c"
    assert community["a"] != community["c"]


def test_isolated_node_is_its_own_community() -> None:
    community = connected_component_communities(["a", "b", "x"], [("a", "b")])
    assert community["x"] == "x"
    assert community["a"] == community["b"] == "a"


def test_pagerank_sums_to_one_and_ranks_the_hub_highest() -> None:
    # A directed graph where every node points at the hub 'h' → h accumulates the most rank.
    nodes = ["h", "a", "b", "c"]
    edges = [("a", "h"), ("b", "h"), ("c", "h")]
    rank = pagerank_centrality(nodes, edges)
    assert sum(rank.values()) == pytest.approx(1.0, abs=1e-6)
    assert rank["h"] == max(rank.values())
    assert rank["h"] > rank["a"]


def test_pagerank_empty_graph_is_empty() -> None:
    assert pagerank_centrality([], []) == {}


def test_label_propagation_merges_a_clique_into_one_community() -> None:
    # A triangle (clique) collapses to a single community; a separate pair to another.
    nodes = ["a", "b", "c", "x", "y"]
    edges = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y")]
    community = label_propagation_communities(nodes, edges)
    assert community["a"] == community["b"] == community["c"] == "a"
    assert community["x"] == community["y"] == "x"


def test_engines_are_deterministic_over_two_runs() -> None:
    nodes = ["n3", "n1", "n2", "n4"]
    edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n1"), ("n4", "n1")]
    assert pagerank_centrality(nodes, edges) == pagerank_centrality(nodes, edges)
    assert label_propagation_communities(nodes, edges) == label_propagation_communities(
        nodes, edges
    )


# ---------------------------------------------------------------------------
# The GraphFeatures contract + the two Providers behind the seam
# ---------------------------------------------------------------------------


def test_both_providers_satisfy_the_capability() -> None:
    assert isinstance(DegreeComponentGraphFeatureComputer(), GraphFeatureComputer)
    assert isinstance(PageRankLabelPropGraphFeatureComputer(), GraphFeatureComputer)


def _assembled_nodes_edges() -> tuple[list[object], list[object]]:
    records = fixtures.assembled_graph()
    nodes: list[object] = [r.node for r in records if r.scope is GraphScope.NODE and r.node is not None]
    edges: list[object] = [r.edge for r in records if r.scope is GraphScope.EDGE and r.edge is not None]
    return nodes, edges


@pytest.mark.parametrize(
    "computer",
    [DegreeComponentGraphFeatureComputer(), PageRankLabelPropGraphFeatureComputer()],
)
def test_providers_cover_every_node_with_finite_centrality(computer: object) -> None:
    nodes, edges = _assembled_nodes_edges()
    features = computer.compute_features(nodes, edges)  # type: ignore[attr-defined]
    node_ids = {n.node_id for n in nodes}  # type: ignore[attr-defined]
    assert set(features.centrality) == node_ids
    assert set(features.community) == node_ids
    assert all(isinstance(v, float) for v in features.centrality.values())
    assert all(isinstance(c, str) and c for c in features.community.values())


def test_providers_empty_graph_yields_empty_features() -> None:
    features = DegreeComponentGraphFeatureComputer().compute_features([], [])
    assert features.centrality == {}
    assert features.community == {}


def test_graphfeatures_rejects_a_partial_map() -> None:
    with pytest.raises(ValueError, match="same node-id set"):
        GraphFeatures(centrality={"a": 0.5}, community={})


def test_graphfeatures_rejects_a_non_finite_centrality() -> None:
    with pytest.raises(ValueError, match="finite"):
        GraphFeatures(centrality={"a": float("inf")}, community={"a": "a"})


def test_graphfeatures_rejects_an_empty_community_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GraphFeatures(centrality={"a": 0.5}, community={"a": ""})


# ---------------------------------------------------------------------------
# Assembler wiring: stamp on config, byte-identical when off
# ---------------------------------------------------------------------------


def _node_records(records: Sequence[object]) -> list[object]:
    return [r for r in records if r.scope is GraphScope.NODE]  # type: ignore[attr-defined]


def test_assembler_stamps_centrality_and_community_on_every_node() -> None:
    assembler = CanonicalGraphAssembler({"graph_features": _DEGREE})
    records = list(assembler.assemble(fixtures.graph_records()))
    node_records = _node_records(records)
    assert node_records  # the canonical corpus has entities
    for record in node_records:
        props = record.node.properties  # type: ignore[attr-defined]
        assert "centrality" in props and isinstance(props["centrality"], float)
        assert "community" in props and isinstance(props["community"], str)


def test_assembler_without_features_is_byte_identical() -> None:
    plain = CanonicalGraphAssembler({"corpus_id": "c"})
    enriched_off = CanonicalGraphAssembler({"corpus_id": "c", "graph_features": None})
    records = fixtures.graph_records()
    a = [r.model_dump_json() for r in plain.assemble(records)]
    b = [r.model_dump_json() for r in enriched_off.assemble(records)]
    assert a == b
    # And no node carries a centrality property (the enrichment truly did nothing).
    assert all(
        "centrality" not in r.node.properties  # type: ignore[attr-defined]
        for r in _node_records(list(plain.assemble(records)))
    )


def test_weighted_assembler_also_stamps_features() -> None:
    assembler = WeightedGraphAssembler({"graph_features": _PAGERANK})
    node_records = _node_records(list(assembler.assemble(fixtures.graph_records())))
    assert node_records
    assert all(
        "centrality" in r.node.properties and "community" in r.node.properties  # type: ignore[attr-defined]
        for r in node_records
    )


def test_assembler_is_deterministic_with_features() -> None:
    records = fixtures.graph_records()
    a = [
        r.model_dump_json()
        for r in CanonicalGraphAssembler({"graph_features": _PAGERANK}).assemble(records)
    ]
    b = [
        r.model_dump_json()
        for r in CanonicalGraphAssembler({"graph_features": _PAGERANK}).assemble(records)
    ]
    assert a == b


def test_assembler_rejects_a_non_graphfeaturecomputer_provider() -> None:
    # A mis-named provider (an Embedder, not a GraphFeatureComputer) fails LOUDLY at wiring time.
    bad = {"provider": "embedding.hashing", "config": {}}
    with pytest.raises(ContractError, match="GraphFeatureComputer"):
        CanonicalGraphAssembler({"graph_features": bad})


def test_assembler_rejects_a_malformed_graph_features_spec() -> None:
    with pytest.raises(ConfigError, match="graph_features"):
        CanonicalGraphAssembler({"graph_features": "not-a-mapping"})


# ---------------------------------------------------------------------------
# KG-export columns: promoted when present, absent when off
# ---------------------------------------------------------------------------


def _export_nodes_table(records: Sequence[object], tmp_path: Path) -> object:
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    KnowledgeGraphExport().export(records, storage, f"{store}/kg")  # type: ignore[arg-type]
    with storage.open(f"{store}/kg/graph-nodes.parquet", "rb") as fh:
        return pq.read_table(fh)  # type: ignore[no-untyped-call]


def test_kg_export_promotes_feature_columns_when_present(tmp_path: Path) -> None:
    records = list(CanonicalGraphAssembler({"graph_features": _DEGREE}).assemble(
        fixtures.graph_records()
    ))
    table = _export_nodes_table(records, tmp_path)
    cols = set(table.column_names)  # type: ignore[attr-defined]
    assert {"centrality", "community"} <= cols
    # Every node row carries a real value (not null) — the assembler stamped all nodes.
    centralities = table.column("centrality").to_pylist()  # type: ignore[attr-defined]
    communities = table.column("community").to_pylist()  # type: ignore[attr-defined]
    assert all(c is not None for c in centralities)
    assert all(c for c in communities)


def test_kg_export_omits_feature_columns_when_off(tmp_path: Path) -> None:
    records = list(CanonicalGraphAssembler({}).assemble(fixtures.graph_records()))
    table = _export_nodes_table(records, tmp_path)
    cols = set(table.column_names)  # type: ignore[attr-defined]
    assert "centrality" not in cols
    assert "community" not in cols


def test_kg_export_bytes_identical_when_off(tmp_path: Path) -> None:
    records = list(CanonicalGraphAssembler({}).assemble(fixtures.graph_records()))
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    KnowledgeGraphExport().export(records, storage, f"{store}/a")
    KnowledgeGraphExport().export(records, storage, f"{store}/b")
    a = storage.read_bytes(f"{store}/a/graph-nodes.parquet")
    b = storage.read_bytes(f"{store}/b/graph-nodes.parquet")
    assert a == b


# ---------------------------------------------------------------------------
# Context-header projection: aligned when enriched, empty when off
# ---------------------------------------------------------------------------


def _enrich(graph_records: Sequence[object]) -> list[ChunkRecord]:
    corpus = fixtures.canonical_corpus()
    enricher = ContextHeaderEnricher({})
    return [
        c
        for c in enricher.enrich(corpus.chunks, corpus.mentions, graph_records)  # type: ignore[arg-type]
        if isinstance(c, ChunkRecord)
    ]


def _headers(chunks: list[ChunkRecord]) -> list[object]:
    return [c.context_header for c in chunks if c.context_header is not None]


def test_context_header_carries_aligned_features_when_enriched() -> None:
    enriched_graph = list(
        CanonicalGraphAssembler({"graph_features": _DEGREE}).assemble(fixtures.graph_records())
    )
    headers = _headers(_enrich(enriched_graph))
    assert headers  # the canonical corpus has entity-bearing chunks
    projected = [h for h in headers if h.kg_node_ids]  # type: ignore[attr-defined]
    assert projected
    for header in projected:
        assert len(header.kg_node_centrality) == len(header.kg_node_ids)  # type: ignore[attr-defined]
        assert len(header.kg_node_community) == len(header.kg_node_ids)  # type: ignore[attr-defined]
        assert all(isinstance(v, float) for v in header.kg_node_centrality)  # type: ignore[attr-defined]
        assert all(isinstance(c, str) and c for c in header.kg_node_community)  # type: ignore[attr-defined]


def test_context_header_features_empty_when_graph_not_enriched() -> None:
    plain_graph = list(CanonicalGraphAssembler({}).assemble(fixtures.graph_records()))
    headers = _headers(_enrich(plain_graph))
    assert headers  # entities still projected — only the features are absent
    assert all(not h.kg_node_centrality for h in headers)  # type: ignore[attr-defined]
    assert all(not h.kg_node_community for h in headers)  # type: ignore[attr-defined]
