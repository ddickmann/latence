"""The graph **retrieval leg** (T-RET-2): the KG must be able to RETRIEVE, not only reorder.

The defect this file exists to close, stated as the audit found it: ``MultiHopExpander`` and
``Expander`` are :class:`~latence_retrieval.processor.CandidateProcessor`\\ s applied *after* fusion
(``orchestrator.py:266-269``). They add a hop-decayed bonus to already-fused scores and **cannot
introduce a candidate the first stage missed** (``multihop.py:621-628``) — so the claim "graph
traversal recovers documents dense+sparse never retrieved" was not merely unproven, the architecture
could not *express* it.

The headline test is :func:`test_graph_leg_recovers_a_document_dense_and_sparse_never_retrieved`.
Its control, :func:`test_post_fusion_expander_cannot_recover_the_missed_document`, drives the
*existing* post-fusion expander over the *same* fixture and pins that it cannot — that is the
structural claim, kept as a characterization test so the expander is never mistaken for evidence of
recovery again (the ticket keeps post-fusion rescoring as a legitimate second use).

The whole file drives the public seam — a :class:`~latence_retrieval.orchestrator.QueryBundle` and a
backend go in, a :class:`~latence_retrieval.orchestrator.RetrievalResult` comes out — never the
retriever's internals.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from latence_retrieval import (
    AliasQuerySeeder,
    Candidate,
    Capability,
    CompositeBackend,
    DuckDBGraphDocumentSource,
    GraphDocument,
    GraphDocumentSource,
    GraphRetriever,
    OrchestrationError,
    OrchestrationPipeline,
    QueryBundle,
    RetrievalBackend,
    RetrievalQuery,
    RrfFuser,
    build_graph_document_source,
    retrieve,
)
from latence_retrieval.errors import BackendCapabilityError, GraphSourceError
from latence_retrieval.graph_retrieval import (
    _DOCUMENT_TABLE_CACHE,
    clear_document_source_cache,
)
from latence_retrieval.multihop import MultiHopExpander
from latence_retrieval.testing import (
    FakeBackend,
    FakeGraphDocumentSource,
    FakeGraphSource,
)

# --- the fixture the whole architectural claim rests on ---------------------------------------
#
# A three-hop chain of entities:  n_query -- n_bridge -- n_gold
#
# The vector store holds five chunks. The gold chunk scores 0.05 — dead last — so with a
# first-stage depth of 3 neither the dense nor the sparse leg can ever return it. Only a traversal
# from the query's own entity reaches it, at hop 2.

_EDGES: list[tuple[str, str]] = [("n_query", "n_bridge"), ("n_bridge", "n_gold")]

_STORE_CORPUS = [
    Candidate(id="d1", score=0.90, text="alpha", metadata={"kg_node_ids": ["n_query"]}),
    Candidate(id="d2", score=0.80, text="beta", metadata={"kg_node_ids": ["n_bridge"]}),
    Candidate(id="d3", score=0.70, text="gamma", metadata={"kg_node_ids": []}),
    Candidate(id="d4", score=0.60, text="delta", metadata={"kg_node_ids": []}),
    Candidate(id="gold", score=0.05, text="epsilon", metadata={"kg_node_ids": ["n_gold"]}),
]

# The node→chunk index the graph leg resolves through: only chunks that carry KG nodes appear.
_GRAPH_DOCS = [
    GraphDocument(id="d1", node_ids=("n_query",), text="alpha"),
    GraphDocument(id="d2", node_ids=("n_bridge",), text="beta"),
    GraphDocument(id="gold", node_ids=("n_gold",), text="epsilon"),
]

_FETCH_TOP_K = 3


def _store(**over: Any) -> FakeBackend:
    """The customer's vector store: dense + sparse, no idea the graph exists."""
    caps = over.pop("capabilities", {Capability.DENSE, Capability.SPARSE})
    return FakeBackend(_STORE_CORPUS, capabilities=caps, dimension=3, **over)


def _graph_leg(**config: Any) -> GraphRetriever:
    return GraphRetriever(
        {"max_hops": 2, **config},
        graph_source=FakeGraphSource(_EDGES),
        document_source=FakeGraphDocumentSource(_GRAPH_DOCS),
    )


def _bundle(**over: Any) -> QueryBundle:
    base: dict[str, Any] = {
        "text": "alpha",
        "vector": (0.1, 0.2, 0.3),
        "sparse_terms": {"alpha": 1.0},
        "seed_node_ids": ("n_query",),
    }
    base.update(over)
    return QueryBundle(**base)


def _ids(result: Any) -> list[str]:
    return [c.id for c in result.candidates]


# --- AC1: the acceptance test that was impossible to pass -------------------------------------


def test_dense_and_sparse_alone_never_return_the_gold_document() -> None:
    """The premise: without a graph leg the gold chunk is below the first-stage horizon."""
    result = retrieve(
        _bundle(),
        _store(),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert result.fetched_modes == (Capability.DENSE, Capability.SPARSE)
    assert "gold" not in _ids(result)


def test_graph_leg_recovers_a_document_dense_and_sparse_never_retrieved() -> None:
    """THE point of T-RET-2: traversal introduces a candidate the first stage missed."""
    backend = CompositeBackend([_store(), _graph_leg()])
    result = retrieve(
        _bundle(),
        backend,
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert Capability.GRAPH in result.fetched_modes
    assert "gold" in _ids(result)


def test_post_fusion_expander_cannot_recover_the_missed_document() -> None:
    """The structural control: the *existing* post-fusion expander cannot do this, by construction.

    Same fixture, same graph, same seeds — but the expander is a ``CandidateProcessor`` that runs
    over the already-fused pool, so it can only reorder what dense+sparse already returned.
    """
    expander = MultiHopExpander(
        {"max_hops": 2, "seed_top_k": 3},
        graph_source=FakeGraphSource(_EDGES),
    )
    result = retrieve(
        _bundle(),
        _store(),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K, processors=(expander,)),
    )
    assert "gold" not in _ids(result)


def test_the_graph_leg_and_the_post_fusion_expander_compose() -> None:
    """Keeping the expander is legitimate: it rescores the pool the graph leg helped build."""
    expander = MultiHopExpander(
        {"max_hops": 2, "seed_top_k": 3},
        graph_source=FakeGraphSource(_EDGES),
    )
    result = retrieve(
        _bundle(),
        CompositeBackend([_store(), _graph_leg()]),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K, processors=(expander,)),
    )
    assert "gold" in _ids(result)


# --- AC3: the D7 ablation is runnable ----------------------------------------------------------


def test_the_graph_ablation_is_runnable_from_one_switch() -> None:
    """With and without the graph leg, same bundle, same store — the D7 ablation."""
    without = retrieve(_bundle(), _store(), OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K))
    with_graph = retrieve(
        _bundle(),
        CompositeBackend([_store(), _graph_leg()]),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert Capability.GRAPH not in without.fetched_modes
    assert Capability.GRAPH in with_graph.fetched_modes
    assert set(_ids(with_graph)) - set(_ids(without)) == {"gold"}


# --- AC2: Capability.GRAPH is declared and the planner honours it ------------------------------


def test_graph_is_a_declared_first_stage_capability() -> None:
    assert Capability.GRAPH.value == "graph"
    assert Capability.GRAPH in set(Capability)


def test_a_graph_query_requires_seed_node_ids() -> None:
    with pytest.raises(ValueError, match="seed_node_ids"):
        RetrievalQuery(mode=Capability.GRAPH, text="alpha")


def test_graph_query_accepts_seed_node_ids() -> None:
    query = RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",))
    assert query.seed_node_ids == ("n_query",)


def test_the_graph_leg_is_dropped_when_the_backend_cannot_serve_it() -> None:
    """Capability-adaptive composition (AC2): a seeded bundle against a graph-less store."""
    result = retrieve(_bundle(), _store(), OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K))
    assert Capability.GRAPH not in result.fetched_modes


def test_the_graph_leg_is_dropped_when_the_bundle_carries_no_seeds() -> None:
    result = retrieve(
        _bundle(seed_node_ids=None),
        CompositeBackend([_store(), _graph_leg()]),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert Capability.GRAPH not in result.fetched_modes
    assert "gold" not in _ids(result)


def test_a_graph_only_bundle_against_a_graph_less_store_refuses() -> None:
    """No runnable leg is a loud planning failure, not a silent empty result."""
    with pytest.raises(OrchestrationError, match="no runnable fetch leg"):
        retrieve(
            QueryBundle(text="alpha", seed_node_ids=("n_query",)),
            _store(),
            OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
        )


def test_a_graph_only_bundle_runs_the_graph_leg_alone() -> None:
    result = retrieve(
        QueryBundle(text="alpha", seed_node_ids=("n_query",)),
        _graph_leg(),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert result.fetched_modes == (Capability.GRAPH,)
    assert _ids(result) == ["d1", "d2", "gold"]


def test_a_filter_the_composite_cannot_serve_is_refused_before_any_fetch() -> None:
    """The refuse-if-unserveable path (orchestrator.py:291-297) holds for a composite too."""
    backend = CompositeBackend([_store(), _graph_leg()])
    assert Capability.FILTER not in backend.capabilities
    with pytest.raises(OrchestrationError, match="filter"):
        retrieve(
            _bundle(filter={"lang": "de"}),
            backend,
            OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
        )


# --- the GraphRetriever itself: a faithful RetrievalBackend ------------------------------------


def test_graph_retriever_satisfies_the_retrieval_backend_seam() -> None:
    assert isinstance(_graph_leg(), RetrievalBackend)
    assert _graph_leg().capabilities == frozenset({Capability.GRAPH, Capability.FILTER})


def test_graph_retriever_scores_by_hop_decay_and_orders_deterministically() -> None:
    hits = _graph_leg(hop_decay=0.5).search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=10)
    )
    assert [c.id for c in hits] == ["d1", "d2", "gold"]
    assert [c.score for c in hits] == [1.0, 0.5, 0.25]


def test_graph_retriever_honours_the_hop_bound() -> None:
    hits = _graph_leg(max_hops=1).search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=10)
    )
    assert [c.id for c in hits] == ["d1", "d2"]


def test_graph_retriever_honours_top_k() -> None:
    hits = _graph_leg().search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=2)
    )
    assert [c.id for c in hits] == ["d1", "d2"]


def test_graph_retriever_rejects_a_mode_it_does_not_declare() -> None:
    query = RetrievalQuery(mode=Capability.DENSE, text="alpha", vector=(0.1, 0.2, 0.3))
    with pytest.raises(BackendCapabilityError, match="graph"):
        _graph_leg().search(query)


def test_graph_retriever_applies_a_metadata_filter_before_truncation() -> None:
    source = FakeGraphDocumentSource(
        [
            GraphDocument(id="d1", node_ids=("n_query",), metadata={"lang": "en"}),
            GraphDocument(id="d2", node_ids=("n_bridge",), metadata={"lang": "de"}),
            GraphDocument(id="gold", node_ids=("n_gold",), metadata={"lang": "de"}),
        ]
    )
    leg = GraphRetriever(
        {"max_hops": 2},
        graph_source=FakeGraphSource(_EDGES),
        document_source=source,
    )
    hits = leg.search(
        RetrievalQuery(
            mode=Capability.GRAPH,
            text="alpha",
            seed_node_ids=("n_query",),
            top_k=1,
            filter={"lang": "de"},
        )
    )
    # Without pre-truncation filtering the top-1 would be d1 (hop 0) and this would be empty.
    assert [c.id for c in hits] == ["d2"]


def test_graph_retriever_is_stateless_across_calls() -> None:
    leg = _graph_leg()
    query = RetrievalQuery(
        mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=10
    )
    first = leg.search(query)
    second = leg.search(query)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_graph_retriever_returns_nothing_for_seeds_absent_from_the_graph() -> None:
    hits = _graph_leg().search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("nowhere",), top_k=10)
    )
    assert hits == []


def test_graph_retriever_requires_a_graph_and_a_document_source() -> None:
    with pytest.raises(ValueError, match="graph"):
        GraphRetriever({}, document_source=FakeGraphDocumentSource(_GRAPH_DOCS)).search(
            RetrievalQuery(mode=Capability.GRAPH, text="a", seed_node_ids=("n_query",))
        )
    with pytest.raises(ValueError, match="document"):
        GraphRetriever({}, graph_source=FakeGraphSource(_EDGES)).search(
            RetrievalQuery(mode=Capability.GRAPH, text="a", seed_node_ids=("n_query",))
        )


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_graph_retriever_rejects_a_bad_hop_bound(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_hops"):
        GraphRetriever({"max_hops": bad})


@pytest.mark.parametrize("bad", [0, -0.5, 1.5, "0.5", True])
def test_graph_retriever_rejects_a_bad_hop_decay(bad: Any) -> None:
    with pytest.raises(ValueError, match="hop_decay"):
        GraphRetriever({"hop_decay": bad})


def test_graph_retriever_rejects_unknown_config_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        GraphRetriever({"nonsense": 1})


# --- the FakeGraphDocumentSource is a faithful second adapter ----------------------------------


def test_fake_document_source_satisfies_the_seam() -> None:
    assert isinstance(FakeGraphDocumentSource(_GRAPH_DOCS), GraphDocumentSource)


def test_fake_document_source_returns_nothing_for_no_nodes() -> None:
    assert FakeGraphDocumentSource(_GRAPH_DOCS).documents([]) == []


def test_fake_document_source_returns_each_document_once() -> None:
    docs = FakeGraphDocumentSource(
        [GraphDocument(id="d1", node_ids=("a", "b"))]
    ).documents(["a", "b"])
    assert [d.id for d in docs] == ["d1"]


def test_fake_document_source_is_deterministically_ordered() -> None:
    source = FakeGraphDocumentSource(_GRAPH_DOCS)
    assert [d.id for d in source.documents(["n_gold", "n_query"])] == ["d1", "gold"]


# --- CompositeBackend -------------------------------------------------------------------------


def test_composite_unions_the_query_modes() -> None:
    backend = CompositeBackend([_store(), _graph_leg()])
    assert backend.capabilities == frozenset(
        {Capability.DENSE, Capability.SPARSE, Capability.GRAPH}
    )


def test_composite_declares_filter_only_when_every_member_can_serve_it() -> None:
    filterable_store = _store(
        capabilities={Capability.DENSE, Capability.SPARSE, Capability.FILTER}
    )
    assert Capability.FILTER in CompositeBackend([filterable_store, _graph_leg()]).capabilities
    assert Capability.FILTER not in CompositeBackend([_store(), _graph_leg()]).capabilities


def test_composite_routes_each_mode_to_the_member_that_declares_it() -> None:
    backend = CompositeBackend([_store(), _graph_leg()])
    graph_hits = backend.search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=10)
    )
    assert [c.id for c in graph_hits] == ["d1", "d2", "gold"]
    dense_hits = backend.search(
        RetrievalQuery(mode=Capability.DENSE, text="alpha", vector=(0.1, 0.2, 0.3), top_k=2)
    )
    assert [c.id for c in dense_hits] == ["d1", "d2"]


def test_composite_rejects_a_mode_no_member_declares() -> None:
    backend = CompositeBackend([_graph_leg()])
    with pytest.raises(BackendCapabilityError, match="bm25"):
        backend.search(RetrievalQuery(mode=Capability.BM25, text="alpha"))


def test_composite_requires_at_least_one_member() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeBackend([])


def test_composite_rejects_a_member_that_is_not_a_backend() -> None:
    with pytest.raises(TypeError, match="RetrievalBackend"):
        CompositeBackend([object()])  # type: ignore[list-item]


def test_composite_satisfies_the_retrieval_backend_seam() -> None:
    assert isinstance(CompositeBackend([_store(), _graph_leg()]), RetrievalBackend)


# --- weights reach the graph leg (T-FUSE-1 hand-off) -------------------------------------------


def test_the_graph_leg_carries_a_tunable_fusion_weight() -> None:
    """The new leg is a fusion peer, so it gets its own weight like every other signal."""
    pipeline = OrchestrationPipeline(
        fuser=RrfFuser(),
        fetch_top_k=_FETCH_TOP_K,
        fuse_weights={"dense": 1.0, "sparse": 1.0, "graph": 50.0},
    )
    result = retrieve(_bundle(), CompositeBackend([_store(), _graph_leg()]), pipeline)
    # With the graph leg dominating, its own hop-0 hit leads and the recovered gold outranks the
    # store-only distractor d3, which no traversal reaches.
    assert _ids(result).index("gold") < _ids(result).index("d3")


# --- AliasQuerySeeder: the dependency-free reference for "seed from query entities" ------------


def test_alias_seeder_finds_a_single_token_entity() -> None:
    seeder = AliasQuerySeeder({"Kubernetes": "n_k8s"})
    assert seeder.seeds("how does kubernetes schedule pods") == ("n_k8s",)


def test_alias_seeder_prefers_the_longest_match() -> None:
    seeder = AliasQuerySeeder({"European Union": "n_eu", "Union": "n_union"})
    assert seeder.seeds("the European Union budget") == ("n_eu",)


def test_alias_seeder_consumes_a_matched_span() -> None:
    """One mention seeds one node: the tokens of a matched alias are not re-scanned."""
    seeder = AliasQuerySeeder({"New York": "n_ny", "York": "n_york"})
    assert seeder.seeds("New York") == ("n_ny",)


def test_alias_seeder_deduplicates_repeated_mentions() -> None:
    seeder = AliasQuerySeeder({"Berlin": "n_berlin"})
    assert seeder.seeds("Berlin, Berlin, Berlin") == ("n_berlin",)


def test_alias_seeder_preserves_first_occurrence_order() -> None:
    seeder = AliasQuerySeeder({"beta": "n_b", "alpha": "n_a"})
    assert seeder.seeds("beta then alpha") == ("n_b", "n_a")


def test_alias_seeder_returns_nothing_when_no_alias_matches() -> None:
    assert AliasQuerySeeder({"Berlin": "n_berlin"}).seeds("nothing here") == ()


def test_alias_seeder_is_script_agnostic() -> None:
    """The tokenizer is Unicode-alphanumeric, so a non-Latin alias is not silently unseedable."""
    seeder = AliasQuerySeeder({"Zürich": "n_zh", "東京": "n_tokyo"})
    assert seeder.seeds("von zürich nach 東京") == ("n_zh", "n_tokyo")


def test_alias_seeder_feeds_the_graph_leg_end_to_end() -> None:
    seeder = AliasQuerySeeder({"alpha document": "n_query"})
    bundle = QueryBundle(text="the alpha document", seed_node_ids=seeder.seeds("the alpha document"))
    result = retrieve(bundle, _graph_leg(), OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K))
    assert _ids(result) == ["d1", "d2", "gold"]


def test_alias_seeder_rejects_a_non_mapping_alias_table() -> None:
    with pytest.raises(TypeError, match="aliases"):
        AliasQuerySeeder(["Berlin"])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_alias_seeder_rejects_a_bad_max_span(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_span"):
        AliasQuerySeeder({"a": "n"}, max_span=bad)


def test_alias_seeder_rejects_non_string_text() -> None:
    with pytest.raises(TypeError, match="string"):
        AliasQuerySeeder({"a": "n"}).seeds(None)  # type: ignore[arg-type]


# --- pure-expansion mode: isolate the mass the traversal ADDED ---------------------------------


def test_include_seed_documents_false_returns_only_what_traversal_added() -> None:
    hits = _graph_leg(include_seed_documents=False).search(
        RetrievalQuery(mode=Capability.GRAPH, text="alpha", seed_node_ids=("n_query",), top_k=10)
    )
    assert [c.id for c in hits] == ["d2", "gold"]


def test_include_seed_documents_must_be_a_boolean() -> None:
    with pytest.raises(ValueError, match="include_seed_documents"):
        GraphRetriever({"include_seed_documents": "yes"})


# --- build_graph_document_source ----------------------------------------------------------------


def test_build_document_source_selects_duckdb_by_name() -> None:
    source = build_graph_document_source({"source": "duckdb", "chunks_uri": "/tmp/chunks.parquet"})
    assert isinstance(source, DuckDBGraphDocumentSource)


def test_build_document_source_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown graph document source"):
        build_graph_document_source({"source": "nonsense"})


def test_build_document_source_requires_a_chunks_uri() -> None:
    with pytest.raises(ValueError, match="chunks_uri"):
        build_graph_document_source({"source": "duckdb"})


def test_build_document_source_rejects_a_non_mapping_spec() -> None:
    with pytest.raises(TypeError, match="mapping"):
        build_graph_document_source(["duckdb"])  # type: ignore[arg-type]


# --- the REAL DuckDB document source, gated on the extra ---------------------------------------
#
# Anti-false-green: the fake is not an independent re-implementation the engine could drift from —
# these assert the real engine returns exactly what the fake returns on the same fixture table.

_HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None


def _write_chunks_parquet(path: str, documents: list[GraphDocument]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "chunk_id": [d.id for d in documents],
            "kg_node_ids": [list(d.node_ids) for d in documents],
            "content": [d.text for d in documents],
        }
    )
    pq.write_table(table, path)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_document_source_matches_the_faithful_fake(tmp_path: Any) -> None:
    path = str(tmp_path / "chunks.parquet")
    _write_chunks_parquet(path, _GRAPH_DOCS)
    real = DuckDBGraphDocumentSource(path)
    fake = FakeGraphDocumentSource(_GRAPH_DOCS)
    for nodes in (["n_query"], ["n_gold"], ["n_query", "n_gold"], ["absent"], []):
        assert [d.model_dump() for d in real.documents(nodes)] == [
            d.model_dump() for d in fake.documents(nodes)
        ]


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_document_source_drives_the_whole_graph_leg(tmp_path: Any) -> None:
    """The recovery claim, proven against the real embedded engine over real parquet."""
    path = str(tmp_path / "chunks.parquet")
    _write_chunks_parquet(path, _GRAPH_DOCS)
    leg = GraphRetriever(
        {"max_hops": 2},
        graph_source=FakeGraphSource(_EDGES),
        document_source=DuckDBGraphDocumentSource(path),
    )
    result = retrieve(
        _bundle(),
        CompositeBackend([_store(), leg]),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert "gold" in _ids(result)


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_real_duckdb_document_source_maps_a_fault_to_a_typed_error(tmp_path: Any) -> None:
    source = DuckDBGraphDocumentSource(str(tmp_path / "does-not-exist.parquet"))
    with pytest.raises(GraphSourceError) as excinfo:
        source.documents(["n_query"])
    # No raw engine text: no path, no driver stack.
    assert "does-not-exist" not in str(excinfo.value)


# --- ticket 01 (wayfinder-enterprise-serving): tiered resolution, byte-identical + early stop ---
#
# A wider fixture than the recovery chain: a seed fanning out to two hop-1 entities and two hop-2
# entities, with several documents per tier (some spanning tiers, some carrying nodes the
# traversal never reached). The ORACLE below ranks it the pre-ticket way — resolve the whole
# neighbourhood, score every document by its closest reached hop, sort by (-score, id), truncate —
# so every tiered assertion is checked against an independent reimplementation, not against the
# retriever's own internals.

_TIER_EDGES = [
    ("seed", "mid_a"),
    ("seed", "mid_b"),
    ("mid_a", "far_a"),
    ("mid_b", "far_b"),
]

_TIER_DOCS = [
    GraphDocument(id="doc_seed_1", node_ids=("seed",), text="s1"),
    GraphDocument(id="doc_seed_2", node_ids=("seed", "far_a"), text="s2"),  # min hop is 0
    GraphDocument(id="doc_mid_1", node_ids=("mid_a",), text="m1", metadata={"lang": "de"}),
    GraphDocument(id="doc_mid_2", node_ids=("mid_b", "far_b"), text="m2"),  # min hop is 1
    GraphDocument(id="doc_far_1", node_ids=("far_a",), text="f1"),
    GraphDocument(id="doc_far_2", node_ids=("far_b",), text="f2", metadata={"lang": "de"}),
    GraphDocument(id="doc_unreached", node_ids=("elsewhere",), text="x"),
]


class _CountingDocumentSource:
    """The faithful fake plus a call log — which node sets were actually resolved."""

    def __init__(self, documents: list[GraphDocument]) -> None:
        self._inner = FakeGraphDocumentSource(documents)
        self.calls: list[tuple[str, ...]] = []

    def documents(self, node_ids: Any) -> list[GraphDocument]:
        self.calls.append(tuple(sorted(node_ids)))
        return self._inner.documents(node_ids)


def _tier_leg(source: Any, **config: Any) -> GraphRetriever:
    return GraphRetriever(
        {"max_hops": 2, **config},
        graph_source=FakeGraphSource(_TIER_EDGES),
        document_source=source,
    )


def _oracle(
    top_k: int,
    *,
    hop_decay: float = 0.5,
    seed_score: float = 1.0,
    include_seeds: bool = True,
    predicate: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    """The pre-ticket ranking, reimplemented from the documented contract."""
    reachable = {"seed": 0, "mid_a": 1, "mid_b": 1, "far_a": 2, "far_b": 2}
    if not include_seeds:
        reachable = {n: h for n, h in reachable.items() if h > 0}
    hits: list[tuple[str, float]] = []
    for doc in _TIER_DOCS:
        hops = [reachable[n] for n in doc.node_ids if n in reachable]
        if not hops or (not include_seeds and min(hops) == 0):
            continue
        if predicate and any(doc.metadata.get(k) != v for k, v in predicate.items()):
            continue
        hits.append((doc.id, seed_score * hop_decay ** min(hops)))
    hits.sort(key=lambda pair: (-pair[1], pair[0]))
    return hits[:top_k]


def _search(leg: GraphRetriever, top_k: int, predicate: dict[str, Any] | None = None) -> Any:
    return leg.search(
        RetrievalQuery(
            mode=Capability.GRAPH,
            seed_node_ids=("seed",),
            top_k=top_k,
            filter=predicate or {},
        )
    )


@pytest.mark.parametrize("top_k", [1, 2, 3, 6, 50])
def test_tiered_search_is_byte_identical_to_the_exhaustive_ranking(top_k: int) -> None:
    """Ids AND scores match the oracle at every truncation depth — the parity claim."""
    hits = _search(_tier_leg(_CountingDocumentSource(_TIER_DOCS)), top_k)
    assert [(c.id, c.score) for c in hits] == _oracle(top_k)


def test_tiered_search_stops_resolving_once_top_k_is_ranked() -> None:
    """The latency claim: with top_k filled at hop 0, deeper tiers are never resolved."""
    source = _CountingDocumentSource(_TIER_DOCS)
    hits = _search(_tier_leg(source), 2)
    assert [(c.id, c.score) for c in hits] == _oracle(2)
    assert source.calls == [("seed",)], (
        "hop-1/hop-2 node sets must not be resolved when the hop-0 tier already fills top_k"
    )


def test_tiered_search_filters_before_truncation() -> None:
    """A predicate excluding the near tiers must surface deeper hits, never an empty list."""
    hits = _search(_tier_leg(_CountingDocumentSource(_TIER_DOCS)), 1, {"lang": "de"})
    assert [(c.id, c.score) for c in hits] == _oracle(1, predicate={"lang": "de"})
    assert hits[0].id == "doc_mid_1"


def test_tiered_search_honours_pure_expansion() -> None:
    hits = _tier_leg(
        _CountingDocumentSource(_TIER_DOCS), include_seed_documents=False
    ).search(RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=("seed",), top_k=10))
    assert [(c.id, c.score) for c in hits] == _oracle(10, include_seeds=False)
    # doc_seed_1 mentions ONLY the seed → gone; doc_seed_2 also carries a hop-2 entity, so as
    # pure expansion it is a legitimate hop-2 hit — exactly the pre-ticket semantics.
    ids = [c.id for c in hits]
    assert "doc_seed_1" not in ids
    assert "doc_seed_2" in ids


def test_unit_hop_decay_keeps_the_exhaustive_path_and_its_id_order() -> None:
    """``hop_decay=1.0`` ties every tier — ranked purely by id, exactly as before the ticket."""
    source = _CountingDocumentSource(_TIER_DOCS)
    hits = _tier_leg(source, hop_decay=1.0).search(
        RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=("seed",), top_k=3)
    )
    assert [(c.id, c.score) for c in hits] == _oracle(3, hop_decay=1.0)
    assert len(source.calls) == 1, "the degenerate config must resolve in one exhaustive call"


# --- the read cache behind the document source (arch pass Wave 1-A, FINDINGS F1) ---------------
#
# `cache=True` (the default) serves node→document resolution from a process-shared decoded table
# keyed by the chunk parquet's FILE IDENTITY; `cache=False` is the original per-call
# connect-scan-close path, preserved verbatim as the parity oracle. The bar is byte-identity —
# same documents, same ascending-id ORDER (the contract the tiered early-stop depends on) — and a
# rewritten parquet is re-decoded, never served stale.


def _rewrite_chunks_parquet(path: str, documents: list[GraphDocument]) -> None:
    """Rewrite the fixture parquet, forcing a distinct mtime_ns on coarse filesystem clocks."""
    import os

    _write_chunks_parquet(path, documents)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_document_source_is_byte_identical_to_the_uncached_oracle(tmp_path: Any) -> None:
    clear_document_source_cache()
    path = str(tmp_path / "chunks.parquet")
    # Written in NON-ascending id order, so the ascending-id contract is exercised, not inherited
    # from the fixture's layout.
    _write_chunks_parquet(path, list(reversed(_GRAPH_DOCS)))
    cached = DuckDBGraphDocumentSource(path)
    uncached = DuckDBGraphDocumentSource(path, cache=False)
    for nodes in (["n_query"], ["n_gold"], ["n_query", "n_gold"], ["n_gold", "n_query"],
                  ["absent"], [], ["n_query", "n_bridge", "n_gold", "absent"]):
        assert [d.model_dump() for d in cached.documents(nodes)] == [
            d.model_dump() for d in uncached.documents(nodes)
        ]
    ids = [d.id for d in cached.documents(["n_query", "n_bridge", "n_gold"])]
    assert ids == sorted(ids)  # ascending id — the tiered early-stop's ordering contract


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_document_source_shares_one_decode_and_the_oracle_touches_none(
    tmp_path: Any,
) -> None:
    clear_document_source_cache()
    path = str(tmp_path / "chunks.parquet")
    _write_chunks_parquet(path, _GRAPH_DOCS)
    DuckDBGraphDocumentSource(path).documents(["n_query"])
    DuckDBGraphDocumentSource(path).documents(["n_gold"])  # a second instance, same file
    assert len(_DOCUMENT_TABLE_CACHE) == 1  # one decoded table per file version, shared
    DuckDBGraphDocumentSource(path, cache=False).documents(["n_query"])
    assert len(_DOCUMENT_TABLE_CACHE) == 1  # the oracle path never populates the cache


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_document_source_reindexes_a_rewritten_parquet(tmp_path: Any) -> None:
    clear_document_source_cache()
    path = str(tmp_path / "chunks.parquet")
    _write_chunks_parquet(path, _GRAPH_DOCS)
    source = DuckDBGraphDocumentSource(path)
    assert [d.id for d in source.documents(["n_query"])] == ["d1"]
    replacement = [GraphDocument(id="fresh", node_ids=("n_query",), text="zeta")]
    _rewrite_chunks_parquet(path, replacement)
    assert [d.id for d in source.documents(["n_query"])] == ["fresh"]  # never stale
    assert len(_DOCUMENT_TABLE_CACHE) == 1  # the superseded version's table was evicted


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_document_source_missing_parquet_is_a_typed_error_with_no_leak(
    tmp_path: Any,
) -> None:
    clear_document_source_cache()
    source = DuckDBGraphDocumentSource(str(tmp_path / "does-not-exist.parquet"))
    with pytest.raises(GraphSourceError) as excinfo:
        source.documents(["n_query"])
    assert "does-not-exist" not in str(excinfo.value)  # the stat fault leaks no raw path either


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb extra not installed (offline)")
def test_cached_document_source_drives_the_whole_graph_leg(tmp_path: Any) -> None:
    """The recovery claim holds through the cached path too — same leg, same gold hit."""
    clear_document_source_cache()
    path = str(tmp_path / "chunks.parquet")
    _write_chunks_parquet(path, _GRAPH_DOCS)
    leg = GraphRetriever(
        {"max_hops": 2},
        graph_source=FakeGraphSource(_EDGES),
        document_source=DuckDBGraphDocumentSource(path),
    )
    result = retrieve(
        _bundle(),
        CompositeBackend([_store(), leg]),
        OrchestrationPipeline(fetch_top_k=_FETCH_TOP_K),
    )
    assert "gold" in _ids(result)


def test_document_source_factory_passes_cache_through_and_rejects_a_non_boolean() -> None:
    cached = build_graph_document_source({"source": "duckdb", "chunks_uri": "c.parquet"})
    uncached = build_graph_document_source(
        {"source": "duckdb", "chunks_uri": "c.parquet", "cache": False}
    )
    assert cached._cache is True  # type: ignore[union-attr]
    assert uncached._cache is False  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="cache"):
        build_graph_document_source(
            {"source": "duckdb", "chunks_uri": "c.parquet", "cache": "yes"}
        )
