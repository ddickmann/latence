# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The harness-supplied engines — contract-checked, and equivalence-checked against the shipped ones.

Two of the four adapters here exist only because a shipped adapter's *packaging* does not fit
this environment (``duckdb`` absent) or its per-call re-read would dominate the latency the
harness is measuring. Those two are the ones that could silently diverge, so they are tested
against the real thing on the real export rather than against a fixture.
"""

from __future__ import annotations

import pytest
from benchmark.s5.corpus import Export, load_export
from benchmark.s5.datasets import DATASETS
from benchmark.s5.legs import (
    LEG_NAMES,
    Bm25Backend,
    ChunkGraphDocumentSource,
    DenseBackend,
    EdgeGraphSource,
    missing_inputs,
    query_ngrams,
)
from latence_retrieval.contracts import Capability, RetrievalQuery

_VIDOSEEK = DATASETS["vidoseek"].export_dir

requires_export = pytest.mark.skipif(
    not (_VIDOSEEK / "corpus.parquet").is_file(),
    reason="the S3 exports are not on this machine",
)


@pytest.fixture(scope="module")
def export() -> Export:
    return load_export(_VIDOSEEK, "vidoseek")


def test_query_ngrams_match_the_m6_entry_surfaces() -> None:
    """Byte-identical to ``benchmark/musique-subset``'s ``question_ngrams`` (M6 tier-1 entry)."""
    grams = query_ngrams("Who directed Blade Runner?", max_n=2)
    assert grams[:3] == ["Who directed", "directed Blade", "Blade Runner"]
    assert "Runner" in grams


def test_missing_inputs_is_cumulative_and_refuses_rather_than_zero_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert missing_inputs(tmp_path, "dense") == ["corpus.parquet"]
    later = missing_inputs(tmp_path, "coe")
    assert later[0] == "corpus.parquet"
    assert "graph-hyperedges.kv" in later
    assert set(LEG_NAMES) == {"dense", "bm25", "kg", "hyperedges", "coe", "packer"}


@requires_export
def test_dense_backend_declares_only_dense_and_ranks_by_cosine(export: Export) -> None:
    backend = DenseBackend(export)
    assert backend.capabilities == frozenset({Capability.DENSE})
    assert export.embeddings is not None
    vector = tuple(float(x) for x in export.embeddings[0])
    hits = backend.search(RetrievalQuery(mode=Capability.DENSE, vector=vector, top_k=5))
    assert len(hits) == 5
    # A chunk's own exported vector must retrieve that chunk first — the sanity property the
    # whole dense leg rests on.
    assert hits[0].id == export.rows[0].record_id
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@requires_export
def test_bm25_backend_prefilter_does_not_change_the_ranking(export: Export) -> None:
    """The postings prefilter must be a cost optimisation only: excluded docs score exactly 0."""
    from latence_retrieval.bm25 import Bm25Rescorer
    from latence_retrieval.contracts import Candidate

    backend = Bm25Backend(export)
    assert backend.capabilities == frozenset({Capability.BM25})
    question = "energy management platform for multi-site estates"
    prefiltered = backend.search(RetrievalQuery(mode=Capability.BM25, text=question, top_k=10))

    full_pool = [
        Candidate(id=row.record_id, score=0.0, text=row.content) for row in export.rows
    ]
    exhaustive = Bm25Rescorer.from_export(str(export.root)).process(question, full_pool)[:10]
    assert [c.id for c in prefiltered] == [c.id for c in exhaustive]
    assert [round(c.score, 6) for c in prefiltered] == [round(c.score, 6) for c in exhaustive]


@requires_export
def test_edge_graph_source_matches_the_shipped_networkx_adapter(export: Export) -> None:
    from latence_retrieval.multihop import NetworkxGraphSource

    edges = export.root / "graph-edges.parquet"
    seeds = [row.node_ids[0] for row in export.rows[:20] if row.node_ids][:5]
    assert seeds
    mine = EdgeGraphSource(edges).neighbors(seeds, 2)
    theirs = NetworkxGraphSource(str(edges)).neighbors(seeds, 2)
    assert mine == theirs


@requires_export
def test_graph_document_source_returns_each_document_once_in_id_order(export: Export) -> None:
    source = ChunkGraphDocumentSource(export)
    assert source.documents([]) == []
    nodes = list(export.rows[0].node_ids[:3])
    documents = source.documents(nodes + nodes)
    ids = [d.id for d in documents]
    assert ids == sorted(set(ids))
    assert any(d.id == export.rows[0].record_id for d in documents)
