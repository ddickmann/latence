# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The harness-supplied engines — contract-checked, and equivalence-checked against the shipped ones.

Two of the four adapters here exist only because a shipped adapter's *packaging* does not fit
this environment (``duckdb`` absent) or its per-call re-read would dominate the latency the
harness is measuring. Those two are the ones that could silently diverge, so they are tested
against the real thing on the real export rather than against a fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmark.s5.corpus import CONTENT_ONLY_VECTORS, ChunkRow, Export, load_export
from benchmark.s5.datasets import DATASETS
from benchmark.s5.legs import (
    ABLATION_LEG_NAMES,
    ALL_LEG_NAMES,
    FUSION_SIGNALS,
    LEG_NAMES,
    Bm25Backend,
    ChunkGraphDocumentSource,
    DenseBackend,
    EdgeGraphSource,
    build_runner,
    leg_signals,
    missing_inputs,
    query_ngrams,
    tunable_signals,
    vector_source,
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


# -- the context-header ablation leg -----------------------------------------------------------

#: Two deliberately DIFFERENT vector sources over the same three chunks: the query ``[1, 0, 0]``
#: ranks ``c0`` first under the corpus matrix and ``c2`` first under the sidecar one, so a leg
#: that read the wrong source could not accidentally produce the right answer.
_CORPUS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_SIDECAR_MATRIX = _CORPUS_MATRIX[::-1].copy()
_QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _tiny_export(root: Path) -> Export:
    rows = [ChunkRow(f"c{i}", f"chunk {i}", "d.md", i * 10, i * 10 + 9, ()) for i in range(3)]
    return Export(dataset="t", root=root, rows=rows, embeddings=_CORPUS_MATRIX, pages={})


def test_every_leg_declares_the_signals_it_fuses_and_which_of_them_are_tunable() -> None:
    """The weight vector's shape is a property of the leg, declared once, not rederived per call."""
    assert leg_signals("dense") == ("dense",)
    assert leg_signals("coe") == FUSION_SIGNALS
    # The packer does not add a list — it selects over the fused one — so it fuses what coe fuses.
    assert leg_signals("packer") == leg_signals("coe")
    assert leg_signals("dense_content_only") == ("dense_content_only",)
    # Cumulative along the ladder: each rung's signals are the rung below's plus exactly one.
    for lower, upper in zip(LEG_NAMES, LEG_NAMES[1:], strict=False):
        assert leg_signals(lower) == leg_signals(upper)[: len(leg_signals(lower))]
    # `dense` is pinned (RRF weights are scale-invariant), so it is never in the search space.
    for leg in ALL_LEG_NAMES:
        assert tunable_signals(leg) == leg_signals(leg)[1:]
        assert leg_signals(leg)[0] not in tunable_signals(leg)


def test_the_ablation_leg_is_beside_the_ladder_not_on_it() -> None:
    """The six specced legs stay exactly the six specced legs; the ablation is appended."""
    assert LEG_NAMES == ("dense", "bm25", "kg", "hyperedges", "coe", "packer")
    assert ABLATION_LEG_NAMES == ("dense_content_only",)
    assert ALL_LEG_NAMES == LEG_NAMES + ABLATION_LEG_NAMES
    assert not set(LEG_NAMES) & set(ABLATION_LEG_NAMES)
    assert [vector_source(leg) for leg in LEG_NAMES] == ["corpus"] * len(LEG_NAMES)
    assert vector_source("dense_content_only") == "sidecar"


def test_each_leg_ranks_with_its_own_vector_source(tmp_path: Path) -> None:
    """`dense` reads the export's enriched vectors; `dense_content_only` reads the sidecar's."""
    export = _tiny_export(tmp_path)
    runner = build_runner(
        export, ("dense", "dense_content_only"), content_only_vectors=_SIDECAR_MATRIX
    )
    assert runner.dense is not None
    assert runner.dense_content_only is not None
    assert runner.dense_content_only.source == CONTENT_ONLY_VECTORS
    assert runner.run("q", _QUERY, "dense").ranked[0] == "c0"
    assert runner.run("q", _QUERY, "dense_content_only").ranked[0] == "c2"
    # The ablation's single list passes through the fusion unreordered — RRF over one list is
    # `w / (k + rank)`, strictly decreasing in rank — so the leg stays subtractable against
    # `dense` no matter what weight it is given.
    assert runner.dense_content_only is not None
    hits = runner.dense_content_only.search(
        RetrievalQuery(mode=Capability.DENSE, vector=tuple(float(x) for x in _QUERY), top_k=100)
    )
    assert runner.run("q", _QUERY, "dense_content_only").ranked == [c.id for c in hits]


def test_the_ablation_leg_returns_nothing_rather_than_the_vectors_it_contrasts_with(
    tmp_path: Path,
) -> None:
    """No sidecar means no number — never a silent fallback to the enriched matrix."""
    runner = build_runner(_tiny_export(tmp_path), ("dense", "dense_content_only"))
    assert runner.dense_content_only is None
    assert runner.run("q", _QUERY, "dense_content_only").ranked == []
    assert runner.run("q", _QUERY, "dense").ranked[0] == "c0"


def test_dense_backend_refuses_a_matrix_that_does_not_cover_the_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing to rank a misaligned matrix"):
        DenseBackend(_tiny_export(tmp_path), matrix=_SIDECAR_MATRIX[:2], source="sidecar")


def test_missing_inputs_refuses_the_ablation_leg_by_its_own_file(tmp_path: Path) -> None:
    """Not cumulative: the ablation needs the corpus table and its sidecar, nothing above."""
    assert missing_inputs(tmp_path, "dense_content_only") == [
        "corpus.parquet",
        CONTENT_ONLY_VECTORS,
    ]
    (tmp_path / "corpus.parquet").touch()
    assert missing_inputs(tmp_path, "dense_content_only") == [CONTENT_ONLY_VECTORS]
    (tmp_path / CONTENT_ONLY_VECTORS).touch()
    assert missing_inputs(tmp_path, "dense_content_only") == []
    # …and it never demands the BM25/graph artifacts the ladder legs need.
    assert "bm25-stats.json" in missing_inputs(tmp_path, "bm25")


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
