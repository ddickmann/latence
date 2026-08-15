# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The harness-supplied engines — contract-checked, and equivalence-checked against the shipped ones.

Most of the adapters here exist only because a shipped adapter's *packaging* does not fit this
environment (``duckdb`` absent), or because its per-call re-read / per-candidate Python loop
would dominate the latency the harness is measuring. Those are the ones that could silently
diverge, so each is tested against the real shipped component on the real export rather than
against a fixture: :class:`EdgeGraphSource` against ``NetworkxGraphSource``, and the vectorised
BM25 first stage against ``Bm25Rescorer``, score for score and rank for rank.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    RESCUE_LEG_NAMES,
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


def test_the_other_families_are_beside_the_ladder_not_on_it() -> None:
    """The six specced legs stay exactly the six specced legs; the other families are appended.

    Three disjoint families, in ladder-first order: the cumulative ladder, the input ablations, and
    the graph-as-rescue legs. Disjointness is the property that matters — a leg that appeared in two
    families would be both an increment and not an increment, and the report's `Δ` chain (which
    walks `LEG_NAMES` alone) would silently subtract it from something it is not a step above.
    """
    assert LEG_NAMES == ("dense", "bm25", "kg", "hyperedges", "coe", "packer")
    assert ABLATION_LEG_NAMES == ("dense_content_only",)
    assert RESCUE_LEG_NAMES == ("graph_rescue", "graph_rescue_packed", "graph_rescue_coe")
    assert ALL_LEG_NAMES == LEG_NAMES + ABLATION_LEG_NAMES + RESCUE_LEG_NAMES
    families = (set(LEG_NAMES), set(ABLATION_LEG_NAMES), set(RESCUE_LEG_NAMES))
    assert sum(len(family) for family in families) == len(set().union(*families))
    assert [vector_source(leg) for leg in LEG_NAMES] == ["corpus"] * len(LEG_NAMES)
    assert vector_source("dense_content_only") == "sidecar"
    # The rescue family ranks the export's own vectors, like the ladder — only the *application*
    # of the graph lane differs, never the corpus side it is measured on.
    assert [vector_source(leg) for leg in RESCUE_LEG_NAMES] == ["corpus"] * 3


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


# -- the vectorised BM25 first stage: same numbers as the shipped rescorer, ~10^3 times faster --
#
# `Bm25Backend` no longer hands a candidate pool to `Bm25Rescorer.process` — it scores the whole
# corpus with one sparse mat-vec over precomputed BM25 document-term weights. That is only
# legitimate if it is EXACTLY the rescorer's answer, so the rescorer stays in the tests as the
# oracle: `_assert_matches_the_rescorer` re-scores the full corpus with the shipped component and
# demands the same ids, in the same order, with the same scores.
#
# **Tolerance.** The per-`(term, document)` products are bit-identical by construction (same
# float64 formula, same operation order as `Bm25Rescorer._term_saturation`); only the order the
# handful of per-term products are SUMMED in differs (ascending term id vs Python `set` iteration
# order), which is a last-ulp effect. `1e-9` relative is therefore ~6 orders of magnitude looser
# than what is measured — over 270 real benchmark questions across three real exports (vidoseek
# 200/200, multihop_rag 50/50, wiki2multihop 20/20) the returned order was identical every time
# and the worst relative score deviation was 6.1e-16, ~2 ulps — and would still catch any real
# formula, statistics or tokenisation drift, all of which move scores by percent.
_BM25_RTOL = 1e-9


def _assert_matches_the_rescorer(
    export: Export, backend: Bm25Backend, questions: Sequence[str], top_k: int
) -> None:
    from latence_retrieval.bm25 import Bm25Rescorer
    from latence_retrieval.contracts import Candidate

    oracle = Bm25Rescorer.from_export(str(export.root))
    full_pool = [Candidate(id=row.record_id, score=0.0, text=row.content) for row in export.rows]
    for question in questions:
        mine = backend.search(RetrievalQuery(mode=Capability.BM25, text=question, top_k=top_k))
        # The rescorer scores the WHOLE corpus here — no prefilter — so this is the exhaustive
        # first stage the vectorised one claims to reproduce, not a cheaper approximation of it.
        theirs = [c for c in oracle.process(question, full_pool) if c.score > 0.0][:top_k]
        assert [c.id for c in mine] == [c.id for c in theirs], question
        for a, b in zip(mine, theirs, strict=True):
            assert a.score == pytest.approx(b.score, rel=_BM25_RTOL, abs=0.0), (question, a.id)
            assert a.text == b.text


def _write_export(root: Path, texts: dict[str, str]) -> Export:
    """A minimal S3 export: the two BM25 artifacts, written the way the Export stage writes them.

    The postings are derived by tokenising the very texts the rows carry (`doc_len` = token count,
    `terms`/`term_frequencies` = the per-document term histogram) and the statistics are the true
    global ones — which is exactly the agreement the real artifacts hold, verified over all 3,264
    rows of the `multihop_rag` export. So this fixture exercises the real code path, and the
    equivalence it asserts is not fixture-shaped.
    """
    import json
    import math
    import re
    from collections import Counter

    import pyarrow as pa
    import pyarrow.parquet as pq

    token = re.compile(r"[^\W_]+")
    tokenised = {rid: token.findall(text.lower()) for rid, text in texts.items()}
    histograms = {rid: Counter(tokens) for rid, tokens in tokenised.items()}
    doc_freq: Counter[str] = Counter()
    for histogram in histograms.values():
        doc_freq.update(histogram.keys())
    n = len(texts)
    avgdl = sum(len(t) for t in tokenised.values()) / n
    root.mkdir(parents=True, exist_ok=True)
    (root / "bm25-stats.json").write_text(
        json.dumps(
            {
                "doc_count": n,
                "avgdl": avgdl,
                "terms": [
                    {
                        "term": term,
                        "df": df,
                        "idf": math.log(1.0 + (n - df + 0.5) / (df + 0.5)),
                    }
                    for term, df in sorted(doc_freq.items())
                ],
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "record_id": list(texts),
                "doc_len": [len(tokenised[rid]) for rid in texts],
                "terms": [sorted(histograms[rid]) for rid in texts],
                "term_frequencies": [
                    [histograms[rid][term] for term in sorted(histograms[rid])] for rid in texts
                ],
            }
        ),
        root / "bm25-postings.parquet",
    )  # type: ignore[no-untyped-call]
    rows = [
        ChunkRow(rid, text, "d.md", 0, len(text), ()) for rid, text in texts.items()
    ]
    return Export(dataset="fixture", root=root, rows=rows, embeddings=None, pages={})


#: Deliberately adversarial for a top-k that must reproduce a `(-score, id)` sort exactly: `z-dup`
#: and `a-dup` hold IDENTICAL text (an exact score tie whose order is decided purely by the id
#: tiebreak, and whose ids sort against their row order), `long` repeats a query term but is
#: length-penalised, `common` shares only the corpus-wide term, and `none` shares nothing at all.
_FIXTURE_TEXTS = {
    "z-dup": "the quantum compiler lowers the circuit",
    "m-long": (
        "the quantum compiler lowers the circuit and then the quantum compiler lowers the "
        "circuit again while the quantum scheduler waits for the circuit to settle down"
    ),
    "a-dup": "the quantum compiler lowers the circuit",
    "b-common": "the harvest festival ran late",
    "c-none": "unrelated prose about gardening",
}


def test_bm25_first_stage_reproduces_the_shipped_rescorer_exactly(tmp_path: Path) -> None:
    """The whole justification for vectorising: the same ids, the same order, the same scores."""
    export = _write_export(tmp_path, _FIXTURE_TEXTS)
    backend = Bm25Backend(export)
    assert backend.capabilities == frozenset({Capability.BM25})
    questions = [
        "quantum compiler circuit",
        "the",  # the corpus-wide term: every document matches, so nothing is prefiltered away
        "circuit",
        "quantum tunnelling",  # one known term, one the corpus never emitted
        "gardening prose",
    ]
    for top_k in (1, 2, 3, 100):
        _assert_matches_the_rescorer(export, backend, questions, top_k)

    # The exact tie is real, and it is broken by ascending id — not by row order (`z-dup` is row 0).
    tied = backend.search(
        RetrievalQuery(mode=Capability.BM25, text="quantum compiler lowers", top_k=2)
    )
    assert [c.id for c in tied] == ["a-dup", "z-dup"]
    assert tied[0].score == tied[1].score

    # A query the corpus shares no term with returns NOTHING — never zero-scored padding.
    assert backend.search(RetrievalQuery(mode=Capability.BM25, text="zzz", top_k=10)) == []
    assert backend.search(RetrievalQuery(mode=Capability.BM25, text="!?", top_k=10)) == []
    # …and a query matching a single document returns that one document, not `top_k` of them.
    assert [
        c.id for c in backend.search(RetrievalQuery(mode=Capability.BM25, text="harvest", top_k=10))
    ] == ["b-common"]


def test_bm25_first_stage_uses_the_export_global_idf_not_a_pool_local_one(tmp_path: Path) -> None:
    """The IDF must come from `bm25-stats.json` — the corpus's, not the matching subset's."""
    export = _write_export(tmp_path, _FIXTURE_TEXTS)
    backend = Bm25Backend(export)
    matrix = backend.matrix
    assert matrix.n_docs == len(_FIXTURE_TEXTS)
    # `the` is in 4 of the 5 documents and `harvest` in 1: the global statistics separate them.
    # A pool-local IDF (the rescorer's no-stats mode) would compute both over the matching subset.
    n = len(_FIXTURE_TEXTS)
    for term, df in (("the", 4), ("harvest", 1), ("quantum", 3)):
        expected = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        assert matrix._idf[matrix._lookup[term]] == pytest.approx(expected, rel=1e-15)  # noqa: SLF001
    assert matrix.nnz == sum(len(set(re.findall(r"[^\W_]+", t.lower()))) for t in _FIXTURE_TEXTS.values())
    assert matrix.nbytes > 0


def test_bm25_first_stage_refuses_postings_that_are_not_the_corpus_rows(tmp_path: Path) -> None:
    """Postings row N scores corpus row N, so a reordered index is refused, never silently ranked."""
    export = _write_export(tmp_path, _FIXTURE_TEXTS)
    shuffled = Export(
        dataset="fixture",
        root=tmp_path,
        rows=list(reversed(export.rows)),
        embeddings=None,
        pages={},
    )
    with pytest.raises(ValueError, match="refusing to score a misaligned index"):
        Bm25Backend(shuffled)


@requires_export
def test_bm25_backend_matches_the_rescorer_on_the_real_export(export: Export) -> None:
    """The same equivalence, on 2,434 real chunks and real benchmark questions."""
    backend = Bm25Backend(export)
    assert backend.capabilities == frozenset({Capability.BM25})
    questions = [q.question for q in DATASETS["vidoseek"].queries()[:12]]
    # …plus queries lifted from the corpus itself: long, many-termed, and matching most of the
    # corpus — the shape that made the old pool-then-rescore path cost 1.78 s per query.
    questions += [" ".join(row.content.split()[:40]) for row in export.rows[:4]]
    questions.append("energy management platform for multi-site estates")
    _assert_matches_the_rescorer(export, backend, questions, 100)


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
