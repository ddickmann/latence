"""Exercise ``run_benchmark.py`` on a tiny synthetic fixture (no pod, no network, no fake).

Two prongs, both offline:

* **the measurement layer** — a planted, in-memory :class:`RetrievalEngine` (per-config ranked id
  lists, NOT a ``FakeQdrantClient``) drives the whole load → measure → aggregate → emit pipeline
  through ``main``, so recall@k, the per-type breakdown, the §6 fused-only-win bar, and the
  results-JSON + dependency-free SVG are proven with no Qdrant at all.
* **the real fused chain** — a :class:`LatenceRetrievalEngine` wired from ``latence_retrieval``'s
  own faithful doubles (:class:`FakeBackend`, :class:`FakeCrossEncoder`, :class:`FakeGraphSource`,
  the heuristic token counter) exercises the actual RRF → KG-injection → 2-hop hubs-filtered
  expand → rerank → pack@budget product code — again with no pod, network, or Qdrant fake.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_PATH = Path(__file__).with_name("run_benchmark.py")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_benchmark", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses resolve their string annotations
    # (``from __future__ import annotations``) via ``sys.modules[__name__]``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rb = _load_module()


# ---------------------------------------------------------------------------
# Fixture writers.
# ---------------------------------------------------------------------------


def _write_questions(path: Path, questions: list[dict[str, Any]], run: str) -> None:
    payload = {"meta": {"run": run, "top_k": 1}, "questions": questions}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_corpus(run: Path, records: list[dict[str, Any]]) -> None:
    export = run / "export"
    export.mkdir(parents=True, exist_ok=True)
    (export / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


class _StubEngine:
    """A planted :class:`RetrievalEngine`: per-question, per-config ranked chunk-id lists."""

    def __init__(self, plans: dict[str, dict[str, list[str]]]) -> None:
        self._plans = plans

    def rank(self, query: str, top_k: int, configs: Any) -> dict[str, list[str]]:
        plan = self._plans.get(query, {})
        return {c: plan.get(c, [])[:top_k] for c in configs}


# ---------------------------------------------------------------------------
# Prong 1 — the measurement layer via main() (deterministic, planted rankings).
# ---------------------------------------------------------------------------


def test_main_measures_recall_and_emits_json_and_svg(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_corpus(
        run,
        [
            {"record_id": "c1", "content": "alpha"},
            {"record_id": "c2", "content": "beta"},
            {"record_id": "c5", "content": "omega"},
        ],
    )
    questions = [
        {"id": "q1", "type": "multi-hop", "question": "Q1", "gold_chunk_ids": ["c5"]},
        {"id": "q2", "type": "lexical", "question": "Q2", "gold_chunk_ids": ["c2"]},
    ]
    qpath = tmp_path / "rfc-benchmark.json"
    _write_questions(qpath, questions, run=str(run))

    plans = {
        # fused surfaces c5 that no single signal finds → a fused-only win.
        "Q1": {"dense": ["c1"], "bm25": ["c1"], "splade": ["c2"], "fused": ["c5", "c1"]},
        # every signal (and fused) finds c2 → not a fused-only win.
        "Q2": {"dense": ["c1"], "bm25": ["c2"], "splade": ["c2"], "fused": ["c2"]},
    }
    engine = _StubEngine(plans)

    out = tmp_path / "results.json"
    rc = rb.main(
        [
            "--questions", str(qpath),
            "--out", str(out),
            "--configs", "dense,bm25,splade,fused",
            "--top-k", "1",
        ],
        engine_factory=lambda _run, _corpus, _args: engine,
    )
    assert rc == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    recall = payload["recall_at_k"]
    assert recall == {"dense": 0.0, "bm25": 0.5, "splade": 0.5, "fused": 1.0}

    by_type = payload["by_type"]
    assert by_type["multi-hop"] == {"dense": 0.0, "bm25": 0.0, "splade": 0.0, "fused": 1.0}
    assert by_type["lexical"] == {"dense": 0.0, "bm25": 1.0, "splade": 1.0, "fused": 1.0}

    assert payload["meta"]["fused_only_wins"] == 1
    assert payload["meta"]["n_questions"] == 2
    assert payload["meta"]["run"] == str(run)
    assert len(payload["per_question"]) == 2
    q1 = next(q for q in payload["per_question"] if q["id"] == "q1")
    assert q1["fused_only_win"] is True

    # The dependency-free SVG chart was written next to the results and references no external host.
    svg = out.with_suffix(".svg")
    assert svg.is_file()
    text = svg.read_text(encoding="utf-8")
    assert text.startswith("<svg") and text.rstrip().endswith("</svg>")
    # Self-contained: the only URL is the SVG namespace identifier — no external asset is fetched.
    assert "href=" not in text and "src=" not in text and "<image" not in text
    assert "url(http" not in text
    assert "recall@1" in text


def test_configs_subset_only_runs_requested(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_corpus(run, [{"record_id": "c1", "content": "alpha"}])
    qpath = tmp_path / "b.json"
    _write_questions(
        qpath,
        [{"id": "q1", "type": "lexical", "question": "Q", "gold_chunk_ids": ["c1"]}],
        run=str(run),
    )
    engine = _StubEngine({"Q": {"dense": ["c1"], "bm25": ["c1"]}})
    out = tmp_path / "r.json"
    rc = rb.main(
        ["--questions", str(qpath), "--out", str(out), "--configs", "dense,bm25", "--top-k", "1"],
        engine_factory=lambda _run, _corpus, _args: engine,
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["recall_at_k"]) == {"dense", "bm25"}
    # fused was not requested, so no fused-only-win claim is made.
    assert payload["meta"]["fused_only_wins"] == 0


def test_missing_corpus_for_bm25_is_a_finding_not_a_fake(tmp_path: Path) -> None:
    # meta.run points nowhere → no exported corpus. bm25/fused cannot run for real; the CLI must
    # refuse (fabricating the pool is what §6 forbids), never silently green.
    qpath = tmp_path / "b.json"
    _write_questions(
        qpath,
        [{"id": "q1", "type": "lexical", "question": "Q", "gold_chunk_ids": ["c1"]}],
        run=str(tmp_path / "nonexistent"),
    )
    out = tmp_path / "r.json"
    rc = rb.main(
        ["--questions", str(qpath), "--out", str(out), "--configs", "bm25"],
        engine_factory=lambda _run, _corpus, _args: _StubEngine({}),
    )
    assert rc == 2
    assert not out.exists()


def test_unknown_config_rejected(tmp_path: Path) -> None:
    qpath = tmp_path / "b.json"
    _write_questions(
        qpath,
        [{"id": "q1", "type": "lexical", "question": "Q", "gold_chunk_ids": ["c1"]}],
        run=str(tmp_path),
    )
    out = tmp_path / "r.json"
    rc = rb.main(
        ["--questions", str(qpath), "--out", str(out), "--configs", "dense,bogus"],
        engine_factory=lambda _run, _corpus, _args: _StubEngine({}),
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# Unit — the metric, the hub filter, question validation.
# ---------------------------------------------------------------------------


def test_recall_at_k() -> None:
    assert rb.recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert rb.recall_at_k(["a", "b"], ["a", "b"], 10) == 1.0
    assert rb.recall_at_k(["a"], [], 5) == 0.0
    # gold outside the top-k does not count.
    assert rb.recall_at_k(["x", "y", "gold"], ["gold"], 2) == 0.0


def test_hub_nodes_and_hub_filtered_graph_source() -> None:
    # nhub is connected to everything → highest degree → the single hub at fraction 0.2.
    edges = [("nhub", "n1"), ("nhub", "n2"), ("nhub", "n3"), ("n1", "n2")]
    hubs = rb.hub_nodes_from_edges(edges, 0.2)
    assert hubs == frozenset({"nhub"})

    from latence_retrieval.testing import FakeGraphSource

    inner = FakeGraphSource(edges)
    # Without filtering: n3 is 2 hops from n1 (via nhub) and nhub is 1 hop.
    assert inner.neighbors(["n1"], 2) == {"nhub": 1, "n2": 1, "n3": 2}

    filtered = rb.HubFilteredGraphSource(inner, hubs)
    # nhub is dropped from the neighbourhood (earns no bonus); n3, reachable through it, remains.
    assert filtered.neighbors(["n1"], 2) == {"n2": 1, "n3": 2}
    # A hub seed contributes nothing (it is removed from the seed set → nothing to expand from).
    assert filtered.neighbors(["nhub"], 2) == {}


def test_hub_fraction_zero_disables_filter() -> None:
    edges = [("a", "b"), ("b", "c")]
    assert rb.hub_nodes_from_edges(edges, 0.0) == frozenset()


def test_load_questions_rejects_malformed(tmp_path: Path) -> None:
    import pytest

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"meta": {}, "questions": [{"id": "q1", "type": "t"}]}), "utf-8")
    with pytest.raises(ValueError, match="question"):
        rb.load_questions(bad)


# ---------------------------------------------------------------------------
# Prong 2 — the REAL fused chain over latence_retrieval's faithful doubles.
# ---------------------------------------------------------------------------


def _real_engine(rbmod: ModuleType, corpus: Any) -> Any:
    """Wire a real :class:`LatenceRetrievalEngine` from the shipped faithful doubles (offline)."""
    from latence_retrieval import (
        Bm25Rescorer,
        Candidate,
        HeuristicTokenCounter,
        MultiHopExpander,
        Packer,
        Reranker,
        RrfFuser,
    )
    from latence_retrieval.contracts import Capability
    from latence_retrieval.testing import FakeBackend, FakeCrossEncoder, FakeGraphSource

    # Dense/sparse backends: query-independent ranking (score order), faithful to the seam contract.
    dim = 4
    dense_corpus = [
        Candidate(id=c.chunk_id, score=score, text=c.content, metadata={"record_id": c.chunk_id})
        for c, score in zip(corpus, [5.0, 4.0, 3.0, 2.0, 1.0], strict=False)
    ]
    dense_backend = FakeBackend(dense_corpus, [Capability.DENSE], dimension=dim)
    sparse_backend = FakeBackend(
        [Candidate(id=c.id, score=c.score, text=c.text, metadata=c.metadata) for c in dense_corpus],
        [Capability.SPARSE],
    )

    def dense_encoder(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def sparse_encoder(text: str) -> dict[str, float]:
        return {tok: 1.0 for tok in text.lower().split()}

    edges = [("n1", "n5")]  # a direct, non-hub bridge from c1's entity to c5's.
    graph = rbmod.HubFilteredGraphSource(
        FakeGraphSource(edges), rbmod.hub_nodes_from_edges(edges, 0.0)
    )
    multihop = MultiHopExpander({"max_hops": 2, "seed_top_k": 3}, graph_source=graph)

    return rbmod.LatenceRetrievalEngine(
        corpus,
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        bm25=Bm25Rescorer(),
        fuser=RrfFuser(),
        multihop=multihop,
        reranker=Reranker(model=FakeCrossEncoder()),
        packer=Packer({"budget": 16000}, tokens=HeuristicTokenCounter()),
        pool=100,
    )


def test_real_fused_chain_runs_offline(tmp_path: Path) -> None:
    corpus = [
        rb.CorpusChunk(chunk_id="c1", content="alpha protocol header", kg_node_ids=("n1",)),
        rb.CorpusChunk(chunk_id="c2", content="beta status code", kg_node_ids=("n2",)),
        rb.CorpusChunk(chunk_id="c3", content="gamma method", kg_node_ids=("n3",)),
        rb.CorpusChunk(chunk_id="c4", content="delta field", kg_node_ids=("n4",)),
        rb.CorpusChunk(chunk_id="c5", content="omega alpha protocol", kg_node_ids=("n5",)),
    ]
    engine = _real_engine(rb, corpus)

    ranked = engine.rank("alpha protocol", top_k=3, configs=rb.ALL_CONFIGS)
    corpus_ids = {c.chunk_id for c in corpus}
    for config in rb.ALL_CONFIGS:
        assert config in ranked
        assert ranked[config], f"{config} returned no candidates"
        assert set(ranked[config]) <= corpus_ids  # real ids only, nothing fabricated
    # The full RRF → KG-inject → 2-hop expand → rerank → pack chain ran and produced a real
    # permutation of the retrieved pool: the two "alpha protocol"-bearing chunks (c1, c5) top it.
    assert set(ranked["fused"][:2]) == {"c1", "c5"}
    # bm25 is genuinely query-driven: an "alpha protocol" chunk outranks the lexically-empty rest.
    assert ranked["bm25"][0] in {"c1", "c5"}

    # And it flows through main() end-to-end with the real engine.
    run = tmp_path / "run"
    _write_corpus(
        run, [{"record_id": c.chunk_id, "content": c.content} for c in corpus]
    )
    qpath = tmp_path / "b.json"
    _write_questions(
        qpath,
        [{"id": "q1", "type": "multi-hop", "question": "alpha protocol",
          "gold_chunk_ids": ["c5"]}],
        run=str(run),
    )
    out = tmp_path / "r.json"
    rc = rb.main(
        ["--questions", str(qpath), "--out", str(out), "--top-k", "5"],
        engine_factory=lambda _run, _corpus, _args: _real_engine(rb, _corpus),
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # top_k=5 over a 5-chunk corpus: every config recovers the gold — the point here is that the
    # real engine flows through main() and emits a real measurement, not a manufactured one.
    assert payload["recall_at_k"]["fused"] == 1.0
