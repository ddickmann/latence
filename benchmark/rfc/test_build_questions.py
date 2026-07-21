"""Exercise ``build_questions.py`` end-to-end on a tiny synthetic run (no pod, no network, no fake).

The fixture hand-builds the exact artifacts the miner consumes — an ``export/records.jsonl`` (chunks
with ``content`` + Provenance pointing at ``rfcNNNN.txt``) and the corpus's independent
``_refs.jsonl`` — then drives the whole mine → literal-guard → recall-gate → emit pipeline through
an **injected in-memory** :class:`Signals` (NOT a ``FakeQdrantClient`` — retrieval is stubbed above
the Qdrant seam, so the gate logic is proven with no Qdrant at all).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = Path(__file__).with_name("build_questions.py")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_questions", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module's dataclasses resolve their string annotations
    # (``from __future__ import annotations``) via ``sys.modules[__name__]``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bq = _load_module()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class _StubSignals:
    """An in-memory :class:`Signals`: per-question ranked chunk-id lists, keyed by exact query text.

    This is the faithful stand-in the offline gate runs on — a plain dict of rankings, never a
    Qdrant client. A query with no planted rankings ranks nothing (every signal empty), so an
    ungated candidate simply fails the gate rather than erroring.
    """

    def __init__(self, plans: dict[str, dict[str, list[str]]]) -> None:
        self._plans = plans

    def rank(self, query: str, top_k: int) -> dict[str, list[str]]:
        plan = self._plans.get(query, {})
        return {
            "dense": plan.get("dense", [])[:top_k],
            "bm25": plan.get("bm25", [])[:top_k],
            "splade": plan.get("splade", [])[:top_k],
            "fused": plan.get("fused", [])[:top_k],
        }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus-rfc"
    corpus.mkdir()
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)

    def _uri(n: int) -> str:
        return str(corpus / f"rfc{n}.txt")

    # Chunks: RFC 9110 (obsoletes 7230, which references 3986) + a compound bridge via RFC 3986.
    rows: list[dict[str, object]] = [
        {"record_id": "c9110", "content": "RFC 9110 defines HTTP semantics. Retry-After: 120",
         "provenance": {"source_uri": _uri(9110)}},
        {"record_id": "c7230", "content": "RFC 7230 message syntax and routing (see RFC 3986), 404",
         "provenance": {"source_uri": _uri(7230)}},
        {"record_id": "c3986a", "content": "RFC 3986 URI generic syntax, the terminal spec",
         "provenance": {"source_uri": _uri(3986)}},
        {"record_id": "c3986b", "content": "More of RFC 3986: percent-encoding and reserved set",
         "provenance": {"source_uri": _uri(3986)}},
        {"record_id": "c8288", "content": "RFC 8288 web linking builds on RFC 3986 references",
         "provenance": {"source_uri": _uri(8288)}},
    ]
    _write_jsonl(run / "export" / "records.jsonl", rows)

    # Independent gold graph: 9110 obsoletes 7230; 7230 references 3986; 8288 references 3986.
    _write_jsonl(
        corpus / "_refs.jsonl",
        [
            {"rfc": 9110, "obsoletes": [7230], "updates": [], "references": []},
            {"rfc": 7230, "obsoletes": [], "updates": [], "references": [3986]},
            {"rfc": 8288, "obsoletes": [], "updates": [], "references": [3986]},
            {"rfc": 3986, "obsoletes": [], "updates": [], "references": []},
        ],
    )
    return run


def _refs_path(run_dir: Path) -> Path:
    return run_dir.parent.parent / "corpus-rfc" / "_refs.jsonl"


# --- unit-level: readers, literal guard, generators -----------------------------------------


def test_loaders_and_rfc_recovery(run_dir: Path) -> None:
    chunks = bq.load_chunks(run_dir)
    assert {c.chunk_id for c in chunks} == {"c9110", "c7230", "c3986a", "c3986b", "c8288"}
    assert {c.rfc for c in chunks} == {9110, 7230, 3986, 8288}
    assert bq.rfc_of_source_uri("/x/rfc9110.txt") == 9110
    assert bq.rfc_of_source_uri("/x/nope.txt") is None

    graph = bq.load_refs(_refs_path(run_dir))
    assert graph[9110]["obsoletes"] == [7230]
    assert graph[7230]["references"] == [3986]


def test_subject_present_matches_rfc_surface_forms() -> None:
    assert bq.subject_present("RFC 3986", "see rfc3986 for details")
    assert bq.subject_present("RFC 3986", "RFC-3986 generic syntax")
    assert not bq.subject_present("RFC 3986", "RFC 9999 is unrelated")
    assert bq.subject_present("Retry-After", "the Retry-After header")
    assert not bq.subject_present("Retry-After", "no such header here")


def test_generation_covers_the_four_flavours(run_dir: Path) -> None:
    chunks = bq.load_chunks(run_dir)
    graph = bq.load_refs(_refs_path(run_dir))
    cands = bq.generate_candidates(chunks, graph)
    types = {c.qtype for c in cands}
    assert {"multi-hop", "lexical", "semantic", "compound"} <= types

    # The multi-hop chain 9110 -obsoletes-> 7230 -references-> 3986 lands on terminal RFC 3986.
    mh = next(c for c in cands if c.qtype == "multi-hop")
    assert mh.subject == "RFC 3986"
    assert set(mh.gold_chunk_ids) == {"c3986a", "c3986b"}
    assert [h.as_dict() for h in mh.kg_chain] == [
        {"src": 9110, "rel": "obsoletes", "dst": 7230},
        {"src": 7230, "rel": "references", "dst": 3986},
    ]

    # The compound bridges the two citing RFCs (7230, 8288) via the co-referenced RFC 3986; gold is
    # the section in EACH that carries the bridge (not the terminal spec's own chunks).
    cmp = next(c for c in cands if c.qtype == "compound")
    assert cmp.subject == "RFC 3986"
    assert set(cmp.gold_chunk_ids) == {"c7230", "c8288"}
    assert [h.as_dict() for h in cmp.kg_chain] == [
        {"src": 7230, "rel": "references", "dst": 3986},
        {"src": 8288, "rel": "references", "dst": 3986},
    ]


# --- the gate: literal presence + fused-vs-single recall ------------------------------------


def test_literal_presence_drops_non_literal_gold(run_dir: Path) -> None:
    chunks = bq.load_chunks(run_dir)
    index = {c.chunk_id: c for c in chunks}
    # Gold chunk c9110 does NOT contain "RFC 3986" — the guard must reject the candidate outright,
    # before any signal is consulted.
    bad = bq.QuestionCandidate(
        qid="x", qtype="multi-hop", question="q", subject="RFC 3986",
        gold_chunk_ids=("c3986a", "c9110"),
    )
    signals = _StubSignals({"q": {"fused": ["c3986a"], "dense": [], "bm25": [], "splade": []}})
    assert bq.gate_candidate(bad, signals, top_k=10, min_lift=0.24, chunk_index=index) is None


def test_gate_keeps_only_fusion_wins(run_dir: Path) -> None:
    chunks = bq.load_chunks(run_dir)
    index = {c.chunk_id: c for c in chunks}
    cand = bq.QuestionCandidate(
        qid="mh", qtype="multi-hop", question="chain-q", subject="RFC 3986",
        gold_chunk_ids=("c3986a", "c3986b"),
    )

    # Fusion recovers BOTH gold; each single finds at most one → fused=1.0 vs best-single=0.5,
    # lift 0.5 > 0.24, and c3986b is found only by fused. KEEP.
    win = _StubSignals({"chain-q": {
        "dense": ["c3986a", "c9110"],
        "bm25": ["c7230", "c3986a"],
        "splade": ["c9110"],
        "fused": ["c3986a", "c3986b"],
    }})
    kept = bq.gate_candidate(cand, win, top_k=10, min_lift=0.24, chunk_index=index)
    assert kept is not None
    assert kept.recall["fused"] == 1.0
    assert kept.recall["dense"] == 0.5
    assert kept.fused_only_gold_ids == ("c3986b",)

    # A single signal already finds everything fused does → no fused-only gold → DROP even at 1.0.
    no_edge = _StubSignals({"chain-q": {
        "dense": ["c3986a", "c3986b"],
        "bm25": [],
        "splade": [],
        "fused": ["c3986a", "c3986b"],
    }})
    assert bq.gate_candidate(cand, no_edge, top_k=10, min_lift=0.24, chunk_index=index) is None

    # Lift below min_lift → DROP (fused 1.0 but best-single 0.5 too; here make lift exactly 0.5 but
    # raise min_lift above it).
    assert bq.gate_candidate(cand, win, top_k=10, min_lift=0.9, chunk_index=index) is None


def test_recall_at_k() -> None:
    assert bq.recall_at_k(["a", "b", "c"], ["a", "c"], 10) == 1.0
    assert bq.recall_at_k(["a", "x"], ["a", "c"], 10) == 0.5
    assert bq.recall_at_k(["a"], [], 10) == 0.0
    assert bq.recall_at_k(["z", "a"], ["a"], 1) == 0.0  # a is below the top-1 cut-off


# --- end-to-end through main(), with the injected stub factory ------------------------------


def test_main_emits_gated_benchmark_json(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report" / "rfc-benchmark.json"

    def factory(run: Path, chunks: object, args: object) -> _StubSignals:
        # Plan a fusion win for the multi-hop question (terminal RFC 3986), nothing else → exactly
        # one survivor, proving the gate is doing the filtering, not the generator.
        mh_q = (
            "RFC 9110 obsoletes RFC 7230, which references RFC 3986. "
            "What does the specification at the end of that chain define?"
        )
        return _StubSignals({mh_q: {
            "dense": ["c3986a"],
            "bm25": ["c7230"],
            "splade": ["c9110"],
            "fused": ["c3986a", "c3986b"],
        }})

    rc = bq.main(
        [
            "--run", str(run_dir),
            "--refs", str(_refs_path(run_dir)),
            "--out", str(out),
            "--target", "40",
            "--min-lift", "0.24",
        ],
        signals_factory=factory,
    )
    assert rc == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["survived"] == 1
    assert payload["meta"]["generated"] >= 4
    assert payload["meta"]["by_type"] == {"multi-hop": 1}

    (q,) = payload["questions"]
    assert q["type"] == "multi-hop"
    assert q["subject"] == "RFC 3986"
    assert set(q["gold_chunk_ids"]) == {"c3986a", "c3986b"}
    assert q["kg_chain"] == [
        {"src": 9110, "rel": "obsoletes", "dst": 7230},
        {"src": 7230, "rel": "references", "dst": 3986},
    ]
    assert q["recall"] == {"dense": 0.5, "bm25": 0.0, "splade": 0.0, "fused": 1.0}
    assert q["fused_only_gold_ids"] == ["c3986b"]


def test_main_errors_on_missing_run(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    rc = bq.main(
        ["--run", str(tmp_path / "nope"), "--refs", str(tmp_path / "r.jsonl"), "--out", str(out)],
        signals_factory=lambda *_: _StubSignals({}),
    )
    assert rc == 2
    assert not out.exists()
