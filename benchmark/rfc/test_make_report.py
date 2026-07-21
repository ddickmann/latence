"""Exercise ``make_report.py`` on a tiny synthetic fixture (no pod, no network, no fake).

``make_report.py`` consumes only already-produced artifacts — the §6 results JSON, the §5 benchmark
JSON (for worked-example text), and the §3.3 stage-chart SVGs — so the whole assemble → render
pipeline is driven entirely offline through ``main`` with hand-written fixtures. No Qdrant (real or
fake), no model, no ``latence_retrieval`` import is involved anywhere in this module.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_PATH = Path(__file__).with_name("make_report.py")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_report", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mr = _load_module()

_TINY_SVG = "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>\n"


def _rec(dense: float, bm25: float, splade: float, fused: float) -> dict[str, float]:
    """A recall dict in canonical config order — keeps the unit-test rows short."""
    return {"dense": dense, "bm25": bm25, "splade": splade, "fused": fused}


# ---------------------------------------------------------------------------
# Fixture writers.
# ---------------------------------------------------------------------------


def _write_results(path: Path, *, questions_path: Path, run: str) -> None:
    """A §6-shaped results JSON: fused wins overall, plus a fused-only win + an unanswered q."""
    payload = {
        "meta": {
            "questions": str(questions_path),
            "qdrant": "http://localhost:6333",
            "collection": "rfc",
            "configs": ["dense", "bm25", "splade", "fused"],
            "top_k": 10,
            "n_questions": 3,
            "run": run,
            "fused_only_wins": 1,
            "chart": None,
        },
        "recall_at_k": {"dense": 0.33, "bm25": 0.5, "splade": 0.42, "fused": 0.83},
        "by_type": {
            "multi-hop": {"dense": 0.0, "bm25": 0.0, "splade": 0.0, "fused": 1.0},
            "lexical": {"dense": 0.5, "bm25": 1.0, "splade": 0.75, "fused": 1.0},
        },
        "per_question": [
            # A fused-only win: no single signal finds gold, fused does.
            {"id": "q1", "type": "multi-hop",
             "recall": {"dense": 0.0, "bm25": 0.0, "splade": 0.0, "fused": 1.0},
             "fused_only_win": True},
            # A lexical question every signal answers (fusion helps but is not the sole finder).
            {"id": "q2", "type": "lexical",
             "recall": {"dense": 0.5, "bm25": 1.0, "splade": 0.75, "fused": 1.0},
             "fused_only_win": False},
            # A question no configuration answered — the corpus ceiling.
            {"id": "q3", "type": "semantic",
             "recall": {"dense": 0.0, "bm25": 0.0, "splade": 0.0, "fused": 0.0},
             "fused_only_win": False},
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_questions(path: Path, run: str) -> None:
    """A §5-shaped benchmark JSON carrying the question text + KG chain the examples join on."""
    payload = {
        "meta": {"run": run, "target": 40, "min_lift": 0.24, "top_k": 10,
                 "generated": 51, "survived": 3, "by_type": {"multi-hop": 1, "lexical": 1}},
        "questions": [
            {"id": "q1", "type": "multi-hop",
             "question": "RFC 9110 obsoletes 7230 which updates 7231 — what status does it define?",
             "subject": "status code",
             "gold_chunk_ids": ["c9", "c12"],
             "kg_chain": [{"src": 9110, "rel": "obsoletes", "dst": 7230},
                          {"src": 7230, "rel": "updates", "dst": 7231}],
             "recall": {"dense": 0.0, "bm25": 0.0, "splade": 0.0, "fused": 1.0},
             "fused_only_gold_ids": ["c12"]},
            {"id": "q2", "type": "lexical",
             "question": "What does the Retry-After header field mean?",
             "subject": "Retry-After",
             "gold_chunk_ids": ["c3"],
             "kg_chain": [],
             "recall": {"dense": 0.5, "bm25": 1.0, "splade": 0.75, "fused": 1.0},
             "fused_only_gold_ids": []},
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_charts(charts_dir: Path, *, all_present: bool = True) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in mr.STAGE_CHART_FILES]
    if not all_present:
        names = names[:2]  # only some charts present → the report must note the missing ones
    for name in names:
        (charts_dir / name).write_text(_TINY_SVG, encoding="utf-8")


# ---------------------------------------------------------------------------
# The end-to-end happy path through main().
# ---------------------------------------------------------------------------


def test_main_assembles_full_proof(tmp_path: Path) -> None:
    run = str(tmp_path / "run")
    qpath = tmp_path / "rfc-benchmark.json"
    rpath = tmp_path / "rfc-results.json"
    charts = tmp_path / "stage-charts"
    out = tmp_path / "report" / "PROOF.md"
    _write_questions(qpath, run)
    _write_results(rpath, questions_path=qpath, run=run)
    _write_charts(charts, all_present=True)

    rc = mr.main(["--results", str(rpath), "--charts", str(charts), "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    md = out.read_text(encoding="utf-8")

    # 1) Headline recall table carries every config + the fused number.
    assert "| Retrieval config | recall@10 |" in md
    assert "| **fused** | **0.830** |" in md
    assert "| dense | 0.330 |" in md
    # fused's lift over the best single signal (bm25 0.5) is stated.
    assert "best single signal" in md and "+0.330" in md

    # 2) The finding banner leads with the honest verdict (fused wins here).
    assert "the fused stack wins" in md

    # 3) At least one worked example shows the joined question text + KG chain + a fused-only win.
    assert "## Worked examples" in md
    assert "RFC 9110 obsoletes 7230" in md
    assert "RFC 9110 —obsoletes→ RFC 7230 —updates→ RFC 7231" in md
    assert "fused-only win" in md
    assert "`c12`" in md  # the gold chunk only fusion recovered

    # 4) The charts section embeds the headline SVG + every §3.3 correctness chart by relative link.
    assert "![Mean recall per config](recall-headline.svg)" in md
    for name, _caption in mr.STAGE_CHART_FILES:
        assert f"(../stage-charts/{name})" in md

    # 5) Honest limitations: corpus ceiling (51→3), the unanswered question, the SPLADE NC caveat.
    assert "## Limitations" in md
    assert "**51**" in md and "**3**" in md
    assert "`q3`" in md  # the question no configuration answered
    assert "CC-BY-NC-SA-4.0" in md and "non-commercial" in md

    # 6) The one chart this script owns is a self-contained SVG next to the report (ADR-0016).
    svg = out.parent / "recall-headline.svg"
    assert svg.is_file()
    svg_text = svg.read_text(encoding="utf-8")
    assert svg_text.startswith("<svg") and svg_text.rstrip().endswith("</svg>")
    assert "href=" not in svg_text and "src=" not in svg_text and "<image" not in svg_text
    assert "url(http" not in svg_text
    assert "recall@10" in svg_text


def test_missing_stage_charts_are_noted_not_fatal(tmp_path: Path) -> None:
    run = str(tmp_path / "run")
    qpath = tmp_path / "q.json"
    rpath = tmp_path / "r.json"
    charts = tmp_path / "charts"
    out = tmp_path / "PROOF.md"
    _write_questions(qpath, run)
    _write_results(rpath, questions_path=qpath, run=run)
    _write_charts(charts, all_present=False)  # only 2 of 4 charts exist

    rc = mr.main(["--results", str(rpath), "--charts", str(charts), "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    # Present charts embed; absent ones are surfaced as a visible "Missing chart" note, not a crash.
    assert "(charts/chunk-size-distribution.svg)" in md
    assert "Missing chart" in md
    assert "kg-degree-and-centrality.svg" in md


def test_worked_examples_fall_back_without_benchmark_json(tmp_path: Path) -> None:
    # meta.questions points at a file that does not exist → the report still builds, examples use
    # recall only (no question text), with a visible note.
    rpath = tmp_path / "r.json"
    charts = tmp_path / "charts"
    out = tmp_path / "PROOF.md"
    _write_results(rpath, questions_path=tmp_path / "nope.json", run=str(tmp_path / "run"))
    _write_charts(charts)

    rc = mr.main(["--results", str(rpath), "--charts", str(charts), "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Question text unavailable" in md
    assert "## Worked examples" in md
    # The fused-only-win example still renders from recall alone.
    assert "fused-only win" in md


def test_fused_does_not_win_is_surfaced_at_top(tmp_path: Path) -> None:
    # Fused does NOT exceed bm25 → the §6 "most important finding" must lead the report and
    # repeat in limitations, never be buried.
    rpath = tmp_path / "r.json"
    charts = tmp_path / "charts"
    out = tmp_path / "PROOF.md"
    payload: dict[str, Any] = {
        "meta": {"configs": ["dense", "bm25", "splade", "fused"], "top_k": 10, "n_questions": 1,
                 "fused_only_wins": 0, "questions": None},
        "recall_at_k": {"dense": 0.4, "bm25": 0.7, "splade": 0.5, "fused": 0.6},
        "by_type": {},
        "per_question": [
            {"id": "q1", "type": "lexical",
             "recall": {"dense": 0.4, "bm25": 0.7, "splade": 0.5, "fused": 0.6},
             "fused_only_win": False},
        ],
    }
    rpath.write_text(json.dumps(payload), encoding="utf-8")
    _write_charts(charts)

    rc = mr.main(["--results", str(rpath), "--charts", str(charts), "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "fused did NOT beat every single signal" in md.replace("**", "")
    assert "Fusion verdict" in md


def test_malformed_results_is_a_finding(tmp_path: Path) -> None:
    rpath = tmp_path / "r.json"
    rpath.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    out = tmp_path / "PROOF.md"
    rc = mr.main(["--results", str(rpath), "--charts", str(tmp_path), "--out", str(out)])
    assert rc == 2
    assert not out.exists()


def test_missing_results_file_returns_2(tmp_path: Path) -> None:
    rc = mr.main([
        "--results", str(tmp_path / "nope.json"),
        "--charts", str(tmp_path),
        "--out", str(tmp_path / "PROOF.md"),
    ])
    assert rc == 2


# ---------------------------------------------------------------------------
# Unit — the selection + analysis helpers.
# ---------------------------------------------------------------------------


def test_select_worked_examples_prefers_fused_only_wins_across_types() -> None:
    results = mr.Results(
        configs=("dense", "bm25", "splade", "fused"),
        top_k=10,
        recall_at_k={},
        by_type={},
        meta={},
        per_question=[
            mr.PerQuestion("a", "multi-hop", _rec(0.0, 0.0, 0.0, 1.0), True),
            mr.PerQuestion("b", "multi-hop", _rec(0.0, 0.0, 0.0, 0.5), True),
            mr.PerQuestion("c", "lexical", _rec(0.0, 0.0, 0.0, 0.9), True),
            mr.PerQuestion("d", "lexical", _rec(0.1, 0.1, 0.1, 0.2), False),
        ],
    )
    picks = mr.select_worked_examples(results, 3)
    ids = [q.qid for q in picks]
    # All three picks are fused-only wins, interleaved across type (types visited alphabetically:
    # lexical `c` (0.9), then multi-hop `a` (1.0), then the remaining multi-hop `b`).
    assert ids == ["c", "a", "b"]


def test_no_answer_and_best_single_and_beats() -> None:
    results = mr.Results(
        configs=("dense", "bm25", "splade", "fused"),
        top_k=10,
        recall_at_k={"dense": 0.3, "bm25": 0.5, "splade": 0.4, "fused": 0.8},
        by_type={},
        meta={},
        per_question=[
            mr.PerQuestion("q1", "lexical", _rec(0.0, 0.0, 0.0, 0.0), False),
            mr.PerQuestion("q2", "lexical", _rec(0.0, 1.0, 0.0, 1.0), False),
        ],
    )
    assert mr.no_answer_questions(results) == ["q1"]
    assert mr.best_single_recall(results.recall_at_k, results.configs) == ("bm25", 0.5)
    assert mr.fused_beats_every_single(results) is True
