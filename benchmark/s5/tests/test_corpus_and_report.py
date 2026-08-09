# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The export's ChunkIndex behaviour and the report's delta/control rendering."""

from __future__ import annotations

import json

from benchmark.s5.corpus import ChunkRow, Export, normalize_ws
from benchmark.s5.pagemap import DocumentPages
from benchmark.s5.report import write_report


def _export() -> Export:
    rows = [
        ChunkRow("c0", "Alpha beta", "d.md", 0, 100, ("n1",)),
        ChunkRow("c1", "Gamma  DELTA\n text", "d.md", 90, 250, ("n1", "n2")),
        ChunkRow("c2", "Other doc", "e.md", 0, 50, ()),
    ]
    pages = {"d": DocumentPages(doc_id="d", starts=(0, 120), ends=(110, 250))}
    return Export(dataset="t", root=None, rows=rows, embeddings=None, pages=pages)  # type: ignore[arg-type]


def test_chunks_on_page_uses_span_overlap_not_containment() -> None:
    export = _export()
    assert list(export.chunks_on_page("d.md", 1)) == ["c0", "c1"]
    assert list(export.chunks_on_page("d.md", 2)) == ["c1"]


def test_chunks_on_page_returns_nothing_when_no_page_map_exists() -> None:
    """A missing page map must NOT silently widen to document-level gold."""
    export = _export()
    assert list(export.chunks_on_page("e.md", 1)) == []


def test_chunks_containing_is_whitespace_and_case_insensitive() -> None:
    export = _export()
    assert list(export.chunks_containing("d.md", "gamma delta")) == ["c1"]
    assert list(export.chunks_containing("d.md", "")) == []
    assert normalize_ws("  A\n B ") == "a b"


def test_rows_for_nodes_deduplicates_and_sorts() -> None:
    export = _export()
    assert export.rows_for_nodes(["n1", "n2", "n1"]) == [0, 1]
    assert export.index_of("c2") == 2
    assert export.index_of("nope") is None


def _cell(leg: str, recall10: float) -> dict[str, object]:
    return {
        "status": "ok",
        "leg": leg,
        "resolution": {
            "refs_resolved": 10,
            "refs_total": 10,
            "resolution_rate": 1.0,
            "queries_usable": 10,
            "queries_total": 10,
        },
        "metrics": {
            "recall@1": 0.1,
            "recall@5": 0.2,
            "recall@10": recall10,
            "ndcg@10": 0.3,
            "chain_coverage@10": None,
            "full_chain_hit@10": None,
            "latency_ms_median": 1.0,
            "latency_ms_p95": 2.0,
        },
    }


def test_report_renders_cumulative_deltas_and_flags_control_leakage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    matrix = {
        "vidoseek": {
            "dense": _cell("dense", 0.40),
            "bm25": _cell("bm25", 0.50),
            "kg": _cell("kg", 0.50),
            "hyperedges": _cell("hyperedges", 0.70),
        }
    }
    write_report(tmp_path, matrix, legs=("dense", "bm25", "kg", "hyperedges"), limit=50)
    report = (tmp_path / "REPORT.md").read_text()
    assert "+0.1000" in report  # dense → bm25
    assert "possible leakage" in report  # a +0.20 control jump from the hypergraph leg
    assert "CONTROL" in report
    assert json.loads((tmp_path / "metrics.json").read_text())["vidoseek"]["dense"]


def test_report_renders_a_missing_input_cell_as_a_refusal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    matrix = {
        "musique": {
            "dense": {"status": "missing-inputs", "missing": ["corpus.parquet:embedding"]}
        }
    }
    write_report(tmp_path, matrix, legs=("dense",), limit=None)
    report = (tmp_path / "REPORT.md").read_text()
    assert "missing inputs: corpus.parquet:embedding" in report
    assert "not run" in report
