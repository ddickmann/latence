# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The export's ChunkIndex behaviour and the report's delta/control rendering."""

from __future__ import annotations

import json

from benchmark.s5.corpus import ChunkRow, Export
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


def test_has_page_map_separates_unavailable_from_wrong() -> None:
    """The predicate the document-level fallback turns on — per document, not per dataset."""
    export = _export()
    assert export.has_page_map("d.md") is True
    assert export.has_page_map("e.md") is False


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


def test_report_states_the_granularity_mix_and_the_exclusion_reasons(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The full-set resolution file wins over a cell's ``--limit`` copy, and both are surfaced."""
    (tmp_path / "resolution").mkdir()
    (tmp_path / "resolution" / "ohr_bench.json").write_text(
        json.dumps(
            {
                "dataset": "ohr_bench",
                "queries_total": 5039,
                "queries_usable": 4541,
                "queries_excluded": 498,
                "refs_total": 4541,
                "refs_resolved": 4541,
                "resolution_rate": 1.0,
                "granularity": {"page": 4535, "document": 6},
                "exclusions": {"corpus-missing-domain": 498},
            }
        )
    )
    write_report(tmp_path, {"ohr_bench": {"dense": _cell("dense", 0.4)}}, legs=("dense",), limit=50)
    report = (tmp_path / "REPORT.md").read_text()
    assert "4541/4541" in report  # the full set, not the cell's 10/10
    assert "4535 / 6" in report
    assert "498 (corpus-missing-domain 498)" in report


def test_report_puts_the_ablation_in_its_own_table_not_in_the_ladder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The Δ column must stay a chain of increments; the ablation is a separate subtraction."""
    matrix = {
        "multihop_rag": {
            "dense": _cell("dense", 0.60),
            "bm25": _cell("bm25", 0.70),
            "dense_content_only": _cell("dense_content_only", 0.55),
        }
    }
    write_report(
        tmp_path, matrix, legs=("dense", "bm25", "dense_content_only"), limit=None
    )
    report = (tmp_path / "REPORT.md").read_text()
    assert "context header bought" in report
    assert "| multihop_rag | dense_content_only | 0.6000 | 0.5500 | +0.0500 |" in report
    # …and the ladder table below the dataset heading only carries the two ladder rungs.
    ladder = report.split("## multihop_rag")[1].split("## Ablation")[0]
    assert "dense_content_only" not in ladder
    assert "| bm25 |" in ladder


def test_report_omits_the_ablation_section_when_no_ablation_leg_ran(tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_report(tmp_path, {"uda": {"dense": _cell("dense", 0.4)}}, legs=("dense",), limit=None)
    assert "## Ablation" not in (tmp_path / "REPORT.md").read_text()


def test_report_shows_an_unbuilt_sidecar_as_a_refusal_not_as_no_difference(tmp_path) -> None:  # type: ignore[no-untyped-def]
    matrix = {
        "musique": {
            "dense": _cell("dense", 0.5),
            "dense_content_only": {
                "status": "missing-inputs",
                "missing": ["corpus-vectors-content-only.parquet"],
            },
        }
    }
    write_report(tmp_path, matrix, legs=("dense", "dense_content_only"), limit=None)
    report = (tmp_path / "REPORT.md").read_text()
    assert "missing inputs: corpus-vectors-content-only.parquet" in report
    assert "| musique | dense_content_only | — |" in report


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
