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


def test_document_of_maps_a_chunk_back_to_the_file_it_was_chunked_from() -> None:
    """The map document-level recall is computed through — and an honest ``None`` for a stranger."""
    export = _export()
    assert export.document_of("c0") == "d.md"
    assert export.document_of("c1") == "d.md"
    assert export.document_of("c2") == "e.md"
    assert export.document_of("nope") is None


def test_the_kg_projection_defaults_to_empty_so_an_older_export_still_loads() -> None:
    """The three new columns are optional signals, not new requirements."""
    row = ChunkRow("c0", "text", "d.md", 0, 10, ("n1",))
    assert row.neighbor_node_ids == ()
    assert row.community_ids == ()
    assert row.centralities == ()


def _cell(leg: str, recall10: float, chain: float | None = None) -> dict[str, object]:
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
            "chain_coverage@10": chain,
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


def _rescue_cell(leg: str, recall10: float, chain: float, hit_rate: float) -> dict[str, object]:
    cell = _cell(leg, recall10, chain)
    metrics = cell["metrics"]
    assert isinstance(metrics, dict)
    metrics["doc_recall@5"] = 0.72
    metrics["doc_recall@10"] = 0.84
    cell["gate"] = {"mode": "auto", "queries": 89, "fired": 80, "hit_rate": hit_rate}
    cell["budget_curve"] = {
        "2000": {"recall@10": 0.20, "tokens_packed_mean": 1900.0},
        "10000": {"recall@10": recall10, "tokens_packed_mean": 5500.0},
    }
    return cell


def test_report_puts_the_rescue_family_in_its_own_table_against_the_graph_rungs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The rescue legs are not increments, so they get their own section — never a Δ column.

    The comparison that matters is against the ladder's own graph rungs (a second application of
    the same components), so those rows are repeated inside the rescue table rather than left for
    the reader to fetch from the ladder table above.
    """
    matrix = {
        "multihop_rag": {
            "dense": _cell("dense", 0.33, 0.64),
            "kg": _cell("kg", 0.31, 0.64),
            "packer": _cell("packer", 0.19, 0.66),
            "graph_rescue": _rescue_cell("graph_rescue", 0.375, 0.74, 0.90),
            "graph_rescue_packed": _rescue_cell("graph_rescue_packed", 0.31, 0.79, 0.90),
        }
    }
    legs = ("dense", "kg", "packer", "graph_rescue", "graph_rescue_packed")
    write_report(tmp_path, matrix, legs=legs, limit=100)
    report = (tmp_path / "REPORT.md").read_text()
    assert "Graph as RESCUE, not as a fused list" in report
    assert "gate hit rate" in report
    assert "90.0%" in report
    # A ladder leg has no gate at all, and the table says so instead of printing a blank.
    assert "ladder leg: no gate" in report
    # The budget curve is rendered, ascending, with the packed-token witness.
    assert "Recall over the token budget" in report
    assert "| multihop_rag | graph_rescue_packed | 2000 |" in report
    # …and the ladder table above is untouched: no rescue leg appears in its Δ chain.
    ladder_table = report.split("## Graph as RESCUE")[0]
    assert "graph_rescue" not in ladder_table
    # The document-level column is in the ladder table too, so the comparability fix is global.
    assert "docR@5" in ladder_table


def test_the_rescue_section_is_omitted_when_no_rescue_leg_ran(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A plain ladder run's report must not carry an empty promise."""
    write_report(tmp_path, {"vidoseek": {"dense": _cell("dense", 0.4)}}, legs=("dense",), limit=1)
    assert "Graph as RESCUE" not in (tmp_path / "REPORT.md").read_text()


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


# -- the two fusion variants, side by side ------------------------------------------------------


def _weighted_cell(leg: str, tuned: float, equal: float) -> dict[str, object]:
    """A tuned cell: the tuned number, its equal-weight twin on the SAME test queries, the fit."""
    return {
        "status": "ok",
        "leg": leg,
        "fusion": "tuned",
        "metrics": {"recall@10": tuned, "ndcg@10": tuned, "chain_coverage@10": 0.7},
        "metrics_equal_weight_on_test": {
            "recall@10": equal,
            "ndcg@10": equal,
            "chain_coverage@10": 0.6,
        },
        "split": {"dev_queries": 300, "dev_queries_tuned_on": 300, "test_queries": 700},
        "tuning": {
            "weights": {"dense": 1.0, "bm25": 0.75, "kg": 0.0},
            "objective": "ndcg@10",
            "grid_evaluations": 29,
            "converged": True,
            "max_passes": 3,
            "dev_ndcg@10_equal_weight": 0.41,
            "dev_ndcg@10_tuned": 0.47,
        },
    }


def test_report_shows_both_fusion_variants_with_the_delta_on_the_same_split(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Equal-weight baseline and dev-tuned variant, each labelled as what it is."""
    equal_matrix = {"multihop_rag": {"kg": _cell("kg", 0.42)}}
    weighted = {"multihop_rag": {"kg": _weighted_cell("kg", tuned=0.50, equal=0.44)}}
    write_report(tmp_path, equal_matrix, legs=("kg",), limit=None, weighted=weighted)
    report = (tmp_path / "REPORT.md").read_text()

    assert "EQUAL-weight RRF" in report  # the ladder table says what it is
    assert "Dev-tuned fusion weights" in report
    # Both variants and the delta between them, on the held-out split — not against the 0.42 cell.
    assert "| multihop_rag | kg | 300 | 700 | 0.4400 | 0.5000 | +0.0600 |" in report
    assert "0.42" not in report.split("## Dev-tuned")[1]
    # The fitted weights are auditable from the report itself.
    assert "dense 1.00, bm25 0.75, kg 0.00" in report
    assert "converged" in report
    # …and the non-comparability is stated, not left to the reader.
    assert "not comparable to published baselines" in report
    assert json.loads((tmp_path / "metrics-weighted.json").read_text())["multihop_rag"]["kg"]


def test_a_single_signal_legs_weighted_row_says_there_was_nothing_to_fit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The `dense` row exists so the table has a same-split reference — not as a claimed fit."""
    cell = _weighted_cell("dense", tuned=0.30, equal=0.30)
    cell["tunable"] = False
    cell["tuning"] = {**cell["tuning"], "weights": {"dense": 1.0}}  # type: ignore[dict-item]
    write_report(
        tmp_path, {"uda": {}}, legs=("dense",), limit=None, weighted={"uda": {"dense": cell}}
    )
    report = (tmp_path / "REPORT.md").read_text()
    assert "single signal: the RRF weight is a pure scale" in report


def test_report_without_a_weighted_run_is_the_baseline_report_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The second variant must be invisible until it exists — the in-flight baseline is untouched."""
    matrix = {"multihop_rag": {"kg": _cell("kg", 0.42)}}
    write_report(tmp_path, matrix, legs=("kg",), limit=None)
    report = (tmp_path / "REPORT.md").read_text()
    assert "## Dev-tuned" not in report
    assert not (tmp_path / "metrics-weighted.json").exists()


def test_report_surfaces_where_chain_coverage_and_recall_disagree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The signal in the data: +hyperedges loses chunks while covering MORE gold documents."""
    matrix = {
        "multihop_rag": {
            "kg": _cell("kg", 0.42, chain=0.652),
            "hyperedges": _cell("hyperedges", 0.39, chain=0.671),
        }
    }
    write_report(tmp_path, matrix, legs=("kg", "hyperedges"), limit=None)
    report = (tmp_path / "REPORT.md").read_text()
    assert "Chain coverage vs recall" in report
    assert "| multihop_rag | kg → hyperedges | -0.0300 | +0.0190 | **diverges" in report


def test_the_divergence_table_is_omitted_where_there_is_no_chain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Single-hop controls have no chain, so they contribute no row rather than a row of dashes."""
    matrix = {"vidoseek": {"dense": _cell("dense", 0.4), "bm25": _cell("bm25", 0.5)}}
    write_report(tmp_path, matrix, legs=("dense", "bm25"), limit=None)
    assert "Chain coverage vs recall" not in (tmp_path / "REPORT.md").read_text()


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
