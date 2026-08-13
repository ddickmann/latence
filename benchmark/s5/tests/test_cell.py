# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The result-cell schema module: every shape's bytes pinned against the pre-module literals.

The cells are the campaign's oracle surface, so these tests assert *serialised bytes*, not just
dict equality: the expected payloads below are copied verbatim from what
``run_matrix``'s inline dicts used to build, and each constructor must reproduce them through
``write_cell`` byte-for-byte — key set, ``None``s, float reprs, and the deliberately-omitted
``doc_recall@k`` keys included.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmark.s5.cell import (
    Cell,
    Header,
    cell_path,
    ladder_body,
    load_cells,
    missing_inputs_cell,
    not_fusion_tunable_body,
    rescue_body,
    tuned_body,
    write_cell,
)

if TYPE_CHECKING:
    from pathlib import Path

#: A metric row exactly as ``CellAccumulator.summary`` emits one for a chain-less control
#: dataset: ``None`` chain metrics kept, ``doc_recall@k`` present, ``token_budget`` null.
_METRICS: dict[str, Any] = {
    "queries": 50,
    "recall@1": 0.5567,
    "recall@5": 0.9067,
    "recall@10": 0.96,
    "doc_recall@5": 0.96,
    "doc_recall@10": 0.98,
    "ndcg@10": 0.8059,
    "chain_coverage@10": None,
    "full_chain_hit@10": None,
    "latency_ms_median": 0.43,
    "latency_ms_p95": 0.57,
    "chain_queries": 0,
    "token_budget": None,
}

_RESOLUTION: dict[str, Any] = {
    "dataset": "vidoseek",
    "queries_total": 50,
    "queries_usable": 50,
    "refs_resolved": 50,
    "refs_total": 50,
    "resolution_rate": 1.0,
    "trustworthy": True,
}


def _header() -> Header:
    return Header(
        dataset="vidoseek",
        leg="dense",
        fusion="equal",
        vector_source="corpus",
        control=True,
        limit=50,
        resolution=_RESOLUTION,
    )


def _bytes(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class TestShapesAreBytePinned:
    def test_missing_inputs_is_the_pre_module_literal(self) -> None:
        expected = {
            "status": "missing-inputs",
            "missing": ["graph-hyperedges.kv"],
            "dataset": "vidoseek",
            "leg": "coe",
        }
        cell = missing_inputs_cell("vidoseek", "coe", ["graph-hyperedges.kv"])
        assert _bytes(cell) == _bytes(expected)

    def test_ok_ladder_is_the_pre_module_header_body_merge(self) -> None:
        expected = {
            "status": "ok",
            "dataset": "vidoseek",
            "leg": "dense",
            "fusion": "equal",
            "vector_source": "corpus",
            "control": True,
            "limit": 50,
            "resolution": _RESOLUTION,
            "metrics": _METRICS,
        }
        assert _bytes(_header().merge(ladder_body(_METRICS))) == _bytes(expected)

    def test_ok_rescue_carries_curve_gate_and_select_budget(self) -> None:
        gate = {"queries": 50, "hits": 20, "hit_rate": 0.4}
        curve = {"4000": _METRICS, "10000": _METRICS}
        payload = _header().merge(
            rescue_body(
                metrics=_METRICS, budget_curve=curve, gate=gate, select_budget=10000
            )
        )
        expected = {
            "status": "ok",
            "dataset": "vidoseek",
            "leg": "dense",
            "fusion": "equal",
            "vector_source": "corpus",
            "control": True,
            "limit": 50,
            "resolution": _RESOLUTION,
            "metrics": _METRICS,
            "budget_curve": curve,
            "gate": gate,
            "select_budget": 10000,
        }
        assert _bytes(payload) == _bytes(expected)

    def test_ok_tuned_carries_both_variants_and_the_audit_trail(self) -> None:
        tuning = {"weights": {"dense": 1.0, "bm25": 0.5}, "converged": True}
        split = {"dev_queries": 15, "test_queries": 35}
        payload = _header().merge(
            tuned_body(
                metrics=_METRICS,
                metrics_equal_weight_on_test=_METRICS,
                tuning=tuning,
                split=split,
                tunable=True,
            )
        )
        assert payload["metrics_equal_weight_on_test"] == _METRICS
        assert payload["tuning"] == tuning
        assert payload["split"] == split
        assert payload["tunable"] is True
        assert payload["status"] == "ok"

    def test_a_refusal_body_overwrites_the_headers_ok_status(self) -> None:
        """The documented merge rule: provenance kept, the claim of success replaced."""
        payload = _header().merge(not_fusion_tunable_body())
        assert payload["status"] == "not-fusion-tunable"
        assert "weight vector" in str(payload["reason"])
        assert payload["dataset"] == "vidoseek"  # the header survives the override
        assert payload["fusion"] == "equal"
        assert "metrics" not in payload  # a refusal never carries a metric row


class TestWriteAndLoad:
    def test_write_cell_serialises_sort_keys_newline_terminated(self, tmp_path: Path) -> None:
        payload = missing_inputs_cell("vidoseek", "coe", ["graph-hyperedges.kv"])
        write_cell(tmp_path, "vidoseek", "coe", payload)
        raw = cell_path(tmp_path, "vidoseek", "coe").read_text(encoding="utf-8")
        assert raw == json.dumps(payload, indent=2, sort_keys=True) + "\n"
        assert raw.splitlines()[1].lstrip().startswith('"dataset"')  # sorted, not build order

    def test_load_cells_returns_present_files_only_never_nulls(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "vidoseek", "dense", _header().merge(ladder_body(_METRICS)))
        cells = load_cells(tmp_path, "vidoseek", ["dense", "bm25"])
        assert list(cells) == ["dense"]
        assert cells["dense"]["metrics"] == _METRICS


class TestCellReader:
    def test_of_rejects_non_mappings(self) -> None:
        assert Cell.of(None) is None
        assert Cell.of("nope") is None
        assert Cell.of([]) is None

    def test_accessors_are_total_across_shapes(self) -> None:
        ok = Cell(_header().merge(ladder_body(_METRICS)))
        assert ok.is_ok and ok.status == "ok"
        assert ok.metrics == _METRICS
        assert ok.missing == []
        assert ok.gate is None and ok.budget_curve is None
        assert ok.resolution == _RESOLUTION
        assert ok.limit == 50

        refusal = Cell(missing_inputs_cell("vidoseek", "coe", ["graph-hyperedges.kv"]))
        assert not refusal.is_ok
        assert refusal.missing == ["graph-hyperedges.kv"]
        assert refusal.metrics == {}
        assert refusal.resolution is None

        empty = Cell({})
        assert empty.status == "not run"
        assert not empty.is_ok and empty.missing == [] and empty.metrics == {}
