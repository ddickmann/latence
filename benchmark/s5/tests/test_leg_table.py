# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The LegSpec table: pinned against the dispatch logic it replaced, byte-order included.

``missing_inputs`` order is load-bearing — the cumulative, deduplicated file list appears
VERBATIM in written ``missing-inputs`` cells — so the equality test below runs the replaced
algorithm (copied literally from the pre-table ``legs.missing_inputs``) beside the table lookup
for every (leg × absent-file) combination and asserts identical lists, not identical sets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from benchmark.s5.corpus import CONTENT_ONLY_VECTORS
from benchmark.s5.legs import (
    ABLATION_LEG_NAMES,
    ALL_LEG_NAMES,
    FUSION_SIGNALS,
    LEG_NAMES,
    LEG_TABLE,
    RESCUE_LEG_NAMES,
    leg_signals,
    missing_inputs,
    vector_source,
)
from benchmark.s5.tuning import is_tunable_leg

if TYPE_CHECKING:
    from pathlib import Path

#: Every artifact any leg can require — the universe the absence combinations draw from.
_ALL_FILES = tuple(
    dict.fromkeys(f for spec in LEG_TABLE.values() for f in spec.required_files)
)


def _legacy_missing_inputs(export_dir: Path, leg: str) -> list[str]:
    """The pre-table dispatch, copied verbatim — the behaviour the table must reproduce."""
    if leg in ABLATION_LEG_NAMES:
        return [
            f
            for f in ("corpus.parquet", CONTENT_ONLY_VECTORS)
            if not (export_dir / f).is_file()
        ]
    if leg in RESCUE_LEG_NAMES:
        return [
            f
            for f in (
                "corpus.parquet",
                "bm25-stats.json",
                "bm25-postings.parquet",
                "graph-edges.parquet",
                "graph-nodes.parquet",
                "graph-hyperedges.kv",
                "graph-hyperedges.parquet",
            )
            if not (export_dir / f).is_file()
        ]
    needs: dict[str, tuple[str, ...]] = {
        "dense": ("corpus.parquet",),
        "bm25": ("bm25-stats.json", "bm25-postings.parquet"),
        "kg": ("graph-edges.parquet", "graph-nodes.parquet"),
        "hyperedges": ("graph-hyperedges.kv", "graph-hyperedges.parquet"),
        "coe": ("graph-hyperedges.kv",),
        "packer": (),
    }
    cumulative: list[str] = []
    for name in LEG_NAMES:
        cumulative.extend(needs[name])
        if name == leg:
            break
    return [f for f in dict.fromkeys(cumulative) if not (export_dir / f).is_file()]


def _populate(export_dir: Path, files: tuple[str, ...]) -> None:
    for name in files:
        (export_dir / name).write_text("x")


class TestMissingInputsMatchesTheReplacedDispatch:
    @pytest.mark.parametrize("leg", ALL_LEG_NAMES)
    @pytest.mark.parametrize("absent", [*_ALL_FILES, CONTENT_ONLY_VECTORS])
    def test_each_single_absence(self, tmp_path: Path, leg: str, absent: str) -> None:
        universe = tuple(dict.fromkeys((*_ALL_FILES, CONTENT_ONLY_VECTORS)))
        _populate(tmp_path, tuple(f for f in universe if f != absent))
        assert missing_inputs(tmp_path, leg) == _legacy_missing_inputs(tmp_path, leg)

    @pytest.mark.parametrize("leg", ALL_LEG_NAMES)
    def test_everything_absent_states_the_full_chain_in_verbatim_order(
        self, tmp_path: Path, leg: str
    ) -> None:
        # The order of this list appears byte-for-byte in written missing-inputs cells.
        assert missing_inputs(tmp_path, leg) == _legacy_missing_inputs(tmp_path, leg)
        assert missing_inputs(tmp_path, leg) == list(LEG_TABLE[leg].required_files)

    @pytest.mark.parametrize("leg", ALL_LEG_NAMES)
    def test_nothing_absent_refuses_nothing(self, tmp_path: Path, leg: str) -> None:
        _populate(tmp_path, tuple(dict.fromkeys((*_ALL_FILES, CONTENT_ONLY_VECTORS))))
        assert missing_inputs(tmp_path, leg) == []


class TestTableConsistency:
    def test_all_leg_names_order_is_load_bearing_and_unchanged(self) -> None:
        # Run order AND report row order — ladder first, then ablations, then rescue.
        assert ALL_LEG_NAMES == (
            "dense",
            "bm25",
            "kg",
            "hyperedges",
            "coe",
            "packer",
            "dense_content_only",
            "graph_rescue",
            "graph_rescue_packed",
            "graph_rescue_coe",
        )
        assert LEG_NAMES == ("dense", "bm25", "kg", "hyperedges", "coe", "packer")
        assert ABLATION_LEG_NAMES == ("dense_content_only",)
        assert RESCUE_LEG_NAMES == (
            "graph_rescue",
            "graph_rescue_packed",
            "graph_rescue_coe",
        )

    def test_family_cell_kind_and_report_section_cohere(self) -> None:
        for leg, spec in LEG_TABLE.items():
            assert spec.report_section == spec.family, leg
            assert spec.cell_kind == ("rescue" if spec.family == "rescue" else "fusion"), leg
            assert spec.vector_source == (
                "sidecar" if spec.family == "ablation" else "corpus"
            ), leg
            assert spec.vector_source == vector_source(leg)
            assert spec.needs_graph_features is (spec.family == "rescue"), leg

    def test_ladder_signals_are_cumulative_prefixes_of_the_fusion_order(self) -> None:
        for leg in LEG_NAMES:
            signals = leg_signals(leg)
            assert signals == FUSION_SIGNALS[: len(signals)], leg
        assert leg_signals("packer") == FUSION_SIGNALS  # fuses what coe fuses, then packs

    def test_rescue_legs_declare_the_base_pair_and_nothing_else(self) -> None:
        for leg in RESCUE_LEG_NAMES:
            assert leg_signals(leg) == ("dense", "bm25"), leg

    def test_tunable_matches_the_replaced_predicate(self) -> None:
        for leg, spec in LEG_TABLE.items():
            legacy = leg not in ABLATION_LEG_NAMES and len(leg_signals(leg)) > 1
            assert spec.tunable is legacy, leg
            assert is_tunable_leg(leg) is legacy, leg
