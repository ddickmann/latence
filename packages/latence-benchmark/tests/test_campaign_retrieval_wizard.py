# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The retrieval wizard block: measured defaults, contradictions, determinism, backward compat."""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_benchmark.campaign import cli
from latence_benchmark.campaign.wizard import (
    RETRIEVAL_GRAPH_MODES,
    RETRIEVAL_SELECTORS,
    CampaignAnswers,
    CampaignConfigError,
    RetrievalAnswers,
    build_campaign_config,
    parse_campaign_config,
)


class TestRetrievalDefaults:
    def test_the_measured_s5b_operating_point_is_the_default(self) -> None:
        r = RetrievalAnswers()
        assert r.graph_mode == "auto"
        assert r.selector == "packed"
        # Per-selector headline budgets (operator ruling, s5b): the knapsack saturates ~5.5k
        # so 10k; CoE keeps buying recall through 16k.
        assert (r.packed_budget, r.coe_budget) == (10_000, 16_000)
        assert r.seed_top_k == 5
        assert (r.hop_weight, r.community_weight, r.centrality_weight) == (0.05, 0.01, 0.05)
        assert (r.seed_docs_k, r.seed_node_cap, r.rescue_cap) == (5, 32, 50)
        assert r.pack_centrality_weight == 0.25

    def test_the_choice_sets_mirror_the_production_component(self) -> None:
        # Kept in lockstep with latence_retrieval.rescue (this package deliberately does not
        # import latence-retrieval, so the mirror is pinned here instead).
        from latence_retrieval.rescue import GRAPH_MODES, SELECTORS

        assert RETRIEVAL_GRAPH_MODES == GRAPH_MODES
        assert RETRIEVAL_SELECTORS == SELECTORS


class TestRetrievalContradictions:
    def test_an_unknown_graph_mode_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="off, auto, force"):
            RetrievalAnswers(graph_mode="always")

    def test_an_unknown_selector_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="packed, coe, none"):
            RetrievalAnswers(selector="knapsack")

    def test_a_non_positive_budget_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="packed_budget"):
            RetrievalAnswers(packed_budget=0)
        with pytest.raises(CampaignConfigError, match="coe_budget"):
            RetrievalAnswers(coe_budget=-1)

    def test_a_negative_expander_weight_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="hop_weight"):
            RetrievalAnswers(hop_weight=-0.1)

    def test_a_non_positive_cap_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="rescue_cap"):
            RetrievalAnswers(rescue_cap=0)


class TestGeneratorAndParser:
    def test_the_retrieval_block_round_trips(self) -> None:
        answers = CampaignAnswers(
            campaign_root="/camp",
            datasets=("musique",),
            retrieval=RetrievalAnswers(
                graph_mode="force",
                selector="coe",
                coe_budget=24_000,
                hop_weight=0.1,
                seed_node_cap=16,
            ),
        )
        assert parse_campaign_config(build_campaign_config(answers)) == answers

    def test_identical_answers_yield_byte_identical_yaml(self) -> None:
        a = CampaignAnswers(campaign_root="/c", retrieval=RetrievalAnswers(selector="none"))
        b = CampaignAnswers(campaign_root="/c", retrieval=RetrievalAnswers(selector="none"))
        assert build_campaign_config(a) == build_campaign_config(b)

    def test_the_header_carries_the_retrieval_headlines(self) -> None:
        text = build_campaign_config(CampaignAnswers(campaign_root="/c"))
        header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
        assert "graph auto" in header
        assert "packed @ 10000 tok" in header
        assert "5/0.05/0.01/0.05" in header
        assert "graph appends, never fuses" in header

    def test_the_coe_selector_headline_names_its_own_budget(self) -> None:
        text = build_campaign_config(
            CampaignAnswers(campaign_root="/c", retrieval=RetrievalAnswers(selector="coe"))
        )
        header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
        assert "coe @ 16000 tok" in header

    def test_a_pre_s5b_config_without_the_block_parses_to_the_measured_defaults(self) -> None:
        text = "campaign:\n  root: /c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
        parsed = parse_campaign_config(text)
        assert parsed.retrieval == RetrievalAnswers()

    def test_a_partial_block_fills_the_rest_with_defaults(self) -> None:
        # 'off' is quoted: bare `off` is a YAML 1.1 boolean, and the parser rightly rejects a
        # bool where a mode string belongs.
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "retrieval:\n  graph:\n    mode: 'off'\n"
        )
        parsed = parse_campaign_config(text)
        assert parsed.retrieval.graph_mode == "off"
        assert parsed.retrieval.packed_budget == 10_000  # untouched default

    def test_a_typoed_retrieval_key_is_loud_not_silently_defaulted(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "retrieval:\n  select:\n    packed_budgett: 4000\n"
        )
        with pytest.raises(CampaignConfigError, match="packed_budgett"):
            parse_campaign_config(text)

    def test_a_wrongly_typed_retrieval_value_is_rejected(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "retrieval:\n  expand:\n    hop_weight: heavy\n"
        )
        with pytest.raises(CampaignConfigError, match="hop_weight"):
            parse_campaign_config(text)

    def test_contradictions_in_the_file_are_rejected_at_parse(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "retrieval:\n  graph:\n    mode: sometimes\n"
        )
        with pytest.raises(CampaignConfigError, match="off, auto, force"):
            parse_campaign_config(text)


class TestCliFlags:
    def test_the_retrieval_flags_land_in_the_config(self, tmp_path: Path) -> None:
        out = tmp_path / "campaign.yaml"
        rc = cli.main(
            [
                "init",
                "--non-interactive",
                "--campaign-root",
                "/camp",
                "--datasets",
                "musique",
                "--retrieval-graph-mode",
                "force",
                "--retrieval-selector",
                "coe",
                "--retrieval-coe-budget",
                "24000",
                "--retrieval-hop-weight",
                "0.1",
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        parsed = parse_campaign_config(out.read_text())
        assert parsed.retrieval.graph_mode == "force"
        assert parsed.retrieval.selector == "coe"
        assert parsed.retrieval.coe_budget == 24_000
        assert parsed.retrieval.hop_weight == 0.1

    def test_a_contradictory_retrieval_flag_is_usage_error_exit_2(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "campaign.yaml"
        rc = cli.main(
            [
                "init",
                "--non-interactive",
                "--campaign-root",
                "/camp",
                "--datasets",
                "musique",
                "--retrieval-packed-budget",
                "0",
                "-o",
                str(out),
            ]
        )
        assert rc == 2
        assert not out.exists()
