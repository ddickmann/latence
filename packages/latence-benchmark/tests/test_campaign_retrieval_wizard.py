# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The retrieval wizard block: measured defaults, contradictions, determinism, backward compat.

Plus the block's CONSUMER — :func:`latence_benchmark.campaign.retrieval_lib.rescue_config`, the
one wizard→component mapping — pinned against the real ``GraphRescueRetrieval`` (imported here,
never in the shipped campaign package, which stays import-independent of latence-retrieval).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_benchmark.campaign import cli
from latence_benchmark.campaign.retrieval_lib import render_rescue_config, rescue_config
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


class TestRescueConfig:
    """The one wizard→component mapping, pinned to the component it claims to configure."""

    def test_every_emitted_key_is_a_real_graph_rescue_config_key(self) -> None:
        # The pin: the guide's "maps one-to-one onto GraphRescueRetrieval's config" is CHECKED
        # here, not asserted in prose. A rename on either side (a wizard field, or a key in
        # rescue._CONFIG_KEYS) fails this test instead of surfacing at query time as
        # "unknown config key(s)" on a pod run.
        from latence_retrieval import rescue

        keys: set[str] = set()
        for selector in RETRIEVAL_SELECTORS:
            for mode in RETRIEVAL_GRAPH_MODES:
                keys |= set(rescue_config(RetrievalAnswers(selector=selector, graph_mode=mode)))
        assert keys <= set(rescue._CONFIG_KEYS)  # noqa: SLF001

    def test_the_defaults_construct_the_production_component(self) -> None:
        # Keys being legal is not enough — the VALUES must survive the component's own
        # validation (positive ints, unit floats, known choices), so the mapping is exercised
        # through a real construction, not a membership test.
        from latence_retrieval.contracts import Capability
        from latence_retrieval.rescue import GraphRescueRetrieval
        from latence_retrieval.testing import FakeBackend

        pipeline = GraphRescueRetrieval(
            rescue_config(RetrievalAnswers()),
            bm25=FakeBackend([], {Capability.BM25}),
        )
        assert pipeline is not None

    def test_the_budget_follows_the_selector_in_force(self) -> None:
        # The block carries BOTH per-selector budgets (each at its measured operating point);
        # the component takes ONE. 'none' selects nothing, so it gets no budget at all.
        assert rescue_config(RetrievalAnswers())["budget"] == 10_000
        assert rescue_config(RetrievalAnswers(selector="coe"))["budget"] == 16_000
        assert "budget" not in rescue_config(RetrievalAnswers(selector="none"))

    def test_the_answers_reach_the_config_verbatim(self) -> None:
        answers = RetrievalAnswers(
            graph_mode="force",
            packed_budget=4_000,
            seed_top_k=9,
            hop_weight=0.2,
            community_weight=0.02,
            centrality_weight=0.03,
            seed_docs_k=7,
            seed_node_cap=64,
            rescue_cap=25,
            pack_centrality_weight=0.5,
        )
        assert rescue_config(answers) == {
            "graph_mode": "force",
            "selector": "packed",
            "budget": 4_000,
            "seed_top_k": 9,
            "hop_weight": 0.2,
            "community_weight": 0.02,
            "centrality_weight": 0.03,
            "seed_docs_k": 7,
            "seed_node_cap": 64,
            "rescue_cap": 25,
            "pack_centrality_weight": 0.5,
        }

    def test_the_render_is_deterministic_and_headed_by_its_own_values(self) -> None:
        text = render_rescue_config(RetrievalAnswers())
        assert text == render_rescue_config(RetrievalAnswers())
        header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
        assert "selector packed @ 10000 tok | graph auto" in header
        body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
        assert "budget: 10000" in body
        assert "pack_centrality_weight: 0.25" in body

    def test_the_selector_none_render_says_so(self) -> None:
        text = render_rescue_config(RetrievalAnswers(selector="none"))
        assert "ranked union, no budget" in text
        assert "budget:" not in text


class TestRetrievalConfigCommand:
    """The CLI consumer: `latence-campaign retrieval-config` renders the block's real config."""

    def test_it_renders_the_config_of_the_configs_retrieval_block(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "campaign.yaml"
        config.write_text(
            build_campaign_config(
                CampaignAnswers(
                    campaign_root=str(tmp_path),
                    datasets=("musique",),
                    retrieval=RetrievalAnswers(graph_mode="force", rescue_cap=11),
                )
            ),
            encoding="utf-8",
        )
        rc = cli.main(["retrieval-config", "-c", str(config)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "graph_mode: force" in out
        assert "rescue_cap: 11" in out
        assert out == render_rescue_config(RetrievalAnswers(graph_mode="force", rescue_cap=11))

    def test_a_missing_config_is_an_invalid_artifact_exit_1(self, tmp_path: Path) -> None:
        rc = cli.main(["retrieval-config", "-c", str(tmp_path / "nope.yaml")])
        assert rc == 1

    def test_a_typoed_knob_is_the_same_loud_parse_error_here(self, tmp_path: Path) -> None:
        config = tmp_path / "campaign.yaml"
        config.write_text(
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "retrieval:\n  select:\n    packed_budgett: 4000\n",
            encoding="utf-8",
        )
        assert cli.main(["retrieval-config", "-c", str(config)]) == 1


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
