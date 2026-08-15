# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The s3 wizard block: defaults, contradictions, byte-determinism, backward compat."""

from __future__ import annotations

import pytest
from latence_benchmark.campaign.wizard import (
    S3_ENTITY_LABELS,
    S3_RELATION_LABELS,
    CampaignAnswers,
    CampaignConfigError,
    S3Answers,
    build_campaign_config,
    parse_campaign_config,
)


class TestS3Defaults:
    def test_the_spec_rulings_are_the_defaults(self) -> None:
        s3 = S3Answers()
        assert (s3.chunk_max_tokens, s3.chunk_overlap_tokens, s3.chunk_min_tokens) == (
            640, 80, 8,
        )
        assert s3.extraction_provider == "fused_entity_relation.gliner2"
        assert s3.extraction_model == "fastino/gliner2-multi-v1"
        assert s3.extraction_threshold == 0.5
        assert s3.induction_mode == "per_chunk"
        # Dennis's 2026-08 ruling off the live A/B: deepseek-v4-flash with reasoning
        # DISABLED (9/10 valid schemas at ~$0.000118/call; reasoning-on burned the whole
        # completion budget on reasoning tokens — 7/10 invalid).
        assert s3.induction_model == "deepseek/deepseek-v4-flash-0731"
        assert s3.induction_reasoning is False
        assert s3.induction_base_url == "https://openrouter.ai/api/v1"
        assert s3.induction_budget_usd_cap == 150.0
        assert s3.redaction_provider == "redaction.gliner2"
        assert s3.redaction_guardrails is True

    def test_the_hand_authored_label_sets_are_the_ticket_schema(self) -> None:
        assert S3_ENTITY_LABELS == (
            "PERSON", "ORGANIZATION", "LOCATION", "WORK", "EVENT", "DATE", "PRODUCT",
        )
        assert S3_RELATION_LABELS == (
            "located_in", "part_of", "member_of", "created_by", "founded_by",
            "spouse_of", "directed_by", "works_for", "released_in",
        )


class TestS3Contradictions:
    def test_overlap_swallowing_the_chunk_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="overlap"):
            S3Answers(chunk_max_tokens=100, chunk_overlap_tokens=100)

    def test_min_tokens_above_max_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="min_tokens"):
            S3Answers(chunk_max_tokens=100, chunk_min_tokens=101)

    def test_threshold_out_of_range_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="threshold"):
            S3Answers(extraction_threshold=1.5)

    def test_unknown_induction_mode_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="per_chunk, doc, none"):
            S3Answers(induction_mode="per_document")

    def test_zero_budget_with_induction_on_is_a_contradiction(self) -> None:
        with pytest.raises(CampaignConfigError, match="contradiction"):
            S3Answers(induction_budget_usd_cap=0.0)

    def test_zero_budget_with_mode_none_is_fine(self) -> None:
        s3 = S3Answers(induction_mode="none", induction_budget_usd_cap=0.0)
        assert s3.induction_mode == "none"

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="budget_usd_cap"):
            S3Answers(induction_budget_usd_cap=-1.0)

    def test_empty_label_sets_are_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="labels.entity"):
            S3Answers(entity_labels=())
        with pytest.raises(CampaignConfigError, match="labels.relation"):
            S3Answers(relation_labels=("", " "))


class TestGeneratorAndParser:
    def test_s3_block_round_trips(self) -> None:
        answers = CampaignAnswers(
            campaign_root="/camp",
            datasets=("musique", "ohr_bench"),
            s3=S3Answers(
                induction_mode="doc",
                induction_budget_usd_cap=42.5,
                induction_max_concurrency=16,
                entity_labels=("PERSON",),
                relation_labels=("works_for",),
                redaction_guardrails=False,
            ),
        )
        assert parse_campaign_config(build_campaign_config(answers)) == answers

    def test_identical_answers_yield_byte_identical_yaml(self) -> None:
        a = CampaignAnswers(campaign_root="/c", s3=S3Answers(induction_mode="none"))
        b = CampaignAnswers(campaign_root="/c", s3=S3Answers(induction_mode="none"))
        assert build_campaign_config(a) == build_campaign_config(b)

    def test_header_carries_the_s3_headlines(self) -> None:
        text = build_campaign_config(CampaignAnswers(campaign_root="/c"))
        header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
        assert "chunk 640/80/8" in header
        assert "per_chunk" in header
        assert "$150.00" in header
        assert "guardrails on" in header

    def test_reasoning_round_trips_in_all_three_states(self) -> None:
        for reasoning in (True, False, None):
            answers = CampaignAnswers(
                campaign_root="/c", s3=S3Answers(induction_reasoning=reasoning)
            )
            parsed = parse_campaign_config(build_campaign_config(answers))
            assert parsed.s3.induction_reasoning is reasoning

    def test_a_non_bool_reasoning_value_is_rejected(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "s3:\n  induction:\n    reasoning: 'false'\n"
        )
        with pytest.raises(CampaignConfigError, match="reasoning"):
            parse_campaign_config(text)

    def test_an_s1_era_config_without_s3_parses_to_the_defaults(self) -> None:
        # Backward compat: the operator's existing campaign.yaml predates the s3 block.
        text = "campaign:\n  root: /c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
        parsed = parse_campaign_config(text)
        assert parsed.s3 == S3Answers()

    def test_a_partial_s3_block_fills_the_rest_with_defaults(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "s3:\n  induction:\n    mode: none\n    budget_usd_cap: 0\n"
        )
        parsed = parse_campaign_config(text)
        assert parsed.s3.induction_mode == "none"
        assert parsed.s3.chunk_max_tokens == 640  # untouched default

    def test_a_typoed_s3_key_is_loud_not_silently_defaulted(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "s3:\n  induction:\n    budget_usd_gap: 10\n"
        )
        with pytest.raises(CampaignConfigError, match="budget_usd_gap"):
            parse_campaign_config(text)

    def test_a_wrongly_typed_s3_value_is_rejected(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "s3:\n  chunk:\n    max_tokens: many\n"
        )
        with pytest.raises(CampaignConfigError, match="max_tokens"):
            parse_campaign_config(text)

    def test_contradictions_in_the_file_are_rejected_at_parse(self) -> None:
        text = (
            "campaign:\n  root: /c\n  pod_ssh: ''\n"
            "corpus:\n  datasets: [musique]\n"
            "s3:\n  chunk:\n    max_tokens: 64\n    overlap_tokens: 64\n"
        )
        with pytest.raises(CampaignConfigError, match="overlap"):
            parse_campaign_config(text)
