# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The campaign wizard: byte-determinism, contradiction rejection, extensible parsing."""

from __future__ import annotations

import pytest
from latence_benchmark.campaign.wizard import (
    KNOWN_DATASETS,
    CampaignAnswers,
    CampaignConfigError,
    build_campaign_config,
    parse_campaign_config,
)


class TestAnswers:
    def test_defaults_select_all_known_datasets(self) -> None:
        answers = CampaignAnswers(campaign_root="/tmp/c")
        assert answers.datasets == KNOWN_DATASETS
        assert answers.pod_ssh == ""

    def test_empty_root_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="campaign_root"):
            CampaignAnswers(campaign_root="   ")

    def test_empty_selection_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="empty"):
            CampaignAnswers(campaign_root="/tmp/c", datasets=())

    def test_unknown_dataset_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="beir"):
            CampaignAnswers(campaign_root="/tmp/c", datasets=("musique", "beir"))

    def test_duplicate_dataset_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="more than once"):
            CampaignAnswers(campaign_root="/tmp/c", datasets=("uda", "musique", "uda"))

    def test_operator_order_is_preserved(self) -> None:
        answers = CampaignAnswers(campaign_root="/tmp/c", datasets=("uda", "musique"))
        assert answers.datasets == ("uda", "musique")


class TestGeneratorDeterminism:
    def test_identical_answers_yield_byte_identical_yaml(self) -> None:
        a = CampaignAnswers(campaign_root="/c", datasets=("musique", "uda"), pod_ssh="x@y")
        b = CampaignAnswers(campaign_root="/c", datasets=("musique", "uda"), pod_ssh="x@y")
        assert build_campaign_config(a) == build_campaign_config(b)
        assert build_campaign_config(a) == build_campaign_config(a)

    def test_different_answers_yield_different_yaml(self) -> None:
        a = CampaignAnswers(campaign_root="/c", datasets=("musique",))
        b = CampaignAnswers(campaign_root="/c", datasets=("uda",))
        assert build_campaign_config(a) != build_campaign_config(b)

    def test_header_is_derived_from_the_config(self) -> None:
        answers = CampaignAnswers(
            campaign_root="/camp", datasets=("musique", "ohr_bench"), pod_ssh="root@pod"
        )
        text = build_campaign_config(answers)
        header = [line for line in text.splitlines() if line.startswith("#")]
        joined = "\n".join(header)
        assert "/camp" in joined
        assert "2 dataset(s)" in joined
        assert "musique, ohr_bench" in joined
        assert "root@pod" in joined

    def test_no_pod_reads_as_local_only(self) -> None:
        text = build_campaign_config(CampaignAnswers(campaign_root="/c"))
        assert "none (local-only" in text


class TestParsing:
    def test_roundtrip(self) -> None:
        answers = CampaignAnswers(
            campaign_root="/camp", datasets=("hotpotqa", "vidoseek"), pod_ssh="a@b:2222"
        )
        assert parse_campaign_config(build_campaign_config(answers)) == answers

    def test_unknown_top_level_blocks_are_accepted(self) -> None:
        # Extensibility contract: an S2-era config (extra `ocr:` block) still loads here.
        text = build_campaign_config(CampaignAnswers(campaign_root="/c"))
        text += "ocr:\n  base_url: http://127.0.0.1:8000/v1\n"
        parsed = parse_campaign_config(text)
        assert parsed.campaign_root == "/c"

    def test_invalid_yaml_is_a_config_error(self) -> None:
        with pytest.raises(CampaignConfigError, match="not valid YAML"):
            parse_campaign_config("campaign: [unclosed")

    def test_missing_campaign_block_is_loud(self) -> None:
        with pytest.raises(CampaignConfigError, match="campaign"):
            parse_campaign_config("corpus:\n  datasets: [musique]\n")

    def test_missing_corpus_block_is_loud(self) -> None:
        with pytest.raises(CampaignConfigError, match="corpus"):
            parse_campaign_config("campaign:\n  root: /c\n")

    def test_unknown_dataset_in_file_is_rejected_at_parse(self) -> None:
        text = "campaign:\n  root: /c\n  pod_ssh: ''\ncorpus:\n  datasets: [made_up]\n"
        with pytest.raises(CampaignConfigError, match="made_up"):
            parse_campaign_config(text)
