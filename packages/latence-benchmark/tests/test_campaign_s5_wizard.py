# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The ``s5:`` wizard block: the full-matrix default, contradictions, determinism, the mirrors.

Same discipline as the ``s3:`` and ``retrieval:`` blocks, plus the two lockstep pins the block
needs and the campaign package cannot get by importing: its leg vocabulary and its fusion
vocabulary are the S5 harness's, and the harness is repo code, not a dependency. Importing both
is free HERE, so the mirror is proven from the test side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from latence_benchmark.campaign import cli
from latence_benchmark.campaign.wizard import (
    KNOWN_DATASETS,
    S5_FUSIONS,
    S5_LEGS,
    CampaignAnswers,
    CampaignConfigError,
    S5Answers,
    build_campaign_config,
    parse_campaign_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_REPO_ROOT))


def answers(**kwargs: object) -> CampaignAnswers:
    return CampaignAnswers(campaign_root="/tmp/camp", s5=S5Answers(**kwargs))  # type: ignore[arg-type]


class TestDefaults:
    def test_the_default_is_the_full_matrix_over_the_corpus_selection(self) -> None:
        s5 = S5Answers()
        assert s5.datasets == ()  # empty = follow corpus.datasets
        assert s5.legs == S5_LEGS
        assert s5.limit is None  # the full gold sets, not a smoke sample
        assert s5.fusion == "equal"

    def test_the_vocabularies_mirror_the_harness(self) -> None:
        # Kept in lockstep with benchmark/s5 (the campaign package configures the harness and
        # deliberately does not import it — numpy/pyarrow/latence-retrieval stay out).
        #
        # `benchmark.s5.legs` imports numpy at module scope, and the CPU-first dev env is
        # deliberately numpy-free (ADR-0016): a pristine `uv sync` — which is exactly what
        # `.github/actions/setup-workspace` produces — has no numpy, so this import raised
        # ModuleNotFoundError in CI while passing on any developer machine that had picked
        # numpy up transitively. Skipped-with-a-reason where the harness is not installable;
        # the lockstep assertion still runs wherever it is.
        pytest.importorskip(
            "numpy",
            reason="benchmark/s5 imports numpy; the CPU-first dev env is numpy-free (ADR-0016)",
        )
        from benchmark.s5.legs import ALL_LEG_NAMES
        from benchmark.s5.run_matrix import FUSIONS

        assert S5_LEGS == ALL_LEG_NAMES  # order included: it is the report's column order
        assert S5_FUSIONS == FUSIONS

    def test_the_dataset_vocabulary_is_the_harness_table(self) -> None:
        from benchmark.s5.datasets import DATASET_NAMES

        # Same set, so `s5.datasets` validated against KNOWN_DATASETS can never name a
        # dataset the matrix has no spec for. (Order differs: KNOWN_DATASETS is campaign
        # order, DATASET_NAMES is matrix order — neither is the other's business.)
        assert set(S5Answers().datasets) <= set(KNOWN_DATASETS)
        assert set(KNOWN_DATASETS) == set(DATASET_NAMES)


class TestContradictions:
    def test_an_unknown_dataset_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="s5.datasets has unknown"):
            S5Answers(datasets=("nasa",))

    def test_a_repeated_dataset_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="s5.datasets lists musique"):
            S5Answers(datasets=("musique", "musique"))

    def test_an_unknown_leg_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="s5.legs has unknown"):
            S5Answers(legs=("dense", "telepathy"))

    def test_a_repeated_leg_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="s5.legs lists dense"):
            S5Answers(legs=("dense", "dense"))

    def test_an_empty_leg_set_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="measures nothing"):
            S5Answers(legs=())

    def test_a_non_positive_limit_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="s5.limit"):
            S5Answers(limit=0)

    def test_an_unknown_fusion_is_rejected(self) -> None:
        with pytest.raises(CampaignConfigError, match="equal, tuned"):
            S5Answers(fusion="magic")


class TestRoundTrip:
    def test_defaults_round_trip_byte_identically(self) -> None:
        original = answers()
        text = build_campaign_config(original)
        assert parse_campaign_config(text) == original
        assert build_campaign_config(parse_campaign_config(text)) == text

    def test_a_narrowed_matrix_round_trips(self) -> None:
        original = answers(
            datasets=("vidoseek", "multihop_rag"),
            legs=("dense", "graph_rescue_packed"),
            limit=50,
            fusion="tuned",
        )
        text = build_campaign_config(original)
        assert "limit: 50" in text
        assert parse_campaign_config(text).s5 == original.s5

    def test_the_header_states_the_matrix(self) -> None:
        text = build_campaign_config(answers())
        assert "# s5 matrix     : 7 dataset(s) (the corpus selection) × 10 leg(s); " in text
        assert "full query sets; fusion equal" in text
        narrowed = build_campaign_config(answers(datasets=("uda",), limit=8, fusion="tuned"))
        assert "1 dataset(s) — uda" in narrowed
        assert "limit 8 queries/dataset; fusion tuned" in narrowed

    def test_a_config_without_an_s5_block_still_loads(self) -> None:
        text = "campaign:\n  root: /tmp/c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
        assert parse_campaign_config(text).s5 == S5Answers()

    def test_an_explicit_empty_datasets_list_means_the_corpus_selection(self) -> None:
        text = (
            "campaign:\n  root: /tmp/c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
            "s5:\n  datasets: []\n"
        )
        assert parse_campaign_config(text).s5.datasets == ()

    def test_an_unknown_s5_key_is_loud(self) -> None:
        text = (
            "campaign:\n  root: /tmp/c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
            "s5:\n  legz: [dense]\n"
        )
        with pytest.raises(CampaignConfigError, match="`s5` has unknown key"):
            parse_campaign_config(text)

    def test_a_wrongly_typed_limit_is_rejected_at_parse(self) -> None:
        text = (
            "campaign:\n  root: /tmp/c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
            "s5:\n  limit: 'lots'\n"
        )
        with pytest.raises(CampaignConfigError, match="`s5.limit`"):
            parse_campaign_config(text)

    def test_a_wrongly_typed_leg_list_is_rejected_at_parse(self) -> None:
        text = (
            "campaign:\n  root: /tmp/c\n  pod_ssh: ''\ncorpus:\n  datasets: [musique]\n"
            "s5:\n  legs: dense\n"
        )
        with pytest.raises(CampaignConfigError, match="`s5.legs` must be a list"):
            parse_campaign_config(text)


class TestInitFlags:
    def _init(self, tmp_path: Path, *flags: str) -> tuple[int, Path]:
        out = tmp_path / "campaign.yaml"
        rc = cli.main(["init", "--non-interactive", "-o", str(out), *flags])
        return rc, out

    def test_non_interactive_init_is_byte_deterministic(self, tmp_path: Path) -> None:
        # The config is an artifact, not a session: identical answers, identical bytes.
        first_rc, first = self._init(tmp_path / "a", "--campaign-root", "benchmark/x")
        second_rc, second = self._init(tmp_path / "b", "--campaign-root", "benchmark/x")
        assert (first_rc, second_rc) == (0, 0)
        assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")

    def test_flags_reach_the_block(self, tmp_path: Path) -> None:
        rc, out = self._init(
            tmp_path,
            "--s5-datasets",
            "uda,vidoseek",
            "--s5-legs",
            "packer,dense",
            "--s5-limit",
            "25",
            "--s5-fusion",
            "tuned",
        )
        assert rc == 0
        s5 = parse_campaign_config(out.read_text(encoding="utf-8")).s5
        assert s5.datasets == ("uda", "vidoseek")
        assert s5.legs == ("packer", "dense")  # verbatim here; the stage canonicalises order
        assert (s5.limit, s5.fusion) == (25, "tuned")

    def test_an_explicit_empty_datasets_flag_is_the_corpus_selection(
        self, tmp_path: Path
    ) -> None:
        rc, out = self._init(tmp_path, "--s5-datasets", "")
        assert rc == 0
        assert parse_campaign_config(out.read_text(encoding="utf-8")).s5.datasets == ()

    @pytest.mark.parametrize(
        ("flag", "value", "needle"),
        [
            ("--s5-legs", "", "measures nothing"),  # NOT silently the full ladder
            ("--s5-limit", "0", "s5.limit"),
            ("--s5-limit", "", "invalid literal"),  # int('') is loud, never a default
            ("--s5-datasets", "nasa", "s5.datasets has unknown"),
            ("--s5-legs", "telepathy", "s5.legs has unknown"),
        ],
    )
    def test_an_explicit_bad_value_is_loud_not_defaulted(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        flag: str,
        value: str,
        needle: str,
    ) -> None:
        rc, _out = self._init(tmp_path, flag, value)
        assert rc == 2
        assert needle in capsys.readouterr().err
