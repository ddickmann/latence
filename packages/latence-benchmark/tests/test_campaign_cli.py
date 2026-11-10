# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""``latence-campaign`` end to end (offline): init parity, status, run/approve/resume flow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.campaign import cli, corpus_stage, pipeline_stage
from latence_benchmark.campaign.corpus_stage import (
    KIND_HF_SNAPSHOT,
    KIND_URL,
    DatasetPlan,
    FetchResult,
)
from latence_benchmark.campaign.state import STATE_FILENAME, CampaignState
from latence_benchmark.campaign.wizard import KNOWN_DATASETS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from latence_benchmark.campaign.corpus_stage import Fetcher

REPO_ROOT = Path(__file__).resolve().parents[3]

#: A quality report in the shape the s3 stage reads: non-empty, with a graph.
POD_REPORT = {
    "document_count": 2,
    "stages": [],
    "chunk": {"chunks_total": 4, "chunks_flagged": 0},
    "entities": {"mentions_total": 6},
    "relations": {"relations_total": 2},
    "redaction": {"pii_total": 0, "per_type": {}},
    "graph": {"nodes_total": 5, "edges_total": 2},
}


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    dst = repo / "benchmark" / "corpus" / "manifest.py"
    dst.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "benchmark" / "corpus" / "manifest.py", dst)
    return repo


def init_campaign(
    tmp_path: Path, datasets: str = "multihop_rag", pod_ssh: str = ""
) -> Path:
    config = tmp_path / "camp" / "campaign.yaml"
    argv = [
        "init",
        "--non-interactive",
        "--campaign-root",
        str(tmp_path / "camp"),
        "--datasets",
        datasets,
        "-o",
        str(config),
    ]
    if pod_ssh:
        argv += ["--pod-ssh", pod_ssh]
    rc = cli.main(argv)
    assert rc == 0
    return config


def _fake_fetch(plan: DatasetPlan, dest: Path) -> FetchResult:
    """Land the one raw file the multihop_rag prep reads; skip it when already there."""
    target = dest / "corpus.json"
    if target.exists():
        return FetchResult(fetched=0, skipped=1)
    dest.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [
                {"title": "Alpha buys Beta", "body": "Alpha acquired Beta in 2023."},
                {"title": "Gamma", "body": "Gamma partnered with Delta."},
            ]
        ),
        encoding="utf-8",
    )
    return FetchResult(fetched=1, skipped=0)


class FinishedPod:
    """A pod whose run already finished: the status probe answers ``done``.

    Built by the stage with the configured ``pod_ssh``, like the real transport, so the CLI
    flow reaches the stage's pull, floor check and metrics without ssh.
    """

    def __init__(self, pod_ssh: str) -> None:
        self.pod_ssh = pod_ssh

    def run(self, cmd: str) -> tuple[int, str]:
        if "du -sk" in cmd:
            return 0, "done 9"
        if cmd.startswith("find "):
            return 0, "/remote/export/corpus.jsonl\n"
        return 0, ""

    def push(self, local: Path, remote: str) -> None:
        raise AssertionError(f"a finished run must not be pushed again: {local} -> {remote}")

    def pull(self, remote: str, local: Path) -> None:
        if remote.endswith("quality-report.json"):
            local.write_text(json.dumps(POD_REPORT), encoding="utf-8")
        elif remote.endswith("quality-report.md"):
            local.write_text("# Quality Report\n", encoding="utf-8")


@pytest.fixture
def offline_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the two network seams the CLI would otherwise build for itself.

    Every dataset in the registry is fetched from the Hub by s1 and run on the pod by s3,
    so the CLI flow needs both seams faked to stay offline. The stages' own behaviour behind
    the seams is covered by the stage suites; what runs here is the real CLI, the real stage
    registry and the real stage functions.
    """

    def fetchers() -> Mapping[str, Fetcher]:
        return {KIND_HF_SNAPSHOT: _fake_fetch, KIND_URL: _fake_fetch}

    monkeypatch.setattr(corpus_stage, "default_fetchers", fetchers)
    monkeypatch.setattr(pipeline_stage, "SubprocessTransport", FinishedPod)


class TestInit:
    def test_non_interactive_writes_a_validated_config(self, tmp_path: Path) -> None:
        config = init_campaign(tmp_path)
        text = config.read_text()
        assert "campaign:" in text
        assert "- multihop_rag" in text

    def test_byte_determinism_across_runs(self, tmp_path: Path) -> None:
        a = init_campaign(tmp_path / "a", datasets="multihop_rag,uda")
        b = init_campaign(tmp_path / "b", datasets="multihop_rag,uda")
        # Different roots differ only in the root line; same answers = same bytes.
        fixed_a = a.read_text().replace(str(tmp_path / "a" / "camp"), "ROOT")
        fixed_b = b.read_text().replace(str(tmp_path / "b" / "camp"), "ROOT")
        assert fixed_a == fixed_b

    def test_interactive_and_flags_produce_identical_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        by_flags = tmp_path / "flags.yaml"
        rc = cli.main(
            [
                "init",
                "--non-interactive",
                "--campaign-root",
                "/camp",
                "--datasets",
                "multihop_rag,vidoseek",
                "--pod-ssh",
                "root@pod",
                "-o",
                str(by_flags),
            ]
        )
        assert rc == 0

        # The S3 headline prompts (induction mode, budget cap) and the graph-rescue headline
        # prompts (policy gate, selector) answer with their defaults.
        answers: Iterator[str] = iter(
            ["/camp", "multihop_rag,vidoseek", "root@pod", "per_chunk", "150", "auto", "packed"]
        )
        monkeypatch.setattr("builtins.input", lambda: next(answers))
        by_prompts = tmp_path / "prompts.yaml"
        rc = cli.main(["init", "-o", str(by_prompts)])
        assert rc == 0

        assert by_flags.read_bytes() == by_prompts.read_bytes()

    def test_prompt_defaults_apply_on_empty_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "")
        out = tmp_path / "campaign.yaml"
        rc = cli.main(["init", "-o", str(out)])
        assert rc == 0
        # The "why it matters" lines go to stderr, per the setup wizard contract.
        assert "(the" in capsys.readouterr().err
        text = out.read_text()
        for name in KNOWN_DATASETS:
            assert f"- {name}" in text
        assert "musique" not in text
        assert "hotpotqa" not in text

    def test_explicit_zero_overlap_is_honored_not_defaulted(self, tmp_path: Path) -> None:
        # 0 is a VALID overlap; an explicit falsy answer must never fall back to the default.
        out = tmp_path / "campaign.yaml"
        rc = cli.main(
            ["init", "--non-interactive", "-o", str(out), "--s3-chunk-overlap-tokens", "0"]
        )
        assert rc == 0
        assert "overlap_tokens: 0\n" in out.read_text()

    def test_unknown_dataset_is_usage_error_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cli.main(
            [
                "init",
                "--non-interactive",
                "--datasets",
                "multihop_rag,frames",
                "-o",
                str(tmp_path / "c.yaml"),
            ]
        )
        assert rc == 2
        assert "frames" in capsys.readouterr().err
        assert not (tmp_path / "c.yaml").exists()  # nothing written on refusal

    def test_summary_names_exactly_one_next_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        init_campaign(tmp_path)
        out = capsys.readouterr().out
        assert out.count("latence-campaign") == 1
        assert "status" in out


class TestExplicitFalsyFlagsReachValidation:
    """Regression: `flag or default` swallowed explicit falsy values silently.

    Every flag resolves via `is not None` now, so an explicit-but-invalid answer must reach
    the dataclass ``__post_init__`` (or the int/float cast) and fail LOUDLY with exit 2 —
    never fall back to the default and write a config.
    """

    def _refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str
    ) -> str:
        out = tmp_path / "campaign.yaml"
        rc = cli.main(["init", "--non-interactive", "-o", str(out), flag, value])
        assert rc == 2, f"{flag} {value!r} was not refused"
        assert not out.exists(), f"{flag} {value!r} still wrote a config"
        err = capsys.readouterr().err
        assert "error:" in err
        return err

    @pytest.mark.parametrize(
        ("flag", "knob"),
        [
            ("--s3-chunk-max-tokens", "max_tokens"),
            ("--s3-chunk-min-tokens", "min_tokens"),
            ("--s3-induction-max-concurrency", "max_concurrency"),
        ],
    )
    def test_explicit_zero_int_hits_post_init_rejection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, knob: str
    ) -> None:
        err = self._refused(tmp_path, capsys, flag, "0")
        assert knob in err  # the dataclass rejection, naming the knob — not a silent default

    def test_explicit_zero_float_hits_post_init_rejection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._refused(tmp_path, capsys, "--s3-extraction-threshold", "0")
        assert "threshold" in err

    @pytest.mark.parametrize(
        "flag",
        [
            "--s3-chunk-max-tokens",
            "--s3-chunk-overlap-tokens",
            "--s3-chunk-min-tokens",
            "--s3-induction-max-concurrency",
            "--s3-extraction-threshold",
        ],
    )
    def test_explicit_empty_numeric_is_loud_not_silently_defaulted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
    ) -> None:
        self._refused(tmp_path, capsys, flag, "")  # int('') / float('') -> ValueError, exit 2

    @pytest.mark.parametrize(
        ("flag", "knob"),
        [
            ("--s3-extraction-model", "extraction.model"),
            ("--s3-induction-model", "induction.model"),
            ("--s3-induction-base-url", "induction.base_url"),
        ],
    )
    def test_explicit_empty_string_hits_post_init_rejection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, knob: str
    ) -> None:
        err = self._refused(tmp_path, capsys, flag, "")
        assert knob in err

    @pytest.mark.parametrize(
        ("flag", "knob"),
        [
            ("--s3-entity-labels", "labels.entity"),
            ("--s3-relation-labels", "labels.relation"),
        ],
    )
    def test_explicit_empty_label_list_hits_post_init_rejection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, knob: str
    ) -> None:
        err = self._refused(tmp_path, capsys, flag, "")
        assert knob in err


class TestStatusAndFlow:
    def test_status_renders_the_stage_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path)
        repo = make_repo(tmp_path)
        rc = cli.main(["status", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "s1-corpus" in out
        assert "pending" in out
        assert "unsatisfied" in out
        assert "metrics.json" in out

    @pytest.mark.usefixtures("offline_seams")
    def test_run_approve_resume_flow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path, pod_ssh="root@pod")  # multihop_rag: text-native
        repo = make_repo(tmp_path)

        rc = cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "s1-corpus: done" in out
        assert "approve s1-corpus" in out  # the CLI names the next move

        state = CampaignState(tmp_path / "camp" / STATE_FILENAME, ["s1-corpus"])
        assert state.status_of("s1-corpus") == "done"

        rc = cli.main(["approve", "s1-corpus", "-c", str(config)])
        assert rc == 0
        state = CampaignState(tmp_path / "camp" / STATE_FILENAME, ["s1-corpus"])
        assert state.status_of("s1-corpus") == "approved"

        # resume now advances into s2-ocr; a text-only selection completes it as a
        # recorded no-op (nothing to OCR), still requiring its own sign-off.
        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        assert "s2-ocr: done" in capsys.readouterr().out

        rc = cli.main(["approve", "s2-ocr", "-c", str(config)])
        assert rc == 0
        capsys.readouterr()

        # resume advances into s3-pipeline, the last stage: it prepares the markdown from
        # the fetched raw file, finds the pod run finished, pulls its report and records
        # the dataset, then awaits sign-off.
        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "s3-pipeline: done" in out
        assert "1 dataset(s) processed (4 chunks)" in out
        markdown = repo / "benchmark" / "sota-corpus" / "multihop_rag" / "markdown"
        assert len(list(markdown.glob("*.md"))) == 2

        rc = cli.main(["approve", "s3-pipeline", "-c", str(config)])
        assert rc == 0
        capsys.readouterr()

        # Every stage is approved, so resume has nothing left to advance into.
        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        assert "the campaign is complete" in capsys.readouterr().out

    @pytest.mark.usefixtures("offline_seams")
    def test_rerun_of_done_stage_is_noop(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path)
        repo = make_repo(tmp_path)
        assert cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)]) == 0
        capsys.readouterr()
        assert cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)]) == 0
        assert "no-op" in capsys.readouterr().out

    def test_approve_before_done_fails_loudly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path)
        rc = cli.main(["approve", "s1-corpus", "-c", str(config)])
        assert rc == 1
        assert "not 'done'" in capsys.readouterr().err

    def test_blocked_run_exits_1_and_records_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ohr_bench without --accept-non-commercial: the license gate blocks the stage.
        config = init_campaign(tmp_path, datasets="ohr_bench")
        repo = make_repo(tmp_path)
        rc = cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 1
        assert "accept-non-commercial" in capsys.readouterr().err

        raw = json.loads((tmp_path / "camp" / STATE_FILENAME).read_text())
        stage = next(s for s in raw["stages"] if s["name"] == "s1-corpus")
        assert stage["status"] == "blocked"
        assert "non-commercial" in stage["reason"]

    def test_resume_reports_a_blocked_stage_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path, datasets="ohr_bench")
        repo = make_repo(tmp_path)
        cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)])
        capsys.readouterr()
        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 1
        assert "BLOCKED" in capsys.readouterr().err

    @pytest.mark.usefixtures("offline_seams")
    def test_resume_awaits_approval_on_a_done_stage(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path)
        repo = make_repo(tmp_path)
        cli.main(["run", "s1-corpus", "-c", str(config), "--repo-root", str(repo)])
        capsys.readouterr()
        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        assert "awaiting sign-off" in capsys.readouterr().out

    def test_missing_config_is_a_clear_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cli.main(["status", "-c", str(tmp_path / "nope.yaml"), "--repo-root", str(tmp_path)])
        assert rc == 1
        assert "init" in capsys.readouterr().err
