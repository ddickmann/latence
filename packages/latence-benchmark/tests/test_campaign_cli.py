# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``latence-campaign`` end to end (offline): init parity, status, run/approve/resume flow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.campaign import cli
from latence_benchmark.campaign.state import STATE_FILENAME, CampaignState

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for incumbent in ("musique-full", "hotpot-full"):
        d = repo / "benchmark" / "official-data" / incumbent
        d.mkdir(parents=True)
        (d / "data.jsonl").write_text('{"q": 1}\n')
    dst = repo / "benchmark" / "corpus" / "manifest.py"
    dst.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "benchmark" / "corpus" / "manifest.py", dst)
    return repo


def init_campaign(tmp_path: Path, datasets: str = "musique,hotpotqa") -> Path:
    config = tmp_path / "camp" / "campaign.yaml"
    rc = cli.main(
        [
            "init",
            "--non-interactive",
            "--campaign-root",
            str(tmp_path / "camp"),
            "--datasets",
            datasets,
            "-o",
            str(config),
        ]
    )
    assert rc == 0
    return config


class TestInit:
    def test_non_interactive_writes_a_validated_config(self, tmp_path: Path) -> None:
        config = init_campaign(tmp_path)
        text = config.read_text()
        assert "campaign:" in text
        assert "- musique" in text

    def test_byte_determinism_across_runs(self, tmp_path: Path) -> None:
        a = init_campaign(tmp_path / "a", datasets="musique,uda")
        b = init_campaign(tmp_path / "b", datasets="musique,uda")
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
                "musique,vidoseek",
                "--pod-ssh",
                "root@pod",
                "-o",
                str(by_flags),
            ]
        )
        assert rc == 0

        answers: Iterator[str] = iter(["/camp", "musique,vidoseek", "root@pod"])
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
        for name in ("musique", "uda", "ohr_bench"):
            assert f"- {name}" in text

    def test_unknown_dataset_is_usage_error_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cli.main(
            [
                "init",
                "--non-interactive",
                "--datasets",
                "musique,frames",
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

    def test_run_approve_resume_flow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = init_campaign(tmp_path)  # musique,hotpotqa — local references only
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

        rc = cli.main(["resume", "-c", str(config), "--repo-root", str(repo)])
        assert rc == 0
        assert "complete" in capsys.readouterr().out

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
