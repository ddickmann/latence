# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The s3-pipeline stage, fully offline: gating, reuse verification, preps, the pod loop.

Same posture as the s2 suite: no ssh, no GPU, no OpenRouter — the pod is a scripted
:class:`FakePod` behind the transport seam and the poll loop runs on a fake clock. What is
proven here is every decision the stage makes AROUND the real run: refusal until s2 is
approved, blocked-with-reason on a lost M6 export, deterministic prep bytes, launch/
re-enter/relaunch semantics, stall + death handling, and the consolidated metrics contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_benchmark.campaign import pipeline_stage
from latence_benchmark.campaign.pipeline_stage import (
    STAGE_NAME,
    launch_cmd,
    poll_pipeline_run,
    prepare_multihop_rag,
    prepare_wiki2multihop,
    run_s3_stage,
    s3_done_probe,
    verify_reused_export,
)
from latence_benchmark.campaign.s3_lib import REMOTE_ROOT
from latence_benchmark.campaign.stages import (
    CampaignContext,
    GateRefusedError,
    StageBlockedError,
    registry,
    run_stage,
    stage_dir,
)
from latence_benchmark.campaign.state import STATE_FILENAME, CampaignState
from latence_benchmark.campaign.wizard import CampaignAnswers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# -- fixtures --------------------------------------------------------------------------------


def fixture_report(*, with_induction: bool = True) -> dict[str, Any]:
    """A minimal quality-report.json in the real shape (same numbers as the s3_lib suite)."""
    stages: list[dict[str, Any]] = []
    if with_induction:
        stages.append(
            {
                "name": "induce",
                "capability": "schema_induction",
                "provider": "label_inducer.llm",
                "induction_calls": 100,
                "induction_cache_hits": 40,
                "induction_prompt_tokens": 100_000,
                "induction_completion_tokens": 20_000,
                "induction_budget_exhausted": False,
            }
        )
    return {
        "pipeline_name": "s3-x",
        "run_id": "s3-x-v1",
        "document_count": 3,
        "stages": stages,
        "chunk": {"chunks_total": 42, "chunks_flagged": 1},
        "entities": {"mentions_total": 120},
        "relations": {"relations_total": 30},
        "redaction": {"pii_total": 5, "per_type": {"email": 5}},
        "graph": {
            "nodes_total": 80,
            "edges_total": 25,
            "hyperedges": {"hyperedges_total": 12},
        },
    }


def make_repo(tmp_path: Path, *, datasets: Sequence[str] = ("ohr_bench",)) -> Path:
    repo = tmp_path / "repo"
    corpus = repo / "benchmark" / "sota-corpus"
    for ds in ("ohr_bench", "vidoseek", "uda"):
        if ds in datasets:
            md = corpus / ds / "markdown"
            md.mkdir(parents=True)
            (md / "doc1.md").write_text("# Doc\n\nText.\n", encoding="utf-8")
    if "multihop_rag" in datasets:
        raw = corpus / "multihop_rag" / "raw"
        raw.mkdir(parents=True)
        (raw / "corpus.json").write_text(
            json.dumps(
                [
                    {"title": "Alpha buys Beta", "body": "Alpha acquired Beta in 2023."},
                    {"title": "Gamma", "body": "Gamma partnered with Delta."},
                ]
            ),
            encoding="utf-8",
        )
    if "wiki2multihop" in datasets:
        raw = corpus / "wiki2multihop" / "raw"
        raw.mkdir(parents=True)
        contexts = [
            json.dumps([["Xawery", ["A director."]], ["War", ["A 2009 film."]]]),
            json.dumps([["Xawery", ["A director."]], ["Maheen", ["A designer."]]]),
        ]
        # cast, not type-ignore: pyarrow is dynamically typed in this repo's mypy config.
        cast("Any", pq).write_table(
            pa.table({"context": contexts}), raw / "dev.parquet"
        )
    for incumbent, name in (("musique", "musique-full"), ("hotpotqa", "hotpot-full")):
        if incumbent in datasets:
            export = repo / "benchmark" / "official-data" / name / "export"
            export.mkdir(parents=True)
            (export / "graph-hyperedges.kv").write_bytes(b"LHKV1-fixture")
            (export / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    return repo


def make_ctx(tmp_path: Path, datasets: tuple[str, ...], repo: Path) -> CampaignContext:
    return CampaignContext(
        campaign_root=tmp_path / "campaign",
        repo_root=repo,
        answers=CampaignAnswers(campaign_root=str(tmp_path / "campaign"), datasets=datasets),
    )


class FakePod:
    """A scripted pod: per-dataset status sequences, materialising quality-report pulls."""

    def __init__(
        self,
        *,
        states: Mapping[str, Sequence[tuple[str, int]]],
        report: Mapping[str, Any] | None = None,
        log_tail: str = "Traceback: fixture boom",
    ) -> None:
        self._states = {ds: list(seq) for ds, seq in states.items()}
        self._report = dict(report if report is not None else fixture_report())
        self._log_tail = log_tail
        self.commands: list[str] = []
        self.pushes: list[tuple[str, str]] = []
        self.pulls: list[tuple[str, str]] = []

    def run(self, cmd: str) -> tuple[int, str]:
        self.commands.append(cmd)
        for ds, seq in self._states.items():
            if f"du -sk {REMOTE_ROOT}/{ds}" in cmd:
                phase, progress = seq.pop(0) if len(seq) > 1 else seq[0]
                return 0, f"{phase} {progress}"
        if cmd.startswith("tail "):
            return 0, self._log_tail
        if "find " in cmd:
            return 0, f"{REMOTE_ROOT}/x/export/corpus.jsonl\n"
        return 0, ""

    def push(self, local: Path, remote: str) -> None:
        self.pushes.append((str(local), remote))

    def pull(self, remote: str, local: Path) -> None:
        self.pulls.append((remote, str(local)))
        if remote.endswith("quality-report.json"):
            local.write_text(json.dumps(self._report), encoding="utf-8")
        elif remote.endswith("quality-report.md"):
            local.write_text("# Quality Report — fixture\n", encoding="utf-8")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


# -- gating ----------------------------------------------------------------------------------


class TestGating:
    def test_s3_is_registered_directly_after_s2(self) -> None:
        assert [spec.name for spec in registry()] == ["s1-corpus", "s2-ocr", STAGE_NAME]

    def test_run_refuses_until_s2_is_approved(self, tmp_path: Path) -> None:
        specs = registry()
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        state = CampaignState(ctx.campaign_root / STATE_FILENAME, [s.name for s in specs])
        with pytest.raises(GateRefusedError, match="s2-ocr"):
            run_stage(specs, STAGE_NAME, ctx, state)


# -- reuse verification ----------------------------------------------------------------------


class TestReuse:
    def test_present_export_is_recorded_not_reprocessed(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("musique",))
        entry = verify_reused_export(repo, "musique")
        assert entry["reused_from_m6"] is True
        assert entry["export"] == "benchmark/official-data/musique-full/export"
        assert entry["hyperedge_kv_bytes"] > 0

    def test_missing_export_blocks_with_the_fix(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=())
        with pytest.raises(StageBlockedError, match="graph-hyperedges.kv"):
            verify_reused_export(repo, "hotpotqa")

    def test_empty_kv_blocks_too(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("musique",))
        kv = repo / "benchmark/official-data/musique-full/export/graph-hyperedges.kv"
        kv.write_bytes(b"")
        with pytest.raises(StageBlockedError, match="missing/empty"):
            verify_reused_export(repo, "musique")


# -- local preps -----------------------------------------------------------------------------


class TestPreps:
    def test_multihop_rag_prep_writes_then_skips(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("multihop_rag",))
        first = prepare_multihop_rag(repo)
        assert first == {"files": 2, "written": 2, "skipped": 0}
        md = repo / "benchmark" / "sota-corpus" / "multihop_rag" / "markdown"
        names = sorted(p.name for p in md.iterdir())
        assert len(names) == 2
        assert any(n.startswith("alpha-buys-beta-") for n in names)
        # Idempotent: the second pass rewrites nothing (resumability at file granularity).
        assert prepare_multihop_rag(repo) == {"files": 2, "written": 0, "skipped": 2}

    def test_wiki2_prep_dedups_the_question_pool(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("wiki2multihop",))
        first = prepare_wiki2multihop(repo)
        assert first == {"files": 3, "written": 3, "skipped": 0}  # Xawery deduped
        assert prepare_wiki2multihop(repo) == {"files": 3, "written": 0, "skipped": 3}

    def test_missing_raw_blocks_pointing_at_s1(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=())
        with pytest.raises(StageBlockedError, match="s1-corpus"):
            prepare_multihop_rag(repo)
        with pytest.raises(StageBlockedError, match="s1-corpus"):
            prepare_wiki2multihop(repo)


# -- the poll loop ---------------------------------------------------------------------------


class TestPollLoop:
    def test_returns_on_done(self) -> None:
        pod = FakePod(states={"uda": [("running", 1), ("running", 2), ("done", 9)]})
        clock = FakeClock()
        poll_pipeline_run(pod, "uda", clock=clock, sleep=clock.sleep)

    def test_stalled_bytes_block_after_the_budget(self) -> None:
        pod = FakePod(states={"uda": [("running", 5)]})  # bytes never grow
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="stalled"):
            poll_pipeline_run(pod, "uda", clock=clock, sleep=clock.sleep)
        assert clock.now > 3600.0  # the full budget was granted before blocking

    def test_growing_bytes_reset_the_stall_window(self) -> None:
        states = [("running", i) for i in range(70)] + [("done", 99)]
        pod = FakePod(states={"uda": states})
        clock = FakeClock()
        poll_pipeline_run(pod, "uda", clock=clock, sleep=clock.sleep)
        assert clock.now > 3600.0

    def test_a_dead_run_blocks_immediately_with_the_log_tail(self) -> None:
        pod = FakePod(
            states={"uda": [("running", 1), ("absent", 1)]}, log_tail="OOM: fixture"
        )
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="OOM: fixture"):
            poll_pipeline_run(pod, "uda", clock=clock, sleep=clock.sleep)
        assert clock.now < 3600.0  # no point waiting out the budget on a corpse


# -- the stage run ---------------------------------------------------------------------------


class TestStageRun:
    def test_fresh_run_pushes_launches_polls_pulls_and_records(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("musique", "ohr_bench"))
        ctx = make_ctx(tmp_path, ("musique", "ohr_bench"), repo)
        pod = FakePod(states={"ohr_bench": [("absent", 0), ("running", 1), ("done", 9)]})
        clock = FakeClock()
        note = run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert "1 dataset(s) processed" in note
        assert "1 reused from M6" in note

        # Launched exactly once: nohup + pid file + the same-run-id resume contract.
        launches = [c for c in pod.commands if "nohup" in c]
        assert launches == [launch_cmd("ohr_bench")]
        assert "--run-id s3-ohr_bench-v1" in launches[0]
        assert "PYTHONHASHSEED=0" in launches[0]

        # Pushed: the dataset's markdown and its generated pipeline.yaml.
        pushed = [src for src, _ in pod.pushes]
        assert any(src.endswith("ohr_bench/markdown") for src in pushed)
        assert any(src.endswith("ohr_bench.yaml") for src in pushed)

        # The generated pipeline is a committed-able stage artifact with the budget wired.
        yaml_path = stage_dir(ctx.campaign_root, STAGE_NAME) / "pipelines" / "ohr_bench.yaml"
        assert "max_induction_calls" in yaml_path.read_text(encoding="utf-8")

        # Reports + export listing landed; metrics consolidated the quality report.
        report_dir = stage_dir(ctx.campaign_root, STAGE_NAME) / "reports" / "ohr_bench"
        assert (report_dir / "quality-report.json").is_file()
        assert (report_dir / "quality-report.md").is_file()
        assert (report_dir / "export-files.txt").read_text().strip().endswith("corpus.jsonl")

        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert set(metrics["datasets"]) == {"musique", "ohr_bench"}
        ohr = metrics["datasets"]["ohr_bench"]
        assert (ohr["chunks"], ohr["mentions"], ohr["relations"]) == (42, 120, 30)
        assert (ohr["entities"], ohr["hyperedges"], ohr["pii_total"]) == (80, 12, 5)
        assert ohr["induction"]["usd_actual"] == pytest.approx(0.072)
        assert ohr["run_id"] == "s3-ohr_bench-v1"
        assert metrics["datasets"]["musique"]["reused_from_m6"] is True
        budget = metrics["induction_budget"]
        assert budget["induced_datasets"] == ["ohr_bench"]
        assert budget["max_calls_per_dataset"] > 0
        assert budget["usd_actual_total"] == pytest.approx(0.072)

        assert s3_done_probe(ctx) is True

    def test_resume_reenters_a_live_run_without_pushing(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        pod = FakePod(states={"ohr_bench": [("running", 5), ("running", 6), ("done", 9)]})
        clock = FakeClock()
        run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert not any("nohup" in c for c in pod.commands)  # never relaunched
        assert pod.pushes == []  # nothing re-pushed: the job was simply re-entered
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["datasets"]["ohr_bench"]["resumed"] is True

    def test_a_dead_run_is_relaunched_under_the_same_run_id(self, tmp_path: Path) -> None:
        # phase absent on ENTRY = launch (fresh or after death — same id, checkpoints replay).
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        pod = FakePod(states={"ohr_bench": [("absent", 3), ("done", 9)]})
        clock = FakeClock()
        run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert sum("nohup" in c for c in pod.commands) == 1

    def test_an_already_done_pod_run_skips_the_poll(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        pod = FakePod(states={"ohr_bench": [("done", 9)]})
        clock = FakeClock()
        run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert clock.now == 0.0  # no poll sleeps

    def test_a_locally_pulled_report_never_touches_the_pod(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        report_dir = stage_dir(ctx.campaign_root, STAGE_NAME) / "reports" / "ohr_bench"
        report_dir.mkdir(parents=True)
        (report_dir / "quality-report.json").write_text(
            json.dumps(fixture_report()), encoding="utf-8"
        )
        pod = FakePod(states={"ohr_bench": [("absent", 0)]})
        run_s3_stage(ctx, pod, clock=FakeClock(), sleep=lambda _s: None)
        assert pod.commands == []
        assert pod.pushes == []
        assert s3_done_probe(ctx) is True

    def test_text_prep_dataset_runs_end_to_end(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("multihop_rag",))
        ctx = make_ctx(tmp_path, ("multihop_rag",), repo)
        pod = FakePod(
            states={"multihop_rag": [("absent", 0), ("done", 9)]},
            report=fixture_report(with_induction=False),
        )
        clock = FakeClock()
        run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["preps"]["multihop_rag"] == {"files": 2, "written": 2, "skipped": 0}
        assert metrics["datasets"]["multihop_rag"]["induction"] is None
        assert metrics["datasets"]["multihop_rag"]["induction_mode"] == "none"
        assert metrics["induction_budget"]["induced_datasets"] == []
        # The prep'd markdown (not the raw json) is what went to the pod.
        assert any(src.endswith("multihop_rag/markdown") for src, _ in pod.pushes)

    def test_missing_s2_markdown_blocks_pointing_at_s2(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=())
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        pod = FakePod(states={"vidoseek": [("absent", 0)]})
        with pytest.raises(StageBlockedError, match="s2-ocr"):
            run_s3_stage(ctx, pod, clock=FakeClock(), sleep=lambda _s: None)

    def test_empty_pod_ssh_blocks_before_any_remote_call(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        with pytest.raises(StageBlockedError, match="pod_ssh"):
            run_s3_stage(ctx)  # no transport injected, no pod_ssh configured

    def test_reused_only_selection_never_needs_a_pod(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("musique", "hotpotqa"))
        ctx = make_ctx(tmp_path, ("musique", "hotpotqa"), repo)
        note = run_s3_stage(ctx)  # no transport at all: nothing to process
        assert "0 dataset(s) processed" in note
        assert s3_done_probe(ctx) is True

    def test_done_probe_tracks_the_selection(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench", "vidoseek"))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        pod = FakePod(states={"ohr_bench": [("absent", 0), ("done", 9)]})
        clock = FakeClock()
        run_s3_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert s3_done_probe(ctx) is True
        # Widening the corpus selection re-opens the stage: vidoseek has no report yet.
        wider = make_ctx(tmp_path, ("ohr_bench", "vidoseek"), repo)
        assert s3_done_probe(wider) is False

    def test_progress_probe_counts_pulled_reports(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("ohr_bench",))
        ctx = make_ctx(tmp_path, ("ohr_bench",), repo)
        assert pipeline_stage.s3_progress_probe(ctx) == "0/1 dataset quality report(s) pulled"
