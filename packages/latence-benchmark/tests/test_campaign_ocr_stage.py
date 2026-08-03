# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The s2-ocr stage, fully offline: fin cap, zip extraction, the transport seam.

No ssh, no rsync, no GPU: the pod lives behind a scripted :class:`FakePod` transport and
the poll loop runs on a fake clock. The REAL pod run is the operator's — nothing here
claims it (no fake green); what IS proven here is every decision the stage makes around
it: gating, determinism, resumability, stall handling, and the metrics contract.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_benchmark.campaign.ocr_stage import (
    FIN_CAP,
    REMOTE_ROOT,
    STAGE_NAME,
    extract_zip,
    fin_cap_selection,
    ocr_done_probe,
    parse_pod_ssh,
    poll_pod_job,
    run_ocr_stage,
    zip_jobs,
)
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


def write_parquet(path: Path, doc_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # cast, not type-ignore: pyarrow is dynamically typed in this repo's mypy config.
    cast("Any", pq).write_table(pa.table({"doc_name": list(doc_names)}), path)


def write_zip(path: Path, entries: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def make_repo(tmp_path: Path, *, datasets: Sequence[str] = ("vidoseek",)) -> Path:
    """A fixture checkout holding the raw zips (and pod scripts) the stage consumes."""
    repo = tmp_path / "repo"
    pod = repo / "benchmark" / "sota-campaign" / "pod"
    pod.mkdir(parents=True)
    scripts = ("serve_ocr.sh", "run_worker.sh", "ocr_worker.py", "stop_ocr.sh", "podcheck_s2.sh")
    for script in scripts:
        (pod / script).write_text("# fixture stand-in\n")
    corpus = repo / "benchmark" / "sota-corpus"
    if "vidoseek" in datasets:
        write_zip(
            corpus / "vidoseek" / "raw" / "vidoseek_pdf_document.zip",
            {"pdf/a.pdf": b"%PDF-a", "pdf/b.pdf": b"%PDF-b"},
        )
    if "uda" in datasets:
        # Three fin docs in the QA table; D4 exists in the archive but is NOT questioned,
        # so the cap must leave it unextracted.
        write_parquet(
            corpus / "uda" / "raw" / "fin" / "test_00000_of_00001.parquet",
            ["D2"] * 3 + ["D1"] * 5 + ["D3"] * 3,
        )
        write_zip(
            corpus / "uda" / "raw" / "src_doc_files" / "fin_docs.zip",
            {f"fin_docs/{d}.pdf": b"%PDF" for d in ("D1", "D2", "D3", "D4")},
        )
        write_zip(
            corpus / "uda" / "raw" / "src_doc_files" / "paper_docs.zip",
            {"paper_docs/1911.01371.pdf": b"%PDF-p"},
        )
    return repo


def make_ctx(tmp_path: Path, datasets: tuple[str, ...], repo: Path) -> CampaignContext:
    return CampaignContext(
        campaign_root=tmp_path / "campaign",
        repo_root=repo,
        answers=CampaignAnswers(campaign_root=str(tmp_path / "campaign"), datasets=datasets),
    )


class FakePod:
    """A scripted pod behind the transport seam: recorded commands, materialising pulls."""

    def __init__(
        self,
        *,
        states: Sequence[str | None],
        markdown: Mapping[str, Sequence[str]],
        pod_metrics: Mapping[str, Any],
    ) -> None:
        self._states = list(states)
        self._markdown = markdown
        self._pod_metrics = pod_metrics
        self.commands: list[str] = []
        self.pushes: list[tuple[str, str]] = []
        self.pulls: list[tuple[str, str]] = []

    def run(self, cmd: str) -> tuple[int, str]:
        self.commands.append(cmd)
        if cmd == f"cat {REMOTE_ROOT}/state.json":
            state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
            return (1, "cat: No such file or directory") if state is None else (0, state)
        return 0, ""

    def push(self, local: Path, remote: str) -> None:
        self.pushes.append((str(local), remote))

    def pull(self, remote: str, local: Path) -> None:
        self.pulls.append((remote, str(local)))
        if remote.endswith("/metrics.json"):
            local.write_text(json.dumps(dict(self._pod_metrics)), encoding="utf-8")
        elif remote.endswith("/failures.jsonl"):
            local.write_text("", encoding="utf-8")
        elif "/markdown/" in remote:
            dataset = remote.rstrip("/").rsplit("/", 1)[-1]
            local.mkdir(parents=True, exist_ok=True)
            for doc_id in self._markdown.get(dataset, []):
                (local / f"{doc_id}.md").write_text(f"# {doc_id}\n", encoding="utf-8")
        elif "/spot-img/" in remote:
            # Only worker-retained spot pages exist remotely; pages-img is deleted
            # per doc by the streaming worker, so the client must never pull it.
            local.write_bytes(b"\xff\xd8fixture-jpeg")
        elif "/pages/" in remote:
            local.write_text("raw page markdown", encoding="utf-8")


def state_json(phase: str, progress: int) -> str:
    return json.dumps({"phase": phase, "progress": progress})


def vidoseek_pod(*, states: Sequence[str | None]) -> FakePod:
    docs = {"vidoseek/pdf__a": 2, "vidoseek/pdf__b": 1}
    return FakePod(
        states=states,
        markdown={"vidoseek": ["pdf__a", "pdf__b"]},
        pod_metrics={
            "datasets": {
                "vidoseek": {"docs": 2, "pages": 3, "tables_normalized": 4,
                             "tables_kept_raw": 1, "pages_missing": 0, "failures": 0},
            },
            "docs": docs,
            "spot_check": [
                {"doc": "vidoseek/pdf__a", "page": 2},
                {"doc": "vidoseek/pdf__b", "page": 1},
            ],
            "pages_total": 3,
            "pages_done": 3,
            "wall_clock_s": 60.0,
            "pages_per_s": 0.05,
        },
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


# -- fin cap ---------------------------------------------------------------------------------


class TestFinCap:
    def test_selection_is_deterministic_with_name_tie_break(self, tmp_path: Path) -> None:
        write_parquet(
            tmp_path / "fin" / "test_00000_of_00001.parquet",
            ["zeta"] * 3 + ["alpha"] * 3 + ["mid"] * 5 + ["rare"],
        )
        selection = fin_cap_selection(tmp_path / "fin", cap=3)
        assert selection.selected == ("mid", "alpha", "zeta")
        assert (selection.qa_covered, selection.qa_total) == (11, 12)
        assert selection.docs_total == 4

    def test_counts_span_multiple_parquet_files(self, tmp_path: Path) -> None:
        write_parquet(tmp_path / "fin" / "test_00000_of_00002.parquet", ["a", "b"])
        write_parquet(tmp_path / "fin" / "test_00001_of_00002.parquet", ["b", "b"])
        selection = fin_cap_selection(tmp_path / "fin", cap=1)
        assert selection.selected == ("b",)
        assert selection.qa_covered == 3

    def test_missing_fin_dir_blocks_with_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(StageBlockedError, match="s1-corpus"):
            fin_cap_selection(tmp_path / "absent")

    def test_default_cap_is_the_ruling(self) -> None:
        assert FIN_CAP == 150

    def test_zip_jobs_restrict_fin_to_the_selection(self) -> None:
        jobs = zip_jobs(["uda"], ("D1", "D2"))
        fin_jobs = [j for j in jobs if "fin_docs" in j.zip_relpath]
        assert len(fin_jobs) == 1
        assert fin_jobs[0].members == ("fin_docs/D1.pdf", "fin_docs/D2.pdf")
        # The paper archive has no member restriction — papers OCR in full.
        paper = next(j for j in jobs if "paper_docs" in j.zip_relpath)
        assert paper.members is None


# -- zip extraction --------------------------------------------------------------------------


class TestZipExtraction:
    def test_extracts_then_skips_on_rerun(self, tmp_path: Path) -> None:
        archive = tmp_path / "a.zip"
        write_zip(archive, {"pdf/x.pdf": b"1", "pdf/y.pdf": b"2"})
        dest = tmp_path / "pdfs"
        assert extract_zip(archive, dest) == (2, 0)
        assert (dest / "pdf" / "x.pdf").read_bytes() == b"1"
        assert extract_zip(archive, dest) == (0, 2)  # resumable: nothing redone

    def test_member_subset_extracts_only_the_selection(self, tmp_path: Path) -> None:
        archive = tmp_path / "fin.zip"
        write_zip(archive, {f"fin_docs/{d}.pdf": b"%" for d in ("D1", "D2", "D3")})
        dest = tmp_path / "pdfs"
        extract_zip(archive, dest, members=("fin_docs/D1.pdf", "fin_docs/D3.pdf"))
        assert sorted(p.name for p in (dest / "fin_docs").iterdir()) == ["D1.pdf", "D3.pdf"]

    def test_missing_member_blocks_loudly(self, tmp_path: Path) -> None:
        archive = tmp_path / "fin.zip"
        write_zip(archive, {"fin_docs/D1.pdf": b"%"})
        with pytest.raises(StageBlockedError, match="absent from the archive"):
            extract_zip(archive, tmp_path / "pdfs", members=("fin_docs/GHOST.pdf",))

    def test_zip_slip_traversal_is_refused(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        write_zip(archive, {"../escaped.pdf": b"boom"})
        dest = tmp_path / "inner" / "pdfs"
        with pytest.raises(StageBlockedError, match="zip-slip"):
            extract_zip(archive, dest)
        assert not (tmp_path / "inner" / "escaped.pdf").exists()
        assert not (tmp_path / "escaped.pdf").exists()

    def test_zip_slip_absolute_path_is_refused(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        write_zip(archive, {"/tmp/abs.pdf": b"boom"})
        with pytest.raises(StageBlockedError, match="zip-slip"):
            extract_zip(archive, tmp_path / "pdfs")

    def test_missing_archive_blocks_pointing_at_s1(self, tmp_path: Path) -> None:
        with pytest.raises(StageBlockedError, match="s1-corpus"):
            extract_zip(tmp_path / "nope.zip", tmp_path / "pdfs")


# -- pod_ssh parsing -------------------------------------------------------------------------


class TestPodSshParsing:
    def test_the_real_config_line_parses(self) -> None:
        endpoint = parse_pod_ssh("ssh root@213.173.108.103 -p 11108 -i ~/.ssh/id_ed25519")
        assert endpoint.target == "root@213.173.108.103"
        assert endpoint.argv == (
            "ssh", "-p", "11108", "-i", "~/.ssh/id_ed25519", "root@213.173.108.103"
        )
        assert endpoint.rsh == "ssh -p 11108 -i '~/.ssh/id_ed25519'"

    def test_a_non_ssh_command_is_refused(self) -> None:
        with pytest.raises(StageBlockedError, match="full ssh command line"):
            parse_pod_ssh("mosh root@pod")

    def test_a_target_less_line_is_refused(self) -> None:
        with pytest.raises(StageBlockedError, match="no user@host"):
            parse_pod_ssh("ssh -p 22")

    def test_an_option_missing_its_value_is_refused(self) -> None:
        with pytest.raises(StageBlockedError, match="missing its value"):
            parse_pod_ssh("ssh root@pod -p")


# -- the poll loop ---------------------------------------------------------------------------


class TestPollLoop:
    def test_returns_the_final_state_on_done(self) -> None:
        pod = vidoseek_pod(
            states=[state_json("ocr", 1), state_json("ocr", 2), state_json("done", 6)]
        )
        clock = FakeClock()
        final = poll_pod_job(pod, clock=clock, sleep=clock.sleep)
        assert final["phase"] == "done"

    def test_stalled_progress_blocks_after_the_budget(self) -> None:
        pod = vidoseek_pod(states=[state_json("ocr", 5)])  # progress never advances
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="stalled"):
            poll_pod_job(pod, clock=clock, sleep=clock.sleep)
        assert clock.now > 1800.0  # the full budget was granted before blocking

    def test_advancing_progress_resets_the_stall_window(self) -> None:
        # 70 reads with advancing progress at 30s intervals — 2100s wall, never stalled.
        states = [state_json("ocr", i) for i in range(70)] + [state_json("done", 99)]
        pod = vidoseek_pod(states=states)
        clock = FakeClock()
        assert poll_pod_job(pod, clock=clock, sleep=clock.sleep)["phase"] == "done"
        assert clock.now > 1800.0

    def test_worker_failure_blocks_with_its_reason(self) -> None:
        failed = json.dumps(
            {"phase": "failed", "failed_phase": "render", "error": "pdfium exploded"}
        )
        pod = vidoseek_pod(states=[failed])
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="pdfium exploded"):
            poll_pod_job(pod, clock=clock, sleep=clock.sleep)

    def test_unreadable_state_counts_toward_the_stall_budget(self) -> None:
        pod = vidoseek_pod(states=[None])  # cat fails forever
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="stalled"):
            poll_pod_job(pod, clock=clock, sleep=clock.sleep)


# -- gating ----------------------------------------------------------------------------------


class TestGating:
    def test_s2_is_registered_directly_after_s1(self) -> None:
        assert [spec.name for spec in registry()] == ["s1-corpus", STAGE_NAME]

    def test_run_refuses_until_s1_is_approved(self, tmp_path: Path) -> None:
        specs = registry()
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        state = CampaignState(
            ctx.campaign_root / STATE_FILENAME, [s.name for s in specs]
        )
        with pytest.raises(GateRefusedError, match="s1-corpus"):
            run_stage(specs, STAGE_NAME, ctx, state)


# -- the stage run ---------------------------------------------------------------------------


class TestStageRun:
    def test_fresh_run_pushes_launches_polls_pulls_and_records(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("musique", "vidoseek"), repo)
        pod = vidoseek_pod(states=[None, state_json("done", 6)])
        clock = FakeClock()
        note = run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert "2 markdown doc(s)" in note

        # Launched exactly once, with the nohup+pid-file shape, THROUGH the supervisor
        # (run_worker.sh relaunches the -u worker; the OOM field-lesson hardening).
        launches = [c for c in pod.commands if "nohup" in c]
        assert len(launches) == 1
        assert "serve_ocr.sh" in launches[0]
        assert "run_worker.sh" in launches[0]
        assert "worker.pid" in launches[0]

        # Pushed: the pod scripts, the unit-tested ocr_lib, and the extracted PDFs.
        pushed = [src for src, _ in pod.pushes]
        assert any(src.endswith("ocr_lib.py") for src in pushed)
        assert any(src.endswith("serve_ocr.sh") for src in pushed)
        assert any(src.endswith("vidoseek/pdfs") for src in pushed)

        # The shared GPU was released.
        assert any("stop_ocr.sh" in c for c in pod.commands)

        # Markdown landed locally; metrics measured it (not the pod's claim).
        md_dir = repo / "benchmark" / "sota-corpus" / "vidoseek" / "markdown"
        assert sorted(p.name for p in md_dir.iterdir()) == ["pdf__a.md", "pdf__b.md"]
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["stage"] == STAGE_NAME
        assert metrics["fin_cap"] is None
        assert metrics["skipped_native"] == ["musique"]
        assert metrics["datasets"]["vidoseek"]["markdown_docs_local"] == 2
        assert metrics["datasets"]["vidoseek"]["tables_normalized"] == 4
        assert metrics["pod"]["resumed"] is False
        assert metrics["pod"]["vllm_stopped"] is True

        # Spot check: exactly the pages the worker RETAINED under spot-img (its
        # metrics' spot_check list) — the streamed-away page images are never pulled.
        assert len(metrics["spot_check"]) == 2
        for sample in metrics["spot_check"]:
            assert Path(sample["image"]).is_file()
            assert Path(sample["markdown"]).is_file()
        spot_pulls = [remote for remote, _ in pod.pulls if "/spot-img/" in remote]
        assert spot_pulls == [
            f"{REMOTE_ROOT}/spot-img/vidoseek/pdf__a/0002.jpg",
            f"{REMOTE_ROOT}/spot-img/vidoseek/pdf__b/0001.jpg",
        ]
        assert not any("/pages-img/" in remote for remote, _ in pod.pulls)

        assert ocr_done_probe(ctx) is True

    def test_resume_reenters_the_poll_without_relaunching(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        pod = vidoseek_pod(
            states=[state_json("ocr", 5), state_json("ocr", 6), state_json("done", 9)]
        )
        clock = FakeClock()
        run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert not any("nohup" in c for c in pod.commands)  # never relaunched
        assert pod.pushes == []  # and nothing re-pushed: the job was simply re-entered
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["pod"]["resumed"] is True

    def test_finished_pod_job_skips_the_poll_entirely(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        pod = vidoseek_pod(states=[state_json("done", 9)])
        clock = FakeClock()
        run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert clock.now == 0.0  # no poll sleeps: the state was already done on entry

    def test_fin_cap_is_recorded_and_caps_the_extraction(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path, datasets=("uda",))
        ctx = make_ctx(tmp_path, ("uda",), repo)
        docs = {"uda/fin_docs__D1": 1, "uda/paper_docs__1911.01371": 1}
        pod = FakePod(
            states=[None, state_json("done", 4)],
            markdown={"uda": ["fin_docs__D1", "paper_docs__1911.01371"]},
            pod_metrics={"datasets": {"uda": {"docs": 2}}, "docs": docs},
        )
        clock = FakeClock()
        run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)

        # D4 is in the archive but not in the QA table: the cap left it unextracted.
        fin_pdfs = repo / "benchmark" / "sota-corpus" / "uda" / "pdfs" / "fin_docs"
        assert sorted(p.name for p in fin_pdfs.iterdir()) == ["D1.pdf", "D2.pdf", "D3.pdf"]
        papers = repo / "benchmark" / "sota-corpus" / "uda" / "pdfs" / "paper_docs"
        assert [p.name for p in papers.iterdir()] == ["1911.01371.pdf"]

        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        fin = metrics["fin_cap"]
        assert fin["selected"] == ["D1", "D2", "D3"]  # 5,3,3 QA; ties by name
        assert (fin["qa_covered"], fin["qa_total"]) == (11, 11)
        assert fin["docs_total"] == 3
        # A worker metrics without a spot_check list (nothing retained) records empty.
        assert metrics["spot_check"] == []

    def test_stalled_pod_blocks_the_stage_with_the_reason(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        pod = vidoseek_pod(states=[None, state_json("ocr", 1)])
        clock = FakeClock()
        with pytest.raises(StageBlockedError, match="stalled"):
            run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)

    def test_empty_pod_ssh_blocks_before_any_remote_call(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        with pytest.raises(StageBlockedError, match="pod_ssh"):
            run_ocr_stage(ctx)  # no transport injected, no pod_ssh configured

    def test_text_only_selection_completes_as_a_recorded_noop(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("musique", "hotpotqa"), repo)
        note = run_ocr_stage(ctx)  # no transport needed: the pod is never touched
        assert "nothing to OCR" in note
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["datasets"] == {}
        assert metrics["skipped_native"] == ["musique", "hotpotqa"]
        assert ocr_done_probe(ctx) is True

    def test_done_probe_tracks_the_selection(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path)
        ctx = make_ctx(tmp_path, ("vidoseek",), repo)
        assert ocr_done_probe(ctx) is False
        pod = vidoseek_pod(states=[None, state_json("done", 6)])
        clock = FakeClock()
        run_ocr_stage(ctx, pod, clock=clock, sleep=clock.sleep)
        assert ocr_done_probe(ctx) is True

        # Widening the corpus selection re-opens the stage: uda has no markdown yet.
        wider = make_ctx(tmp_path, ("vidoseek", "uda"), repo)
        assert ocr_done_probe(wider) is False
