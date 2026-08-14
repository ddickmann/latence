# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Stage ``s5-retrieval``: plan derivation, the probes, the gate, the no-fake-green floor.

The harness itself is behind the :data:`~latence_benchmark.campaign.retrieval_stage.MatrixRunner`
seam here (the ``PodTransport`` pattern), so every claim below is proven offline — no 100 MB
export, no embedding checkpoint. What the *real* runner does with a plan is pinned on the other
side of the seam, in ``benchmark/s5/tests/test_run_matrix.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from latence_benchmark.campaign.retrieval_stage import (
    STAGE_NAME,
    MatrixRequest,
    MatrixSummary,
    harness_module,
    request_for,
    run_s5_stage,
    s5_done_probe,
    s5_progress_probe,
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
from latence_benchmark.campaign.wizard import CampaignAnswers, S5Answers

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_ctx(
    tmp_path: Path,
    *,
    datasets: tuple[str, ...] = ("musique", "vidoseek"),
    s5: S5Answers | None = None,
    repo_root: Path | None = None,
) -> CampaignContext:
    root = tmp_path / "camp"
    return CampaignContext(
        campaign_root=root,
        repo_root=repo_root if repo_root is not None else REPO_ROOT,
        answers=CampaignAnswers(
            campaign_root=str(root), datasets=datasets, s5=s5 or S5Answers()
        ),
    )


def write_cells(
    ctx: CampaignContext, statuses: dict[tuple[str, str], str], *, fusion: str = "equal"
) -> None:
    """Write ``{(dataset, leg): status}`` as real cell files where the harness puts them."""
    directory = stage_dir(ctx.campaign_root, STAGE_NAME) / (
        "results" if fusion == "equal" else "results-weighted"
    )
    directory.mkdir(parents=True, exist_ok=True)
    for (dataset, leg), status in statuses.items():
        (directory / f"{dataset}__{leg}.json").write_text(
            json.dumps({"status": status, "dataset": dataset, "leg": leg}) + "\n",
            encoding="utf-8",
        )


def write_gate_artifacts(ctx: CampaignContext) -> None:
    out = stage_dir(ctx.campaign_root, STAGE_NAME)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text("# S5\n", encoding="utf-8")
    (out / "metrics.json").write_text("{}\n", encoding="utf-8")


def full_matrix(ctx: CampaignContext, status: str = "ok") -> dict[tuple[str, str], str]:
    request = request_for(ctx)
    return {(d, leg): status for d in request.datasets for leg in request.legs}


# -- plan derivation ---------------------------------------------------------------------------


class TestPlanDerivation:
    def test_an_empty_s5_datasets_follows_the_corpus_selection(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, datasets=("musique", "uda"))
        assert request_for(ctx).datasets == ("musique", "uda")

    def test_an_explicit_s5_datasets_wins(self, tmp_path: Path) -> None:
        ctx = make_ctx(
            tmp_path,
            datasets=("musique", "uda"),
            s5=S5Answers(datasets=("vidoseek",)),
        )
        assert request_for(ctx).datasets == ("vidoseek",)

    def test_legs_are_canonicalised_into_matrix_order(self, tmp_path: Path) -> None:
        # The order is the report's column order, so it is the table's, never the operator's.
        ctx = make_ctx(tmp_path, s5=S5Answers(legs=("packer", "dense")))
        assert request_for(ctx).legs == ("dense", "packer")

    def test_the_output_root_is_the_stage_directory(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path)
        assert request_for(ctx).out == tmp_path / "camp" / "stages" / STAGE_NAME

    def test_limit_and_fusion_come_straight_from_the_block(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, s5=S5Answers(limit=50, fusion="tuned"))
        request = request_for(ctx)
        assert (request.limit, request.fusion) == (50, "tuned")


# -- probes ------------------------------------------------------------------------------------


class TestProbes:
    def test_done_needs_every_planned_cell(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense", "bm25")))
        assert not s5_done_probe(ctx)
        write_cells(ctx, {("musique", "dense"): "ok"})
        write_gate_artifacts(ctx)
        assert not s5_done_probe(ctx)  # one cell short is not done
        assert s5_progress_probe(ctx) == "1/2 matrix cell(s) written"
        write_cells(ctx, {("musique", "bm25"): "ok"})
        assert s5_done_probe(ctx)
        assert s5_progress_probe(ctx) == "2/2 matrix cell(s) written"

    def test_done_needs_the_gate_artifacts_too(self, tmp_path: Path) -> None:
        # A hand-filled cell directory is not something an operator can sign off on: approve
        # is an approval OF REPORT.md + metrics.json, and the harness writes them last.
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense",)))
        write_cells(ctx, {("musique", "dense"): "ok"})
        assert not s5_done_probe(ctx)
        write_gate_artifacts(ctx)
        assert s5_done_probe(ctx)

    def test_editing_the_corpus_selection_reopens_the_stage(self, tmp_path: Path) -> None:
        one = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense",)))
        write_cells(one, {("musique", "dense"): "ok"})
        write_gate_artifacts(one)
        assert s5_done_probe(one)
        two = make_ctx(tmp_path, datasets=("musique", "vidoseek"), s5=S5Answers(legs=("dense",)))
        assert not s5_done_probe(two)

    def test_the_fusion_variant_has_its_own_cell_namespace(self, tmp_path: Path) -> None:
        ctx = make_ctx(
            tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense",), fusion="tuned")
        )
        write_cells(ctx, {("musique", "dense"): "ok"}, fusion="equal")
        write_gate_artifacts(ctx)
        assert not s5_done_probe(ctx)  # the baseline's cell is not the tuned variant's
        write_cells(ctx, {("musique", "dense"): "ok"}, fusion="tuned")
        assert s5_done_probe(ctx)

    def test_probes_are_unsatisfied_when_the_harness_is_absent(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, repo_root=tmp_path / "not-a-checkout")
        assert not s5_done_probe(ctx)
        assert "not importable" in s5_progress_probe(ctx)


# -- gate --------------------------------------------------------------------------------------


class TestGating:
    def test_s5_is_registered_last_and_gated_on_s3(self, tmp_path: Path) -> None:
        names = [spec.name for spec in registry()]
        assert names[-1] == STAGE_NAME
        assert names[-2] == "s3-pipeline"

        specs = registry()
        ctx = make_ctx(tmp_path)
        state = CampaignState(ctx.campaign_root / STATE_FILENAME, names)
        with pytest.raises(GateRefusedError, match="s3-pipeline"):
            run_stage(specs, STAGE_NAME, ctx, state)


# -- the run + the floor -----------------------------------------------------------------------


def stub_runner(
    ctx: CampaignContext, statuses: dict[tuple[str, str], str]
) -> tuple[list[MatrixRequest], object]:
    """A fake harness that writes the given cell statuses and records the request it got."""
    seen: list[MatrixRequest] = []

    def runner(request: MatrixRequest) -> MatrixSummary:
        seen.append(request)
        write_cells(ctx, statuses)
        write_gate_artifacts(ctx)
        return MatrixSummary(
            written=len(statuses),
            present=len(statuses),
            planned=len(request.datasets) * len(request.legs),
            report=request.out / "REPORT.md",
            metrics=request.out / "metrics.json",
        )

    return seen, runner


class TestRun:
    def test_the_stage_hands_the_harness_its_own_derived_plan(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense",), limit=8))
        seen, runner = stub_runner(ctx, full_matrix(ctx))
        note = run_s5_stage(ctx, runner)  # type: ignore[arg-type]

        (request,) = seen
        assert request.datasets == ("musique",)
        assert request.legs == ("dense",)
        assert request.limit == 8
        assert request.repo_root == ctx.repo_root
        assert "1 dataset(s) × 1 leg(s) = 1 cell(s), 1 measured" in note

    def test_a_dataset_whose_every_leg_refused_blocks(self, tmp_path: Path) -> None:
        # The per-leg refusal is legitimate (the reused M6 exports carry no embedding column);
        # a whole DATASET of refusals is an absent export being advanced as a measurement.
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense", "bm25")))
        _seen, runner = stub_runner(ctx, full_matrix(ctx, "missing-inputs"))
        with pytest.raises(StageBlockedError, match="musique: every leg recorded a refusal"):
            run_s5_stage(ctx, runner)  # type: ignore[arg-type]

    def test_a_partly_refusing_dataset_is_fine(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense", "bm25")))
        _seen, runner = stub_runner(
            ctx, {("musique", "dense"): "missing-inputs", ("musique", "bm25"): "ok"}
        )
        assert "1 measured" in run_s5_stage(ctx, runner)  # type: ignore[arg-type]

    def test_a_run_that_writes_nothing_is_blocked_by_the_probe_not_the_note(
        self, tmp_path: Path
    ) -> None:
        # The supervise.sh rule, inherited: the done-probe is the authority, not the exit
        # path. Here the harness returns cleanly having written only HALF the matrix — the
        # floor is satisfied (the dataset measured something), and the probe still refuses.
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense", "bm25")))
        _seen, runner = stub_runner(ctx, {("musique", "dense"): "ok"})
        specs = [
            spec if spec.name != STAGE_NAME else replace(spec, run=lambda c: run_s5_stage(c, runner))  # type: ignore[arg-type]
            for spec in registry()
        ]
        state = CampaignState(
            ctx.campaign_root / STATE_FILENAME, [spec.name for spec in specs]
        )
        for earlier in ("s1-corpus", "s2-ocr", "s3-pipeline"):
            state.mark_done(earlier)
            state.approve(earlier)
        with pytest.raises(StageBlockedError, match="done-probe is unsatisfied"):
            run_stage(specs, STAGE_NAME, ctx, state, fresh=True)
        assert state.status_of(STAGE_NAME) == "blocked"

    def test_the_runner_blocks_loudly_when_the_harness_is_absent(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, repo_root=tmp_path / "not-a-checkout")
        with pytest.raises(StageBlockedError, match="S5 harness is not importable"):
            run_s5_stage(ctx)


class TestStateFile:
    def test_the_four_stage_state_file_is_byte_exact(self, tmp_path: Path) -> None:
        """STATE.json after the matrix completes — pinned to the byte, injected clock.

        The registry grew a stage, so this is what a campaign's recorded state now IS. The
        clock is the ticking test clock (``state.py``'s seam), so the bytes are a property of
        the transitions, not of when the test ran.
        """
        ctx = make_ctx(tmp_path, datasets=("musique",), s5=S5Answers(legs=("dense",)))
        names = [spec.name for spec in registry()]
        counter = iter(range(1, 99))
        state = CampaignState(
            ctx.campaign_root / STATE_FILENAME, names, clock=lambda: f"t{next(counter)}"
        )
        for earlier in ("s1-corpus", "s2-ocr", "s3-pipeline"):
            state.mark_done(earlier)
            state.approve(earlier)
        _seen, runner = stub_runner(ctx, full_matrix(ctx))
        specs = [
            spec
            if spec.name != STAGE_NAME
            else replace(spec, run=lambda c: run_s5_stage(c, runner))  # type: ignore[arg-type]
            for spec in registry()
        ]
        run_stage(specs, STAGE_NAME, ctx, state)

        assert (ctx.campaign_root / STATE_FILENAME).read_text(encoding="utf-8") == (
            '{\n  "schema": 1,\n  "stages": [\n'
            '    {\n      "approved_at": "t2",\n      "ended": "t1",\n'
            '      "name": "s1-corpus",\n      "reason": "",\n      "started": "",\n'
            '      "status": "approved"\n    },\n'
            '    {\n      "approved_at": "t4",\n      "ended": "t3",\n'
            '      "name": "s2-ocr",\n      "reason": "",\n      "started": "",\n'
            '      "status": "approved"\n    },\n'
            '    {\n      "approved_at": "t6",\n      "ended": "t5",\n'
            '      "name": "s3-pipeline",\n      "reason": "",\n      "started": "",\n'
            '      "status": "approved"\n    },\n'
            '    {\n      "approved_at": "",\n      "ended": "t9",\n'
            '      "name": "s5-retrieval",\n      "reason": "",\n      "started": "t7",\n'
            '      "status": "done"\n    }\n  ],\n  "updated": "t10"\n}\n'
        )


class TestHarnessSeam:
    def test_the_real_harness_loads_from_the_named_checkout(self) -> None:
        # The seam resolves against repo_root, not against whatever is already importable.
        cell = harness_module(REPO_ROOT, "cell")
        assert cell.RESULTS_DIRS == {"equal": "results", "tuned": "results-weighted"}

    def test_a_checkout_without_the_harness_is_a_ModuleNotFoundError(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ModuleNotFoundError, match="no S5 harness at"):
            harness_module(tmp_path, "cell")
