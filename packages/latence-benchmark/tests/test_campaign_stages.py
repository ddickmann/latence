# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The gated runner: approval gates, probe authority, blocked-is-terminal, no-op re-runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.campaign.stages import (
    CampaignContext,
    GateRefusedError,
    StageBlockedError,
    StageSpec,
    registry,
    run_stage,
)
from latence_benchmark.campaign.state import APPROVED, BLOCKED, DONE, CampaignState
from latence_benchmark.campaign.wizard import CampaignAnswers

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FakeStage:
    """A controllable stage: the test flips ``artifact`` to steer the done-probe."""

    name: str
    artifact: bool = False
    fail_with: str = ""
    produce: bool = True  # whether run() creates the artifact
    runs: list[str] = field(default_factory=list)

    def spec(self) -> StageSpec:
        def done(_ctx: CampaignContext) -> bool:
            return self.artifact

        def progress(_ctx: CampaignContext) -> str:
            return "0"

        def run(_ctx: CampaignContext) -> str:
            self.runs.append(self.name)
            if self.fail_with:
                raise StageBlockedError(self.fail_with)
            if self.produce:
                self.artifact = True
            return "ok"

        return StageSpec(
            name=self.name, title=self.name, done_probe=done, progress_probe=progress, run=run
        )


@pytest.fixture
def ctx(tmp_path: Path) -> CampaignContext:
    return CampaignContext(
        campaign_root=tmp_path / "campaign",
        repo_root=tmp_path / "repo",
        answers=CampaignAnswers(campaign_root=str(tmp_path / "campaign")),
    )


def make(tmp_path: Path, *stages: FakeStage) -> tuple[list[StageSpec], CampaignState]:
    specs = [s.spec() for s in stages]
    state = CampaignState(tmp_path / "campaign" / "STATE.json", [s.name for s in specs])
    return specs, state


class TestGate:
    def test_second_stage_refused_until_first_is_approved(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        first, second = FakeStage("a"), FakeStage("b")
        specs, state = make(tmp_path, first, second)

        with pytest.raises(GateRefusedError, match="gated on 'a'"):
            run_stage(specs, "b", ctx, state)

        run_stage(specs, "a", ctx, state)
        assert state.status_of("a") == DONE
        # done is NOT enough — the gate wants the operator's approval.
        with pytest.raises(GateRefusedError, match="not approved"):
            run_stage(specs, "b", ctx, state)

        state.approve("a")
        run_stage(specs, "b", ctx, state)
        assert state.status_of("b") == DONE
        assert second.runs == ["b"]

    def test_first_stage_has_no_gate(self, tmp_path: Path, ctx: CampaignContext) -> None:
        stage = FakeStage("a")
        specs, state = make(tmp_path, stage)
        assert "done" in run_stage(specs, "a", ctx, state)

    def test_unknown_stage_is_refused(self, tmp_path: Path, ctx: CampaignContext) -> None:
        specs, state = make(tmp_path, FakeStage("a"))
        with pytest.raises(GateRefusedError, match="Unknown stage"):
            run_stage(specs, "nope", ctx, state)


class TestProbeAuthority:
    def test_probe_satisfied_on_entry_completes_without_running(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        stage = FakeStage("a", artifact=True)
        specs, state = make(tmp_path, stage)
        note = run_stage(specs, "a", ctx, state)
        assert "without running" in note
        assert stage.runs == []
        assert state.status_of("a") == DONE

    def test_clean_exit_without_artifact_is_blocked(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        stage = FakeStage("a", produce=False)  # exits 0, produces nothing — the classic lie
        specs, state = make(tmp_path, stage)
        with pytest.raises(StageBlockedError, match="done-probe is unsatisfied"):
            run_stage(specs, "a", ctx, state)
        assert state.status_of("a") == BLOCKED

    def test_failure_blocks_with_reason(self, tmp_path: Path, ctx: CampaignContext) -> None:
        stage = FakeStage("a", fail_with="the license gate refused musique")
        specs, state = make(tmp_path, stage)
        with pytest.raises(StageBlockedError, match="license gate refused"):
            run_stage(specs, "a", ctx, state)
        assert state.stage("a").reason == "the license gate refused musique"

    def test_unexpected_exception_blocks_with_typed_reason(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        def run(_ctx: CampaignContext) -> str:
            raise ValueError("surprise")

        spec = StageSpec(
            name="a",
            title="a",
            done_probe=lambda _c: False,
            progress_probe=lambda _c: "0",
            run=run,
        )
        state = CampaignState(tmp_path / "campaign" / "STATE.json", ["a"])
        with pytest.raises(StageBlockedError, match="ValueError: surprise"):
            run_stage([spec], "a", ctx, state)
        assert state.stage("a").reason == "ValueError: surprise"


class TestRerunSemantics:
    def test_done_stage_is_noop_unless_fresh(self, tmp_path: Path, ctx: CampaignContext) -> None:
        stage = FakeStage("a")
        specs, state = make(tmp_path, stage)
        run_stage(specs, "a", ctx, state)
        assert stage.runs == ["a"]

        note = run_stage(specs, "a", ctx, state)
        assert "no-op" in note
        assert stage.runs == ["a"]  # did not run again

        stage.artifact = False  # operator wiped the artifact; --fresh redoes the work
        run_stage(specs, "a", ctx, state, fresh=True)
        assert stage.runs == ["a", "a"]
        assert state.status_of("a") == DONE

    def test_blocked_is_terminal_without_fresh(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        stage = FakeStage("a", fail_with="boom")
        specs, state = make(tmp_path, stage)
        with pytest.raises(StageBlockedError):
            run_stage(specs, "a", ctx, state)

        with pytest.raises(GateRefusedError, match="terminal"):
            run_stage(specs, "a", ctx, state)

        stage.fail_with = ""  # cause fixed
        run_stage(specs, "a", ctx, state, fresh=True)
        assert state.status_of("a") == DONE

    def test_state_survives_on_disk_between_runs(
        self, tmp_path: Path, ctx: CampaignContext
    ) -> None:
        stage = FakeStage("a")
        specs, state = make(tmp_path, stage)
        run_stage(specs, "a", ctx, state)
        state.approve("a")
        state.save()

        # A brand-new process loads the same campaign dir and sees the approval.
        reloaded = CampaignState(tmp_path / "campaign" / "STATE.json", ["a"])
        assert reloaded.status_of("a") == APPROVED


def test_registry_contains_the_corpus_stage_first() -> None:
    specs = registry()
    assert [s.name for s in specs][0] == "s1-corpus"
