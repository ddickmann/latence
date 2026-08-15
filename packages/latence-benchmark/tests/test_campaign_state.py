# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The campaign state machine: legal transitions, terminal blocked, atomic persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.campaign import state as state_mod
from latence_benchmark.campaign.state import (
    APPROVED,
    BLOCKED,
    DONE,
    PENDING,
    RUNNING,
    CampaignState,
    CampaignStateError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

STAGES = ["one", "two"]


def ticking_clock() -> Callable[[], str]:
    """A deterministic clock: t1, t2, t3 … — timestamps become assertable."""
    counter = iter(range(1, 1000))

    def clock() -> str:
        return f"t{next(counter)}"

    return clock


def make_state(tmp_path: Path) -> CampaignState:
    return CampaignState(tmp_path / "STATE.json", STAGES, clock=ticking_clock())


class TestTransitions:
    def test_happy_path_pending_running_done_approved(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        assert st.status_of("one") == PENDING
        st.mark_running("one")
        assert st.status_of("one") == RUNNING
        assert st.stage("one").started == "t1"
        st.mark_done("one")
        assert st.status_of("one") == DONE
        assert st.stage("one").ended == "t2"
        st.approve("one")
        assert st.status_of("one") == APPROVED
        assert st.stage("one").approved_at == "t3"

    def test_done_from_pending_probe_satisfied_on_entry(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_done("one")
        assert st.status_of("one") == DONE

    def test_blocked_is_terminal_for_running(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_running("one")
        st.mark_blocked("one", "the fetch died")
        assert st.status_of("one") == BLOCKED
        assert st.stage("one").reason == "the fetch died"
        with pytest.raises(CampaignStateError, match="terminal"):
            st.mark_running("one")

    def test_blocked_requires_a_reason(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_running("one")
        with pytest.raises(CampaignStateError, match="reason"):
            st.mark_blocked("one", "   ")

    def test_approve_refused_unless_done(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        with pytest.raises(CampaignStateError, match="not 'done'"):
            st.approve("one")
        st.mark_running("one")
        with pytest.raises(CampaignStateError, match="not 'done'"):
            st.approve("one")

    def test_rerun_of_done_requires_reset(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_running("one")
        st.mark_done("one")
        with pytest.raises(CampaignStateError, match="--fresh"):
            st.mark_running("one")
        st.reset("one")
        assert st.status_of("one") == PENDING
        st.mark_running("one")  # now legal again

    def test_unknown_stage_is_loud(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        with pytest.raises(CampaignStateError, match="Unknown stage"):
            st.mark_running("nope")


class TestPersistence:
    def test_roundtrip_preserves_every_field(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_running("one")
        st.mark_blocked("one", "boom")
        st.save()

        reloaded = CampaignState(tmp_path / "STATE.json", STAGES)
        assert reloaded.stage("one").status == BLOCKED
        assert reloaded.stage("one").reason == "boom"
        assert reloaded.stage("one").started == "t1"
        assert reloaded.stage("two").status == PENDING

    def test_new_registry_stage_starts_pending_over_old_file(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.mark_done("one")
        st.save()
        grown = CampaignState(tmp_path / "STATE.json", [*STAGES, "three"])
        assert grown.status_of("one") == DONE
        assert grown.status_of("three") == PENDING

    def test_crash_between_temp_and_rename_leaves_old_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        st = make_state(tmp_path)
        st.mark_done("one")
        st.save()
        before = (tmp_path / "STATE.json").read_bytes()

        def boom(src: object, dst: object) -> None:
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(state_mod, "_atomic_replace", boom)
        st.mark_running("two")
        with pytest.raises(OSError, match="simulated crash"):
            st.save()

        # The crash left the OLD state byte-identical — the temp file is the casualty,
        # never STATE.json.
        assert (tmp_path / "STATE.json").read_bytes() == before
        reloaded = CampaignState(tmp_path / "STATE.json", STAGES)
        assert reloaded.status_of("two") == PENDING

    def test_saved_file_is_valid_sorted_json(self, tmp_path: Path) -> None:
        st = make_state(tmp_path)
        st.save()
        raw = json.loads((tmp_path / "STATE.json").read_text())
        assert raw["schema"] == 1
        assert [entry["name"] for entry in raw["stages"]] == STAGES

    def test_corrupt_status_in_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "STATE.json"
        path.write_text(json.dumps({"stages": [{"name": "one", "status": "weird"}]}))
        with pytest.raises(CampaignStateError, match="unknown status"):
            CampaignState(path, STAGES)
