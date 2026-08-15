# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The stage registry and the gated runner — supervise.sh's loop, importable and testable.

Three supervise.sh rules survive the port unchanged:

* **The done-probe is the authority, not the exit path.** A run callable that returns
  cleanly without its probe passing is ``blocked``, exactly like the corpus fetch that
  exited 0 having left two of three sources empty.
* **The done-probe runs BEFORE any work**, which is what makes ``run`` and ``resume`` safe
  to invoke at any moment, including after a reboot mid-stage.
* **``blocked`` stops the campaign at that stage.** Later stages consume this one's output;
  running them now would measure the previous state of the world and report it as this run.

One rule is new: **stage N+1 refuses until stage N is ``approved``** — the operator gate
(``latence-campaign approve``). ``done`` means the machine finished; ``approved`` means a
human looked at the gate artifact (metrics.json, the manifest) and signed off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from latence_benchmark.campaign.state import (
    APPROVED,
    BLOCKED,
    DONE,
    CampaignState,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from latence_benchmark.campaign.wizard import CampaignAnswers


class GateRefusedError(RuntimeError):
    """``run`` was refused by the gate (previous stage not approved, terminal state, …)."""


class StageBlockedError(RuntimeError):
    """A stage failed and was recorded ``blocked``; the message is the recorded reason."""


@dataclass(frozen=True)
class CampaignContext:
    """Everything a stage callable may read. Stages get context, never globals.

    :param campaign_root: The campaign directory (campaign.yaml, STATE.json, stages/).
    :param repo_root: The latence-framework checkout the stage operates on — corpus stages
        write under ``<repo_root>/benchmark/sota-corpus`` and reference
        ``<repo_root>/benchmark/official-data``.
    :param answers: The parsed campaign.yaml.
    :param accept_non_commercial: The operator's explicit research-only acknowledgement,
        threaded into the license gate (:func:`latence_benchmark.licensing.require_usable`)
        and recorded in the stage metrics. Never defaulted on.
    """

    campaign_root: Path
    repo_root: Path
    answers: CampaignAnswers
    accept_non_commercial: bool = False


@dataclass(frozen=True)
class StageSpec:
    """One campaign stage: name, probes, and the work itself.

    :param name: Stable identifier (also the ``stages/<name>/`` directory name).
    :param title: One human line for the status table.
    :param done_probe: Asks the FILESYSTEM whether the stage's work exists. Cheap, pure
        read; the runner consults it before and after ``run``.
    :param progress_probe: A monotonic-ish observable (a count, a byte size) rendered as a
        string for the status table. Advisory only in this runner.
    :param run: Does the work. Returns a one-line human note; raises
        :class:`StageBlockedError` (or anything else) to block the stage with that reason.
    """

    name: str
    title: str
    done_probe: Callable[[CampaignContext], bool]
    progress_probe: Callable[[CampaignContext], str]
    run: Callable[[CampaignContext], str]


def stage_dir(campaign_root: Path, name: str) -> Path:
    """``<campaign_root>/stages/<name>/`` — where a stage's metrics + artifacts live."""
    return campaign_root / "stages" / name


def registry() -> tuple[StageSpec, ...]:
    """The campaign's stages, in run order. Later tickets append.

    Imported lazily so the stage modules can themselves import this module's types
    without a cycle. Order IS the gate chain: s2-ocr refuses until s1-corpus is approved.
    """
    from latence_benchmark.campaign.corpus_stage import CORPUS_STAGE
    from latence_benchmark.campaign.ocr_stage import OCR_STAGE
    from latence_benchmark.campaign.pipeline_stage import S3_PIPELINE_STAGE
    from latence_benchmark.campaign.retrieval_stage import S5_RETRIEVAL_STAGE

    return (CORPUS_STAGE, OCR_STAGE, S3_PIPELINE_STAGE, S5_RETRIEVAL_STAGE)


def find_stage(specs: Sequence[StageSpec], name: str) -> tuple[int, StageSpec]:
    """Return ``(index, spec)`` for ``name`` or raise :class:`GateRefusedError`."""
    for index, spec in enumerate(specs):
        if spec.name == name:
            return index, spec
    known = ", ".join(s.name for s in specs)
    msg = f"Unknown stage {name!r}. Stages, in order: {known}."
    raise GateRefusedError(msg)


def run_stage(
    specs: Sequence[StageSpec],
    name: str,
    ctx: CampaignContext,
    state: CampaignState,
    *,
    fresh: bool = False,
) -> str:
    """Run one stage under the gate rules; returns a one-line report for the operator.

    The sequence, in order of what it protects against:

    1. **Gate**: the previous stage must be ``approved`` (not merely ``done``) — refused
       otherwise. First stage has no previous.
    2. **Terminal states**: a ``done``/``approved`` stage is a NO-OP unless ``fresh``;
       a ``blocked`` stage is refused unless ``fresh`` (blocked is terminal-with-reason).
    3. **Probe on entry**: if the done-probe is already satisfied the stage is marked
       ``done`` without running — resuming never redoes finished work.
    4. **Run, then probe again**: the probe, not the return, decides ``done``. Any raise or
       an unsatisfied probe records ``blocked`` with the reason and raises
       :class:`StageBlockedError`.

    Every observable transition is saved atomically before the next step begins.
    """
    index, spec = find_stage(specs, name)

    if index > 0:
        previous = specs[index - 1]
        prev_status = state.status_of(previous.name)
        if prev_status != APPROVED:
            msg = (
                f"Stage {name!r} is gated on {previous.name!r}, which is {prev_status!r} — "
                "not approved. Inspect its gate artifact and record the sign-off with "
                f"`latence-campaign approve {previous.name}` first."
            )
            raise GateRefusedError(msg)

    status = state.status_of(name)
    if status in (DONE, APPROVED) and not fresh:
        return f"{name}: already {status} — no-op (use --fresh to redo it)."
    if status == BLOCKED and not fresh:
        msg = (
            f"Stage {name!r} is BLOCKED: {state.stage(name).reason} "
            "Blocked is terminal — fix the cause, then re-run with --fresh."
        )
        raise GateRefusedError(msg)
    if fresh:
        state.reset(name)
        state.save()

    if spec.done_probe(ctx):
        state.mark_done(name)
        state.save()
        return f"{name}: done-probe satisfied on entry — marked done without running."

    state.mark_running(name)
    state.save()
    try:
        note = spec.run(ctx)
    except StageBlockedError as exc:
        state.mark_blocked(name, str(exc))
        state.save()
        raise
    except Exception as exc:  # Any escape blocks the stage, carrying its reason.
        reason = f"{type(exc).__name__}: {exc}"
        state.mark_blocked(name, reason)
        state.save()
        raise StageBlockedError(reason) from exc

    # Exit and completion are independent questions (supervise.sh's core lesson): the
    # filesystem probe is the authority on whether the work exists.
    if not spec.done_probe(ctx):
        reason = (
            "run returned cleanly but the done-probe is unsatisfied — the stage's artifact "
            "does not exist, so the stage is NOT complete."
        )
        state.mark_blocked(name, reason)
        state.save()
        raise StageBlockedError(reason)

    state.mark_done(name)
    state.save()
    return f"{name}: done — {note}"
