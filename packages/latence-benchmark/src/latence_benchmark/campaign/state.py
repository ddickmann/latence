# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Atomic, per-campaign stage state — the Python port of supervise.sh's ``STATE.json``.

Two properties are carried over from the shell supervisor verbatim, because both were paid
for in real incidents:

* **Atomic writes** (temp file + ``os.replace`` in the same directory), so a reader never
  observes a half-written state and a crash mid-save leaves the OLD state intact. STATE.json
  is never the thing that lies.
* **``blocked`` is terminal and carries a reason.** A supervisor that silently skips a broken
  stage and reports success at the end is worse than one that stops.

One property is new here: **``approved``**. A stage that finishes is ``done``; only the
operator's explicit ``approve`` (recorded with a timestamp) makes it ``approved``, which is
what the NEXT stage's gate checks. Timestamps come from an injected clock seam so every
transition is offline-testable to the byte.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from os import replace as _atomic_replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

#: The five stage statuses. ``pending -> running -> done -> approved`` is the happy path;
#: ``blocked`` is terminal-with-reason (only an explicit ``reset`` — the CLI's ``--fresh`` —
#: re-opens it).
PENDING = "pending"
RUNNING = "running"
DONE = "done"
BLOCKED = "blocked"
APPROVED = "approved"
STATUSES = frozenset({PENDING, RUNNING, DONE, BLOCKED, APPROVED})

STATE_FILENAME = "STATE.json"


def utc_now() -> str:
    """The default clock: UTC, second precision, the supervise.sh timestamp format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class CampaignStateError(RuntimeError):
    """An illegal stage-state transition or an unknown stage name."""


@dataclass(frozen=True)
class StageState:
    """One stage's recorded state. ``reason`` is only ever non-empty when ``blocked``."""

    name: str
    status: str = PENDING
    reason: str = ""
    started: str = ""
    ended: str = ""
    approved_at: str = ""


class CampaignState:
    """The stage-state machine of one campaign directory, persisted atomically.

    :param path: The ``STATE.json`` location (``<campaign_root>/STATE.json``).
    :param stage_names: The registry's stage names, in order. An existing file's entries are
        merged over these, so adding a stage to the registry later leaves prior campaigns
        loadable (the new stage simply starts ``pending``).
    :param clock: Timestamp source, injected for tests. Pure state transitions + an injected
        clock = byte-reproducible STATE.json in the offline suite.
    """

    def __init__(
        self,
        path: Path,
        stage_names: Sequence[str],
        *,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if not stage_names:
            msg = "A campaign needs at least one stage."
            raise CampaignStateError(msg)
        self.path = path
        self._clock = clock
        self.stages: dict[str, StageState] = {n: StageState(name=n) for n in stage_names}
        if path.exists():
            self._merge(json.loads(path.read_text(encoding="utf-8")))

    def _merge(self, raw: object) -> None:
        if not isinstance(raw, dict):
            msg = f"{self.path} does not hold a JSON object."
            raise CampaignStateError(msg)
        for entry in raw.get("stages", []):
            name = entry.get("name")
            if name not in self.stages:
                continue  # A stage removed from the registry: its record is simply dropped.
            status = entry.get("status", PENDING)
            if status not in STATUSES:
                msg = f"{self.path} records unknown status {status!r} for stage {name!r}."
                raise CampaignStateError(msg)
            self.stages[name] = StageState(
                name=name,
                status=status,
                reason=str(entry.get("reason", "")),
                started=str(entry.get("started", "")),
                ended=str(entry.get("ended", "")),
                approved_at=str(entry.get("approved_at", "")),
            )

    # -- persistence ---------------------------------------------------------------------

    def save(self) -> None:
        """Rewrite ``STATE.json`` in full, atomically (temp + rename, same directory).

        A crash between writing the temp file and the rename leaves the previous state
        intact — the temp file is garbage, never the state. That is the whole contract.
        """
        payload = {
            "schema": 1,
            "updated": self._clock(),
            "stages": [asdict(self.stages[name]) for name in self.stages],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _atomic_replace(tmp, self.path)

    # -- queries -------------------------------------------------------------------------

    def _require(self, name: str) -> StageState:
        try:
            return self.stages[name]
        except KeyError:
            known = ", ".join(self.stages)
            msg = f"Unknown stage {name!r}. Known stages: {known}."
            raise CampaignStateError(msg) from None

    def status_of(self, name: str) -> str:
        """The recorded status of ``name``."""
        return self._require(name).status

    def stage(self, name: str) -> StageState:
        """The full recorded :class:`StageState` of ``name``."""
        return self._require(name)

    # -- transitions ---------------------------------------------------------------------
    #
    # Each transition validates the source status, so an illegal move is a loud
    # CampaignStateError instead of a silently rewritten history. None of them saves —
    # callers batch transitions and call :meth:`save` (the runner saves around every
    # observable step).

    def mark_running(self, name: str) -> None:
        """``pending``/``running`` → ``running`` (records ``started``, clears any residue)."""
        st = self._require(name)
        if st.status == BLOCKED:
            msg = (
                f"Stage {name!r} is BLOCKED ({st.reason}). Blocked is terminal — "
                "re-run it with --fresh once the cause is fixed."
            )
            raise CampaignStateError(msg)
        if st.status in (DONE, APPROVED):
            msg = (
                f"Stage {name!r} is already {st.status}. Re-running it requires --fresh "
                "(which resets its record first)."
            )
            raise CampaignStateError(msg)
        self.stages[name] = replace(
            st, status=RUNNING, started=self._clock(), ended="", reason=""
        )

    def mark_done(self, name: str) -> None:
        """``running``/``pending`` → ``done``.

        ``pending`` is legal because a done-probe satisfied ON ENTRY (the work already
        exists on disk — a resumed campaign) completes a stage without it ever running here.
        """
        st = self._require(name)
        if st.status not in (RUNNING, PENDING):
            msg = f"Stage {name!r} cannot become done from {st.status!r}."
            raise CampaignStateError(msg)
        self.stages[name] = replace(st, status=DONE, ended=self._clock(), reason="")

    def mark_blocked(self, name: str, reason: str) -> None:
        """Any non-terminal status → ``blocked``. The reason is mandatory, not decorative."""
        if not reason.strip():
            msg = f"Blocking stage {name!r} requires a reason; an unexplained stop is a bug."
            raise CampaignStateError(msg)
        st = self._require(name)
        if st.status == APPROVED:
            msg = f"Stage {name!r} is approved; it cannot be blocked without --fresh."
            raise CampaignStateError(msg)
        self.stages[name] = replace(st, status=BLOCKED, ended=self._clock(), reason=reason)

    def approve(self, name: str) -> None:
        """``done`` → ``approved`` — the operator's recorded sign-off, nothing else's."""
        st = self._require(name)
        if st.status != DONE:
            msg = (
                f"Stage {name!r} is {st.status!r}, not 'done' — only a finished stage can be "
                "approved. Run it first (and let its done-probe pass)."
            )
            raise CampaignStateError(msg)
        self.stages[name] = replace(st, status=APPROVED, approved_at=self._clock())

    def reset(self, name: str) -> None:
        """Back to ``pending`` (the CLI's ``--fresh``): clears timestamps and reason."""
        self._require(name)
        self.stages[name] = StageState(name=name)
