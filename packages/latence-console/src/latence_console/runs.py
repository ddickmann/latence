"""The run index — runs discovered by READING the files a run already left on Storage.

The console introduces **no new state** (ADR-0010: local Runner state *is* files on Storage;
ADR-0017: the AI-ready deliverable is files). There is no database, no registry, no sidecar
the console writes and later reads back. A run exists exactly when
``<storage_uri>/_latence/runs/<run_id>/manifest.json`` exists; its verdict comes from the
``quality-report.json`` the Runner already wrote next to it; its timestamps are those two
files' own modification times.

Everything here is read-only by construction: the only Storage methods this module can reach
are ``exists`` / ``ls`` / ``read_text`` / ``modified``. It never opens a file for writing,
never creates a directory, and never deletes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from latence_core.quality import QualityReport

# The run layout is owned by ``latence_core.run_store`` (Wave 2-E): the console re-exports the
# names it always exposed, now bound to the same constants every Runner writes with — the
# console can never drift from where a run actually lands.
from latence_core.run_store import (
    MANIFEST_NAME as MANIFEST_NAME,
)
from latence_core.run_store import (
    REPORT_NAME as REPORT_NAME,
)
from latence_core.run_store import (
    RUNS_PREFIX as RUNS_PREFIX,
)
from latence_core.run_store import RunLayout

from latence_console.artifacts import RunArtifacts

if TYPE_CHECKING:
    from datetime import datetime

    from latence_core.storage import Storage

# A run id is a single path segment the Runner minted (``run-0001``, ``demo``,
# ``delta-20260718T101500Z``). Anything else — a separator, a traversal dot-dot, a scheme — is
# NOT a run id and is refused before it is ever concatenated into a Storage URI, so a crafted
# console URL cannot walk out of the runs directory and read arbitrary files.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def is_run_id(candidate: str) -> bool:
    """True if ``candidate`` is a syntactically valid, traversal-free run id."""
    return bool(_RUN_ID_RE.fullmatch(candidate)) and ".." not in candidate


class RunStatus(Enum):
    """What a reader — including a non-engineer — should conclude about a run at a glance.

    Deliberately four plain outcomes, not the Runner's internal Stage bookkeeping:

    * ``PASSED`` — the run finished and every automated check held.
    * ``ATTENTION`` — the run finished, but a check it makes did not hold (a contract gap).
    * ``FAILED`` — a Stage raised; the run did not produce trustworthy output.
    * ``UNFINISHED`` — a manifest exists but no Quality Report: the run is still going or it
      died before it could report. (The console never *guesses* which; it says so.)
    """

    PASSED = "passed"
    ATTENTION = "attention"
    FAILED = "failed"
    UNFINISHED = "unfinished"

    @property
    def label(self) -> str:
        return {
            RunStatus.PASSED: "Passed",
            RunStatus.ATTENTION: "Needs attention",
            RunStatus.FAILED: "Failed",
            RunStatus.UNFINISHED: "Unfinished",
        }[self]

    @property
    def blurb(self) -> str:
        """One plain sentence a non-engineer can act on."""
        return {
            RunStatus.PASSED: "This run finished and every quality check held.",
            RunStatus.ATTENTION: (
                "This run finished, but at least one quality check did not hold. "
                "Read the checks below before using the output."
            ),
            RunStatus.FAILED: (
                "A step of this run failed. Its output is incomplete — do not use it."
            ),
            RunStatus.UNFINISHED: (
                "This run started but never wrote a Quality Report. "
                "It is either still running or it stopped early."
            ),
        }[self]


@dataclass(frozen=True)
class RunSummary:
    """One row of the run list: what it was, how it went, and when."""

    run_id: str
    pipeline_name: str
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    document_count: int
    duration_seconds: float
    is_delta: bool


class RunIndex:
    """Read-only view over the runs a Storage already holds.

    Accepts the :class:`~latence_core.storage.Storage` it reads through (it does not build
    one), so the same object serves a local folder, an S3 prefix, or an in-memory fixture.
    """

    def __init__(self, storage: Storage, storage_uri: str) -> None:
        self._storage = storage
        self._root = storage_uri.rstrip("/")

    @property
    def storage_uri(self) -> str:
        return self._root

    @property
    def runs_uri(self) -> str:
        return f"{self._root}/{RUNS_PREFIX}"

    def _layout(self, run_id: str) -> RunLayout:
        return RunLayout(self._root, run_id)

    def _run_dir(self, run_id: str) -> str:
        return self._layout(run_id).run_dir

    def artifacts(self, run_id: str) -> RunArtifacts:
        """A typed, read-only view over everything that run left on Storage (#188).

        Built fresh on every call: the console keeps no index and no cache between requests,
        so a run that grew a checkpoint since the last page load shows it on this one.
        """
        return RunArtifacts(self._storage, self._run_dir(run_id))

    def run_ids(self) -> list[str]:
        """Every run id under the runs root that actually has a manifest, sorted."""
        if not self._storage.exists(self.runs_uri):
            return []
        found: list[str] = []
        for entry in self._storage.ls(self.runs_uri):
            run_id = entry.rstrip("/").rsplit("/", 1)[-1]
            if not is_run_id(run_id):
                continue
            if self._storage.exists(self._layout(run_id).manifest_uri):
                found.append(run_id)
        return sorted(found)

    def report(self, run_id: str) -> QualityReport | None:
        """The run's Quality Report, or ``None`` when it has not written one (yet).

        A report that is present but unreadable (torn, hand-edited, produced by a newer
        schema with a shape this console cannot validate) is reported as absent rather than
        raising: the console's job is to stay legible over whatever is on disk.
        """
        if not is_run_id(run_id):
            return None
        uri = self._layout(run_id).report_uri
        if not self._storage.exists(uri):
            return None
        try:
            return QualityReport.model_validate_json(self._storage.read_text(uri))
        except (ValueError, UnicodeDecodeError, OSError):
            return None

    def _pipeline_name(self, run_id: str) -> str:
        uri = self._layout(run_id).manifest_uri
        try:
            payload = json.loads(self._storage.read_text(uri))
        except (ValueError, UnicodeDecodeError, OSError):
            return "unknown"
        pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
        if isinstance(pipeline, dict):
            name = pipeline.get("name")
            if isinstance(name, str) and name:
                return name
        return "unknown"

    def summary(self, run_id: str) -> RunSummary | None:
        """The list-row view of one run, or ``None`` if there is no such run."""
        if not is_run_id(run_id):
            return None
        manifest_uri = self._layout(run_id).manifest_uri
        if not self._storage.exists(manifest_uri):
            return None
        report = self.report(run_id)
        return RunSummary(
            run_id=run_id,
            pipeline_name=report.pipeline_name if report is not None else self._pipeline_name(
                run_id
            ),
            status=status_of(report),
            started_at=self._storage.modified(manifest_uri),
            finished_at=self._storage.modified(self._layout(run_id).report_uri),
            document_count=report.document_count if report is not None else 0,
            duration_seconds=report.total_duration_seconds if report is not None else 0.0,
            is_delta=report is not None and report.delta is not None,
        )

    def list_runs(self) -> list[RunSummary]:
        """Every run, newest first (runs with no readable timestamp sort last, by id)."""
        summaries = [s for s in (self.summary(r) for r in self.run_ids()) if s is not None]
        return sorted(
            summaries,
            key=lambda s: (s.started_at is not None, s.started_at, s.run_id),
            reverse=True,
        )


def status_of(report: QualityReport | None) -> RunStatus:
    """Derive the headline status from the report the Runner wrote (never from a stored flag).

    The order matters: a Stage that raised outranks a contract gap, which outranks success.
    ``contracts_complete`` on the report is a *mirror* of the completeness detail, so the
    verdict is computed from the detail — the headline can then never disagree with the
    numbers printed under it.
    """
    if report is None:
        return RunStatus.UNFINISHED
    if any(stage.error_count for stage in report.stages):
        return RunStatus.FAILED
    if not report.completeness.passed:
        return RunStatus.ATTENTION
    return RunStatus.PASSED
