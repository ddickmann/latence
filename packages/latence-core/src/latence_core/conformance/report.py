"""Conformance report models — the counts-only, Quality-Report-safe verdict types (P3-F3).

These are pure Pydantic contracts with **no** stage/provider imports, so the Quality Report
(:mod:`latence_core.quality`) can carry a :class:`ConformanceReport` without pulling the whole
conformance harness (and its reference-Provider fixtures) into the lean contract module. The
harness (:mod:`latence_core.conformance.harness`) produces these; the report folds them in.

Counts only — never record content — so a persisted Quality Report carrying a conformance report
stays safe to share (the same no-PII discipline the whole report keeps).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CheckStatus(StrEnum):
    """The outcome of one conformance check for one Provider."""

    PASS = "pass"
    FAIL = "fail"
    #: The check did not apply and was recorded, not run (a GPU Provider skipped on a CPU host;
    #: a non-deterministic Provider's C4 declaration). Never a silent skip — it is surfaced.
    SKIPPED = "skipped"


class CheckOutcome(BaseModel):
    """One check's result for one Provider: status + a short counts-only detail (no PII)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: str
    status: CheckStatus
    detail: str = ""


class ProviderConformance(BaseModel):
    """The full C1–C6 result for one Provider — the per-Provider conformance verdict."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    capability: str
    #: True for an endpoint/GPU Provider verified against a stub — its real numbers need a rig
    #: (#64); the harness surfaces this so a reader never mistakes a stub-pass for a live number.
    stub_scoped: bool = False
    #: True when the Provider was skipped-with-flag by the device seam (a GPU Provider on a
    #: CPU-only host) — the C5 posture; the C1/C6 record checks are then not run.
    device_skipped: bool = False
    skip_reason: str | None = None
    checks: list[CheckOutcome] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no check FAILed (SKIPPED/PASS are both acceptable)."""
        return all(c.status is not CheckStatus.FAIL for c in self.checks)

    def check_named(self, name: str) -> CheckOutcome | None:
        """The outcome of the named check, or ``None`` if this Provider has no such check."""
        for c in self.checks:
            if c.check == name:
                return c
        return None


class ConformanceReport(BaseModel):
    """The roll-up of every Provider's conformance verdict — a Quality-Report-safe artifact.

    Counts only (no record content), so it is safe to persist into the Quality Report. The
    conformance suite is the enterprise-readiness gate (ADR-0036 §5); this report is its
    machine-readable output — the audit that substantiates "model-agnostic" is proven, not
    asserted.
    """

    model_config = ConfigDict(extra="forbid")

    providers_total: int = 0
    providers_passed: int = 0
    providers_failed: int = 0
    providers_stub_scoped: int = 0
    providers_device_skipped: int = 0
    results: list[ProviderConformance] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.providers_failed == 0
