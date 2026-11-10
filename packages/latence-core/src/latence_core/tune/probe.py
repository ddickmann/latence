"""The Sample-Probe seam — how the tuner RUNS one Provider under one knob setting (W11).

The tuner's correctness gate + fastest-eligible pick are pure logic
(:mod:`latence_core.tune.tuner`),
but they need a way to actually *run* a Provider under a candidate :class:`KnobSetting` over the
small
real sample and get back either an ERROR or an (output-fingerprint, timings) pair. That is the
:class:`SampleProbe` protocol.

Two implementations:

* the **real** probe (:class:`RegistrySampleProbe`) — loads the Provider from the registry with the
  candidate knobs folded into its config, runs it on the staged sample, and returns a deterministic
  fingerprint of its output records (labels/spans/offsets/text — the correctness subject) plus timed
  runs. Heavy; only reached on a rig with the Provider installed.
* a **fake** probe (in the tests) — with a KNOWN knob→(speed, correctness, crash) profile, so the
  offline suite proves the gate PICKS the fastest correct setting, REJECTS a compile-crash
  (``BackendCompilerFailed`` at *inference*, the W9 lesson), and REJECTS an output change — with NO
  real GPU numbers.

The seam is deliberately narrow (one method), so a fake is trivial and the gate logic is identical
for real and fake Providers — the anti-false-green bar is met by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from latence_core.tune.report import KnobSetting


class ProbeOutcome(BaseModel):
    """The result of probing ONE Provider under ONE knob setting on the sample (W11).

    Exactly one of two shapes:

    * an **error** — ``error`` carries the exception the run raised (a load or, crucially, an
      *inference-time* fault like ``BackendCompilerFailed``); ``timings`` is empty. The gate REJECTS
      it (and names it precisely when it is a compile-backend failure).
    * a **success** — ``error`` is ``None``; ``fingerprint`` is a deterministic digest of the
      Provider's output records over the sample (the correctness subject — the gate compares it to
      the
      baseline's), and ``timings`` are the per-run wall-clock seconds (median → throughput).

    Not a Pydantic-validated ``error`` field (an exception is not JSON) — it is carried as a plain
    attribute set post-construction, so ``ProbeOutcome`` stays a light value object the fake builds
    directly.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fingerprint: str | None = Field(default=None, description="Digest of output (None on error).")
    timings: list[float] = Field(default_factory=list, description="Per-run wall-clock seconds.")
    error: BaseException | None = Field(default=None, description="The raised exception, or None.")

    @property
    def crashed(self) -> bool:
        """Whether the probe raised (the gate rejects a crashed candidate)."""
        return self.error is not None

    def median_seconds(self) -> float | None:
        """The median timed-run seconds (the throughput basis), or ``None`` if not timed."""
        if not self.timings:
            return None
        ordered = sorted(self.timings)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0


@runtime_checkable
class SampleProbe(Protocol):
    """Runs one Provider under one knob setting over the sample → a :class:`ProbeOutcome` (W11).

    The single seam the tuner drives. A probe MUST NOT raise for a candidate that crashes — it
    catches the fault and returns it in :attr:`ProbeOutcome.error` (so the gate rejects it as a
    documented rejection, never an aborted sweep). ``sample_size`` is the caller's N, passed through
    for a probe that can size its run; the registry probe
    (:class:`~latence_core.tune.realprobe.RegistrySampleProbe`) cannot and ignores it, timing the
    whole stack over the bundled validation corpus. ``repeats`` is the K timed runs after warmup
    whose median the tuner divides N by.
    """

    def probe(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        """Run ``provider`` under ``setting`` on the sample; never raise (fault → outcome.error)."""
        ...
