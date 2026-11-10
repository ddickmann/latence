"""The tune result contracts + their signed-style rendering + the tuned-config (de)serialisation.

The pure contract half of :mod:`latence_core.tune` (Pydantic models + a Markdown renderer + the
``tuned/<device>.yaml`` load/dump), with **no runner/stage imports**, so it sits in the same
dependency-light tier as :mod:`latence_core.bakeoff.report` and a bare ``import latence_core`` stays
thin. The measuring half (running each candidate through a Provider, timing it, correctness-gating)
is :mod:`latence_core.tune.tuner`; it produces these models.

The honest-report discipline (ADR-0036 §4, carried to W11): a :class:`ProviderTuning` records the
winning knobs + measured/baseline throughput + speedup AND every REJECTED knob with the reason it
was
rejected (a crash — named precisely when it is a compile-backend failure — or an output change
beyond
tolerance). Winners AND rejections, never a fabricated number.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from latence_core.tune.fingerprint import DeviceFingerprint

# The tuned-config schema version — bumped on any change to the recorded columns / overlay shape, so
# a persisted ``tuned/<device>.yaml`` carries the shape it was written under.
TUNE_SCHEMA_VERSION = 1


class KnobSetting(BaseModel):
    """One candidate perf-knob setting — the dict overlaid onto a stage config if it wins (W11).

    ``knobs`` is exactly the sub-set of config keys this candidate pins (``dtype``, ``compile``,
    ``batch_size``, ``ocr_batch_size``, ``max_concurrency``, ``quantize``); the empty dict is the
    baseline ("no knobs" = the Provider's own defaults). Frozen so a probed candidate is an
    immutable
    fact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    knobs: dict[str, Any] = Field(default_factory=dict, description="Config keys this pins.")

    @property
    def label(self) -> str:
        """A short, stable label for the report (e.g. ``dtype=bfloat16,compile=True``)."""
        if not self.knobs:
            return "baseline"
        return ",".join(f"{k}={self.knobs[k]}" for k in sorted(self.knobs))


class RejectedKnob(BaseModel):
    """A candidate the correctness gate REJECTED — with the honest reason (W11, the crux).

    ``reason`` is a short machine-readable category (``crash`` / ``compile_failure`` /
    ``output_changed``
    / ``skipped``); ``detail`` is a bounded human string (a capped exception summary, or the
    tolerance
    that was exceeded). ``compile_failure`` is called out distinctly from a generic ``crash``
    because
    catching the inference-time compile fault (the W9 lesson) is the whole point — the report names
    it
    precisely ("compile: rejected (BackendCompilerFailed)").
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    setting: KnobSetting
    reason: str = Field(description="'crash' | 'compile_failure' | 'output_changed' | 'skipped'.")
    detail: str = Field(default="", description="A bounded human-readable reason (no PII).")


class ProviderTuning(BaseModel):
    """The tuned result for ONE Provider — the winner AND every rejection (honest report, W11).

    * ``winner`` — the fastest ELIGIBLE :class:`KnobSetting` (correct + non-crashing). Always
      present:
      the baseline itself is eligible by construction (it defines the correctness reference), so a
      Provider on which every acceleration was rejected still wins the baseline (slow-but-correct).
    * ``baseline_throughput`` / ``winner_throughput`` — the report's ``sample_size`` divided by the
      median wall seconds of the timed runs. The real probe times a run of the whole stack over the
      bundled validation corpus and does not cut it down to ``sample_size`` records, so the figure
      compares settings on one device and is not a count of records per second. The baseline is
      the fp32/plain reference; the winner's is what the tune buys.
    * ``speedup`` — ``winner_throughput / baseline_throughput`` (``1.0`` when the baseline wins).
    * ``rejected`` — the REJECTED candidates with reasons (compile crash, output change, …) — the
      honest half a reader audits.
    * ``eligible_count`` / ``candidate_count`` — how many of the swept candidates passed the gate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    winner: KnobSetting
    baseline_throughput: float | None = Field(
        default=None,
        description=(
            "sample_size divided by the median wall seconds of the baseline's timed runs. "
            "Comparable between settings on one device; not a count of records."
        ),
    )
    winner_throughput: float | None = Field(
        default=None,
        description=(
            "sample_size divided by the median wall seconds of the winner's timed runs. "
            "Comparable between settings on one device; not a count of records."
        ),
    )
    speedup: float | None = Field(default=None, description="winner/baseline throughput ratio.")
    speedup_suspect: bool = Field(
        default=False,
        description=(
            "The measured speedup exceeded the plausible ceiling even after a warm re-measure — "
            "flagged as a probable measurement fault (a cold-baseline leak), never signed as real."
        ),
    )
    candidate_count: int = Field(default=0, description="Total swept candidates.")
    eligible_count: int = Field(default=0, description="Candidates that passed the gate.")
    rejected: list[RejectedKnob] = Field(default_factory=list)

    def overlay_config(self) -> dict[str, Any]:
        """The winning knob dict to overlay onto this Provider's stage config (empty = baseline)."""
        return dict(self.winner.knobs)


class TuneReport(BaseModel):
    """The full tune result for one stack on one device — the ``tuned/<device>.yaml`` payload.

    Carries the :class:`DeviceFingerprint` (the overlay match key), the sample size + seed the sweep
    used (so it is reproducible), and the per-Provider tunings. Serialised to / from
    ``tuned/<device>.yaml`` via :meth:`to_yaml` / :meth:`from_yaml_text`; the runner overlay
    (:mod:`latence_core.tune.overlay`) reads it back and applies the per-Provider knobs ONLY when
    the
    current device's fingerprint matches ``device_fingerprint``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=TUNE_SCHEMA_VERSION)
    stack: str = Field(description="The stack file name the tune was measured against.")
    sample_size: int = Field(
        description=(
            "The numerator of every figure in this report (the --sample value). It is not the "
            "size of the corpus a candidate ran on."
        )
    )
    seed: int = Field(description="The seed the sweep used (determinism).")
    device_fingerprint: DeviceFingerprint
    providers: list[ProviderTuning] = Field(default_factory=list)

    def per_provider(self) -> dict[str, ProviderTuning]:
        """Map each tuned Provider name to its :class:`ProviderTuning`."""
        return {p.provider: p for p in self.providers}

    def to_yaml(self) -> str:
        """Serialise to the committed ``tuned/<device>.yaml`` text (deterministic key order)."""
        text: str = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        return text

    @classmethod
    def from_yaml_text(cls, text: str) -> TuneReport:
        """Parse a ``tuned/<device>.yaml`` back into a validated :class:`TuneReport`."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = "tuned config YAML must be a mapping at the top level."
            raise ValueError(msg)
        return cls.model_validate(data)

    def render(self) -> str:
        """A signed-style table: per-Provider winner + speedup + REJECTED knobs (honest)."""
        fp = self.device_fingerprint
        lines = [
            f"# Tune report — {self.stack}",
            "",
            f"- **device**: {fp.slug()} (key `{fp.key}`)",
            f"  - cuda={fp.cuda} gpu={fp.gpu_name or 'n/a'} vram_mb={fp.gpu_memory_mb or 'n/a'} "
            f"cpu={fp.cpu_count or 'n/a'} torch={fp.torch_version or 'n/a'} "
            f"cuda_ver={fp.cuda_version or 'n/a'}",
            f"- **sample_size**: {self.sample_size}  **seed**: {self.seed}  "
            f"**schema**: v{self.schema_version}",
            "- **figure**: sample_size divided by the median wall seconds of the timed runs. A "
            "timed run is one run of the whole stack over the bundled validation corpus, so a "
            "figure compares settings on this device and is not records per second.",
            "",
            "| provider | winner | baseline figure | winner figure | speedup | eligible/total |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for p in self.providers:
            base = "n/a" if p.baseline_throughput is None else f"{p.baseline_throughput:.3f}"
            win = "n/a" if p.winner_throughput is None else f"{p.winner_throughput:.3f}"
            if p.speedup is None:
                spd = "n/a"
            elif p.speedup_suspect:
                # Never sign a fabricated multiplier: an implausible ratio (a cold-baseline leak)
                # is flagged, not printed as a real speedup.
                spd = f"SUSPECT (~{p.speedup:.0f}x — likely a measurement fault)"
            else:
                spd = f"{p.speedup:.3f}x"
            lines.append(
                f"| {p.provider} | {p.winner.label} | {base} | {win} | {spd} | "
                f"{p.eligible_count}/{p.candidate_count} |"
            )
        lines.append("")
        lines.append("## Rejected knobs (the honest half)")
        any_rejected = False
        for p in self.providers:
            for rej in p.rejected:
                any_rejected = True
                lines.append(
                    f"- **{p.provider}** — `{rej.setting.label}`: rejected "
                    f"({rej.reason}){f' — {rej.detail}' if rej.detail else ''}"
                )
        if not any_rejected:
            lines.append("- (none — every swept candidate passed the correctness gate)")
        lines.append("")
        lines.append(
            "_Figures are machine-relative (this device only). A tuned config applies ONLY on a "
            "matching device_fingerprint; a mismatch falls back to stack defaults (never silently "
            "applied). No number here is fabricated — each is computed from the median of the "
            "timed runs._"
        )
        return "\n".join(lines)
