"""The bake-off result contracts + their signed-style rendering (ADR-0036 §4, P3-F5).

The per-Stage **bake-off** holds every other Stage at ``stacks/default.yaml`` and runs the SAME
bundled corpus (+ gold-set) through each candidate Provider for ONE Stage, recording — per
candidate — the enterprise decision columns (ADR-0036 §4/§6):

* **quality** — the S10 gold-set P/R/F1 where the Stage has a gold set (NER/RE), else a
  stage-specific metric (parse: chars-extracted + offset drift; chunk: chunk count + coverage).
* **latency / throughput** — measured wall-clock for the target Stage (docs/sec). Wall-clock is
  machine-relative — reported, never gated on.
* **peak memory** — measured (``tracemalloc`` Python allocations, with the H-E1 off-heap-RSS
  caveat) **vs the declared ``profile.memory_mb``** — surfacing declared-vs-measured is the point.
* **cost** — ``profile.cost_per_1k`` (endpoint providers), else n/a.
* **license** — ``profile.license`` + whether verified (H-B1) + any opt-in flag.
* **determinism** — ``profile.deterministic``, cross-checked by running the candidate twice and
  asserting byte-identical output (the H-E1 equality helper); a Provider that *declares*
  ``deterministic=True`` but is not is FLAGGED (``determinism_mismatch``).
* **error rate** — from the run's :class:`~latence_core.quality.StageMetrics` ``error_count``.

This module is the **pure contract + rendering half** — Pydantic models + a signed-style Markdown
renderer + the machine-readable JSON companion, with **no runner/stage imports** — so it lives in
the same dependency-light tier as :mod:`latence_core.conformance.report` and a bare
``import latence_core`` stays thin. The measuring half (running each candidate, reading the Quality
Report + ``tracemalloc``) is :mod:`latence_core.bakeoff.harness`; it produces these models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# The bake-off result schema version. Bumped on any change to the recorded columns, so a persisted
# ``MATRIX-RESULTS-<stage>.json`` companion carries the shape it was written under (like
# ``QUALITY_SCHEMA_VERSION`` for the Quality Report).
BAKEOFF_SCHEMA_VERSION = 1


class QualityMetric(BaseModel):
    """The quality of one candidate on the target Stage — gold-set score OR a stage-specific metric.

    The bake-off reuses the S10 gold-set scorer where the Stage has a gold set (the
    :class:`~latence_core.quality.GoldSetEvaluation` the run already computed): ``kind="goldset"``
    and ``precision``/``recall``/``f1`` carry the target Stage's overall score. Where the Stage has
    NO gold set (parse/chunk), ``kind="metric"`` and ``metrics`` carries the stage-specific numbers
    (parse: ``chars_extracted`` + ``offset_drift_recovered``; chunk: ``chunks_total`` +
    ``offset_preserving_fraction``) — pulled from the Quality Report's per-Stage rollup, NOT
    hand-rolled. ``primary`` is the single headline number the results table sorts/leads on (the F1
    for a gold-set Stage; a stage-chosen headline for a metric Stage), so a reader compares
    candidates at a glance.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="'goldset' (P/R/F1) or 'metric' (stage-specific numbers).")
    primary_name: str = Field(description="Name of the headline metric (e.g. 'f1', 'chars').")
    primary: float = Field(description="The headline number the table leads on.")
    precision: float | None = Field(default=None, description="Gold-set precision (goldset kind).")
    recall: float | None = Field(default=None, description="Gold-set recall (goldset kind).")
    f1: float | None = Field(default=None, description="Gold-set F1 (goldset kind).")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Stage-specific metric numbers (metric kind)."
    )


class MemoryMeasurement(BaseModel):
    """Measured peak Python allocations vs declared ``profile.memory_mb`` (declared vs measured).

    The audit flagged ``ProviderProfile.memory_mb`` inert because its consumer didn't exist; the
    bake-off IS that consumer. ``measured_mb`` is the ``tracemalloc`` peak Python-heap allocation
    across the target Stage's run, in MiB. ``declared_mb`` is ``profile.memory_mb`` (or ``None`` if
    the Provider declares no profile). The H-E1 honesty caveat: ``tracemalloc`` tracks **Python
    allocations only** — a Provider whose weights live in an off-heap C/torch buffer (a GPU model)
    is under-counted here, which is exactly why the CPU pure-Python candidates are the honest
    measurement and the GPU rows are skipped-with-flag (measured on a rig, #64). ``caveat`` states
    this on every row so the number is never read as an RSS ceiling.
    """

    model_config = ConfigDict(extra="forbid")

    measured_mb: float = Field(description="Peak tracemalloc Python-heap allocation (MiB).")
    declared_mb: int | None = Field(default=None, description="profile.memory_mb (declared).")
    caveat: str = Field(
        default="tracemalloc Python allocations only; off-heap/native (GPU) memory not counted.",
        description="The H-E1 off-heap-RSS honesty caveat, stated on every row.",
    )

    @property
    def delta_mb(self) -> float | None:
        """Measured − declared (MiB), or ``None`` when the Provider declares no memory footprint."""
        if self.declared_mb is None:
            return None
        return round(self.measured_mb - self.declared_mb, 3)


class LicenseColumn(BaseModel):
    """The license decision column for a candidate (ADR-0012 / H-B1 evidence-bearing).

    ``spdx`` is ``profile.license`` (the headline SPDX or the ``UNVERIFIED`` sentinel);
    ``verified`` is ``profile.license_verified`` (True only with a citation + date, H-B1);
    ``opt_in`` flags a restricted-but-permitted license (e.g. a Gemma-licensed weight that is
    opt-in per ADR-0012) so an adopter sees the flow-down obligation. ``source`` carries the
    citation (where a human checked). All read from the declared profile — never fabricated; a
    Provider with no profile leaves ``spdx=None``.
    """

    model_config = ConfigDict(extra="forbid")

    spdx: str | None = Field(default=None, description="profile.license (SPDX or UNVERIFIED).")
    verified: bool = Field(default=False, description="profile.license_verified (H-B1).")
    opt_in: bool = Field(default=False, description="Restricted-but-permitted (opt-in) license.")
    source: str | None = Field(default=None, description="Citation for the verification.")


class DeterminismColumn(BaseModel):
    """Declared vs cross-checked determinism for a candidate — the determinism check that BITES.

    ``declared`` is ``profile.deterministic``. ``observed_identical`` is whether the candidate's
    target-Stage output was **byte-identical** across two independent runs (the H-E1 export-artifact
    equality helper, reused — NOT a reinvented comparison). ``mismatch`` is the flag the DoD
    requires: a Provider that declares ``deterministic=True`` but whose two runs diverged is a
    LYING declaration and is flagged (``mismatch=True``). A Provider that honestly declares
    ``deterministic=False`` is never a mismatch even if it happened to reproduce. ``checked`` is
    False only when the candidate was skipped (no two runs to compare).
    """

    model_config = ConfigDict(extra="forbid")

    declared: bool | None = Field(default=None, description="profile.deterministic (declared).")
    observed_identical: bool | None = Field(
        default=None, description="Two runs byte-identical (H-E1 equality)."
    )
    checked: bool = Field(default=True, description="False when the candidate was skipped.")

    @property
    def mismatch(self) -> bool:
        """True when a Provider declares deterministic=True but its two runs diverged (a lie)."""
        return (
            self.checked
            and self.declared is True
            and self.observed_identical is False
        )


class CandidateResult(BaseModel):
    """One candidate Provider's full bake-off row: every ADR-0036 §4 dimension for one Stage.

    A row is either **run** (``skipped=False`` and every measured column populated) or
    **skipped-with-flag** (``skipped=True`` — a GPU/endpoint candidate on a CPU-only host, or a
    Provider not installed; ``skip_reason`` states which and references #64) OR **failed**
    (``failed=True`` — the candidate crashed / failed conformance during its run; the bake-off
    degrades per-candidate and records the failure as a row rather than aborting the whole matrix).
    A skipped row NEVER carries a fabricated number — its measured columns stay ``None`` (the honest
    boundary the whole workstream holds).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(description="The candidate Provider's registered name.")
    skipped: bool = Field(default=False, description="Skipped-with-flag (device/not-installed).")
    skip_reason: str | None = Field(default=None, description="Why skipped (references #64).")
    failed: bool = Field(default=False, description="The candidate crashed / failed mid-run.")
    failure_detail: str | None = Field(default=None, description="Type + message of the failure.")

    # -- measured columns (populated only for a run, non-skipped, non-failed candidate) --
    quality: QualityMetric | None = None
    documents: int = Field(default=0, description="Documents the target Stage processed.")
    stage_seconds: float | None = Field(
        default=None, description="Measured wall-clock for the target Stage (machine-relative)."
    )
    throughput_docs_per_s: float | None = Field(
        default=None, description="documents / stage_seconds (machine-relative; not gated)."
    )
    memory: MemoryMeasurement | None = None
    cost_per_1k: float | None = Field(
        default=None, description="profile.cost_per_1k (endpoint providers), else None (n/a)."
    )
    license: LicenseColumn = Field(default_factory=LicenseColumn)
    determinism: DeterminismColumn = Field(default_factory=DeterminismColumn)
    error_count: int = Field(default=0, description="StageMetrics.error_count for the Stage.")
    compute: str | None = Field(default=None, description="profile.compute (cpu|gpu|either|None).")
    model_id: str | None = Field(default=None, description="profile.model_id.")

    @property
    def ran(self) -> bool:
        """True for a candidate that actually ran + was measured (not skipped, not failed)."""
        return not self.skipped and not self.failed


class MatrixResult(BaseModel):
    """The full bake-off for one Stage: every candidate's row + the corpus/stack it ran on.

    The signed-style deliverable (``MATRIX-RESULTS-<stage>.md`` + its JSON companion). ``stage`` is
    the target Stage (a :class:`~latence_core.pipeline.CapabilityKind` value); ``candidates`` is one
    :class:`CandidateResult` per Provider in ``matrix/<stage>.yaml``, in declared order.
    ``base_stack`` names the stack whose OTHER stages were held constant (the controlled compare),
    and ``document_count`` the corpus size, so the report is self-describing.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=BAKEOFF_SCHEMA_VERSION)
    stage: str = Field(description="Target Stage (CapabilityKind) the candidates fulfil.")
    stage_name: str = Field(description="The Stage's name in the base stack (the swapped node).")
    base_stack: str = Field(description="Stack whose OTHER stages were held constant.")
    document_count: int = Field(default=0, description="Documents in the bundled bake-off corpus.")
    candidates: list[CandidateResult] = Field(default_factory=list)

    @property
    def any_flag(self) -> bool:
        """True if any candidate was skipped, failed, or flagged a determinism mismatch."""
        return any(
            c.skipped or c.failed or c.determinism.mismatch for c in self.candidates
        )

    def to_json(self) -> str:
        """The machine-readable companion (indented JSON) — numbers consumable, not just prose."""
        return self.model_dump_json(indent=2)

    def render(self) -> str:  # noqa: C901 - a flat table renderer; branching is per-column, not deep
        """A signed-style ``MATRIX-RESULTS-<stage>.md`` table (like ``G1-RESULTS.md``).

        One row per candidate × the ADR-0036 §4 dimensions + a skipped/why column. Deterministic
        (candidates in declared order, every number pre-rounded), so a seeded bake-off renders
        byte-identically. A skipped/failed candidate renders its reason in the last column, dashing
        its measured columns (never a fabricated number).
        """
        lines: list[str] = [
            f"# Bake-off — `{self.stage}` ({self.stage_name}) — provider matrix",
            "",
            f"- base stack (other stages held constant): `{self.base_stack}`",
            f"- corpus documents: {self.document_count}",
            f"- candidates: {len(self.candidates)}",
            "",
            "| candidate | quality | latency (docs/s) | memory measured/declared (MiB) | "
            "cost/1k | license | deterministic | errors | skipped/why |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in self.candidates:
            if c.skipped:
                why = c.skip_reason or "skipped"
                lines.append(
                    f"| `{c.provider}` | — | — | — | — | {self._license_cell(c)} | — | — | "
                    f"SKIPPED — {why} |"
                )
                continue
            if c.failed:
                why = c.failure_detail or "failed"
                lines.append(
                    f"| `{c.provider}` | — | — | — | — | {self._license_cell(c)} | — | — | "
                    f"FAILED — {why} |"
                )
                continue
            lines.append(
                f"| `{c.provider}` | {self._quality_cell(c)} | {self._latency_cell(c)} | "
                f"{self._memory_cell(c)} | {self._cost_cell(c)} | {self._license_cell(c)} | "
                f"{self._determinism_cell(c)} | {c.error_count} | — |"
            )
        lines.append("")
        lines.append(
            "_Latency/throughput are wall-clock and machine-relative (reported, not gated). "
            "Memory is tracemalloc Python allocations only (off-heap/native memory not counted — "
            "GPU rows are measured on a rig, #64). Skipped GPU/endpoint candidates carry no "
            "fabricated numbers._"
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _quality_cell(c: CandidateResult) -> str:
        q = c.quality
        if q is None:
            return "—"
        if q.kind == "goldset":
            return (
                f"F1={q.f1:.3f} P={q.precision:.3f} R={q.recall:.3f}"
                if q.f1 is not None and q.precision is not None and q.recall is not None
                else f"{q.primary_name}={q.primary:.3f}"
            )
        pieces = ", ".join(f"{k}={v:g}" for k, v in sorted(q.metrics.items()))
        return f"{q.primary_name}={q.primary:g} ({pieces})" if pieces else f"{q.primary:g}"

    @staticmethod
    def _latency_cell(c: CandidateResult) -> str:
        if c.stage_seconds is None or c.throughput_docs_per_s is None:
            return "—"
        return f"{c.throughput_docs_per_s:.1f} ({c.stage_seconds:.4f}s)"

    @staticmethod
    def _memory_cell(c: CandidateResult) -> str:
        m = c.memory
        if m is None:
            return "—"
        declared = "n/a" if m.declared_mb is None else str(m.declared_mb)
        return f"{m.measured_mb:.2f} / {declared}"

    @staticmethod
    def _cost_cell(c: CandidateResult) -> str:
        return "n/a" if c.cost_per_1k is None else f"{c.cost_per_1k:g}"

    @staticmethod
    def _license_cell(c: CandidateResult) -> str:
        lic = c.license
        if lic.spdx is None:
            return "no profile"
        marks = []
        marks.append("verified" if lic.verified else "unverified")
        if lic.opt_in:
            marks.append("opt-in")
        return f"{lic.spdx} ({', '.join(marks)})"

    @staticmethod
    def _determinism_cell(c: CandidateResult) -> str:
        d = c.determinism
        if not d.checked:
            return "—"
        declared = "?" if d.declared is None else ("yes" if d.declared else "no")
        observed = (
            "?"
            if d.observed_identical is None
            else ("identical" if d.observed_identical else "DIVERGED")
        )
        flag = " ⚠ MISMATCH" if d.mismatch else ""
        return f"declared={declared}, observed={observed}{flag}"
