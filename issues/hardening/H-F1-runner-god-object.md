# H-F1 — Split the `runner.py` god-object: extract `QualityReportBuilder`

**Source: hostile 360 audit, THEME F (mattpocock: deep module, not god-object; "the interface is the
reuse surface — if a second impl must reach past it, the shape is wrong").**
`packages/latence-core/src/latence_core/runner.py` is ~2610 lines mixing four responsibilities:
DAG scheduling, persistence (checkpoint/WAL), stage dispatch, and an ~830-line **Quality Report
builder** (`_build_report` + ~15 `_*_quality` methods, roughly lines 1777–2610). The Airflow runner
can only reuse the engine by reaching into **6 private `LocalRunner` methods via `# noqa: SLF001`**
(`_execute_stage`, `_build_report`, `_final_records`, `_run_dir`, `_report_uri`, `_drain_retry_count`)
— the real engine seam was faked with private-method access instead of extracted.

**This is a pure structural refactor: ZERO behaviour change.** The Quality Report must remain
byte-identical; determinism/resume/delta must be preserved. The full existing suite (including the
H-E1 byte-identical resume tests and the report/schema tests) is the safety net — it must stay green
without modification to its assertions.

## Primary deliverable — extract `QualityReportBuilder`
Lift the report-building half into its own module `latence_core/quality_report_builder.py` as a
cohesive unit that depends only on what it needs — the stage `outputs` mapping + `Storage` (+ the
run identity / config it reads), NOT the whole executor:
- `class QualityReportBuilder` with a small public surface (e.g. `build(...) -> QualityReport` /
  `write(...)`), owning `_build_report` and the `_*_quality` helpers.
- `LocalRunner` composes it: `LocalRunner._build_report` becomes a thin delegator to the builder (keep
  the private method as a one-line shim if anything else calls it, or update call sites).
- **Fix the Airflow SLF001 reach-through for report building:** the Airflow adapter uses the **public**
  `QualityReportBuilder` instead of `self._engine._build_report(...)`. (The remaining execution-side
  private reaches — `_execute_stage` etc. — may stay for this slice; if a clean `StageExecutor`
  extraction is low-risk, do it, but do NOT expand scope at the cost of the no-behaviour-change
  guarantee. At minimum, kill the `_build_report`/`_final_records`/`_report_uri` reaches.)

## Secondary (only if genuinely low-risk; skip if it endangers the no-behaviour-change bar)
- Collapse the 13-branch `isinstance` dispatch chain in `_dispatch_stage` into a
  `{CapabilityKind: DispatchSpec}` table (same behaviour, one row per capability).
- Collapse the 6 near-identical `_require_*_inputs` / `_require_*_inputs_streamed` wiring-check methods
  into one parametrized predicate, fixing the already-drifting duplicate error strings.
These are cleanups the audit named; include them **only** if they don't threaten byte-identical output.
If skipped, note them as follow-ups (don't force them into this slice).

## Definition of Done (Baseline bar + these)
- `QualityReportBuilder` extracted into its own module with a minimal public interface; `LocalRunner`
  composes it; the Quality Report is **byte-identical** to pre-refactor (prove it: a test comparing a
  report built through the new builder against a golden/pre-refactor fixture, or assert the existing
  report tests pass unchanged).
- The Airflow adapter builds its report through the **public** `QualityReportBuilder` — the
  `_build_report`/`_final_records`/`_report_uri` `SLF001` reaches are gone (grep-verify the noqa count
  dropped); Airflow tests still pass (the H-E1 Airflow CI lane stays green).
- Determinism/resume/delta/Purge re-verified (core-touching): the streaming-checkpoint byte-identical
  resume tests, delta chaos, and Purge tests pass **unchanged**.
- `runner.py` is meaningfully smaller and each remaining concern is legible; `mypy --strict` clean over
  src+tests; `ruff` clean; full suite green on py3.11+3.12.
- No public contract / SCHEMA_VERSION change (this is internal structure only); ADR note on the
  engine-seam extraction (why the report builder is now its own module, reused by both Runners).

## Reviewer should check
- **No behaviour change:** the Quality Report is byte-identical; no test assertion was weakened to make
  the refactor pass (the suite's expectations are unchanged, only import paths/wiring moved).
- The Airflow report path no longer reaches past the interface (SLF001 count down; it calls the public
  builder); the DAG CI lane is still green.
- `QualityReportBuilder` accepts its dependencies (outputs + storage), doesn't reach back into the
  Runner — the deletion test now passes for the report concern (it's a real, relocatable unit).
- Resume/determinism/delta genuinely re-verified, not assumed — this touches the core execution module.
