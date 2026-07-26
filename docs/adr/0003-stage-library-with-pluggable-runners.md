# Stage library with pluggable Runners, not an orchestration control plane

The framework core is a library of typed Stages bound by strict data contracts; execution is delegated to interchangeable Runners. A built-in local Runner (porting pipeline-worker's proven patterns: per-stage checkpoint/resume, DAG waves, stage-first streaming — with storage abstracted) powers the zero-infrastructure demo; thin adapters let the same Stages run inside whatever the adopter already operates (Databricks Workflows, Airflow, plain containers). We deliberately do NOT ship a control plane: enterprise platform teams reject frameworks that demand their own scheduler, and the private pipeline-worker's RunPod/Supabase/B2/Fly.io hard-wiring showed how fast a control plane accretes vendor lock-in. Rejected: generalizing pipeline-worker into a deployable orchestrator service (adoption friction), contracts-only with no execution (demo degenerates to notebooks, loses checkpoint/resume value), and standardizing on one engine like Temporal/Dagster (forces a dependency into client environments we cannot predict).

## Amendment (C3) — what a Runner adapter is actually allowed to supply

"Interchangeable Runners" was structurally under-specified, and the second Runner drifted because
of it. The only seam the Airflow adapter could reuse was `LocalRunner._execute_stage`, which
dispatched a Stage and drained one of three Provider tallies. Everything else a Runner needs to
*be* a Runner — provider load, device routing against the declared `ProviderProfile`, the
skip-with-flag posture (ADR-0036 §2), the schema-induction and relation-fan-out tallies, typed-error
categorisation, and the 22-column `StageMetrics` assembly — lived inline in the local Runner's
340-line `_run` loop, behind no interface at all. So the adapter re-implemented what it could and
silently dropped the rest: an Airflow run reported `compute=None`, `model_id=None`, `license=None`,
`device=None`, `skipped=False` and zero induction columns, and a GPU-only Provider on a CPU host
was skipped-with-flag locally but **crashed** under Airflow. Same Pipeline, two Quality Reports,
two failure modes.

This ADR is therefore sharpened, not reversed:

* **A Runner supplies exactly two things: scheduling and a substrate.** Which Stage runs when, and
  where its records and run-state live. The local Runner schedules topologically and persists to
  files on Storage (ADR-0010); the Airflow adapter delegates scheduling to Airflow and carries
  records over XCom; the DeltaRunner schedules one run and commits a Corpus Version.
* **Everything else is `latence_core.stage_execution.StageExecution`,** reached through
  `LocalRunner.stage_execution` — a public property precisely because it is the Runner seam.
  `execute(stage, ctx) -> StageOutcome` returns the Stage's records (lazily) *and* the metrics row
  that describes producing them. A Runner cannot forget a metrics column, because it never names
  one; and it cannot invent a different posture for an unrunnable Stage, because the posture is
  behind the same call.
* **`test_runner_conformance.py` is the enforcement.** It runs the identical Pipeline under both
  Runners and asserts every `StageMetrics` column agrees except `runner`, `duration_seconds` and
  `from_checkpoint` (the local Runner's resume concept — Airflow keeps its state in Airflow,
  ADR-0010). A third adapter is expected to be added to that parameterisation, not to re-derive
  a metrics row.

Rejected: leaving the dispatch as the seam and asking adapters to keep their metrics in sync by
review — that is precisely the arrangement that produced the degraded report, and nothing tested it.
