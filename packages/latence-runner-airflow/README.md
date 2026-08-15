# latence-runner-airflow

A **second Runner** for the Latence framework (S12, ADR-0003): run the *identical* Pipeline Stages
under **Apache Airflow**, so an adopter keeps their own orchestrator instead of adopting ours.
Execution is pluggable — the Stage semantics are the framework's, the scheduling and run-state are
Airflow's.

## What differs from the local Runner

The framework core is a library of typed Stages; Runners own scheduling, retries, and checkpointing
(ADR-0003). This adapter **reuses** the local Runner's Stage-execution and report-building seams
verbatim (no fork of the Stage dispatch), but:

- **State/scheduling is Airflow's, not files-on-Storage** (ADR-0010). One Airflow task per Stage,
  wired by the Pipeline's `depends_on` edges; Airflow owns which tasks ran, their retries, and DAG
  bookkeeping. Stage outputs flow between tasks over Airflow **XCom** — there is no local
  `_latence/runs/.../checkpoints` run state.
- Every `StageMetrics` in the Quality Report is stamped `runner: "airflow"`.

## Adopter path — build a real DAG

```python
from latence_core import Pipeline
from latence_runner_airflow import AirflowRunner

pipeline = Pipeline.load("pipeline.yaml", storage)
dag = AirflowRunner().build_dag(pipeline, dag_id="latence-corpus")  # needs the [airflow] extra
```

Drop the `dag` into your Airflow `dags/` folder. Airflow schedules the Stage tasks in order and
retries a failed task (default `retries=2`) using its own state.

## Fault tolerance without a torn Corpus Version

Two layers absorb endpoint failures: the Provider's ported retry (backoff + jitter) inside a Stage,
and Airflow's task-level retries around a whole Stage task. A Corpus Version is only committed after
the DAG run finishes, so a mid-DAG endpoint failure — retried or ultimately failed — leaves the last
committed Corpus Version intact (S12 AC): a failed task fails the DAG run before any commit, never a
torn one.

## Offline proof

`AirflowRunner().run(pipeline, run_id)` executes the same per-Stage task path **in process** (an
Airflow-shaped topological driver + an XCom stand-in) with no Airflow install — the deterministic
end-to-end proof that the identical Stages run under a second Runner.

## Licensing

Framework code, Apache-2.0 (the repository-wide licence; commercial use granted by it, no
agreement needed). Apache Airflow (the optional `[airflow]` extra) is Apache-2.0.
The adapter is offline and Airflow-free unless you build a real DAG.
