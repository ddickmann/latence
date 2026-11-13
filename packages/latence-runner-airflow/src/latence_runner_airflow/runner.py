"""Airflow Runner adapter — the same Stages under a second Runner (S12, ADR-0003, ADR-0010).

The framework core is a library of typed Stages; execution is delegated to interchangeable Runners
(ADR-0003). This is the proof of that seam's generality: an adapter that runs the **identical**
Pipeline Stages inside **Apache Airflow**, so an adopter keeps their own orchestrator instead of
adopting ours. The Stage *semantics* are not re-implemented — the adapter reuses
:class:`latence_core.runner.LocalRunner`'s public
:class:`~latence_core.stage_execution.StageExecution` seam verbatim (no fork) and builds its
Quality Report through the engine's
**public** :class:`~latence_core.quality_report_builder.QualityReportBuilder`
(:attr:`~latence_core.runner.LocalRunner.quality_report_builder`) — the report concern is a
relocatable unit reused through the interface, not a private reach-through (H-F1). What differs is
**where scheduling and run-state live**:

* The **local** Runner persists run state — manifest, per-Stage checkpoints, resume metadata — as
  files on Storage (ADR-0010). That is *its* state model and it does not try to be a control plane.
* This **Airflow** adapter deliberately uses **Airflow's own state and scheduling** (ADR-0010:
  "Adopter-grade Runners deliberately use their own platform's state and scheduling, not this").
  One Airflow task per Stage, wired by the Pipeline's ``depends_on`` edges into an Airflow DAG;
  Airflow owns which tasks ran, their retries, and the DAG-run bookkeeping. Stage outputs flow
  between tasks over Airflow **XCom**, not the local checkpoint files. There is no
  ``_latence/runs/.../checkpoints`` state here.

Endpoint fault-tolerance rides two layers without corrupting the corpus: the Provider's own ported
retry (:mod:`latence_core.endpoint`) absorbs transient endpoint blips inside a Stage, and Airflow's
**task-level retries** re-run a whole Stage task on an unhandled failure. The run's data artifacts
are committed by an explicit **report/commit task** that :meth:`build_dag` wires downstream of every
sink Stage (and that the in-process :meth:`run` performs as its final step): it builds and persists
the Quality Report only once every Stage has succeeded. Because that commit task is downstream of
every Stage, a mid-DAG endpoint failure — retried or ultimately failed — fails the DAG run *before*
the commit task runs, so the last committed state is left intact (S12 AC): a failed Airflow task
fails the DAG run before any commit, never torn. (The adapter commits the run's Quality Report as
its data artifact; a full transactional Corpus Version bump is the Delta Runner's concern, not this
seam's.)

``airflow`` is a heavy optional dependency isolated in THIS adapter package (ADR-0016), imported
lazily only in :meth:`AirflowRunner.build_dag`; the in-process execution path
(:meth:`AirflowRunner.run`) needs no Airflow install, so the same Stage-under-a-second-Runner proof
runs offline and deterministically in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.pipeline import Pipeline
from latence_core.quality import StageMetrics
from latence_core.run_store import decode_records, stage_output_type
from latence_core.runner import LocalRunner, StageOutput, topological_order
from latence_core.stage_execution import StageContext
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Callable

    from latence_core.capability import ProviderRegistry
    from latence_core.pipeline import Stage
    from latence_core.quality import QualityReport

# The Runner name stamped onto every StageMetrics this adapter produces (proves pluggable execution
# in the Quality Report, ADR-0003).
RUNNER_NAME = "airflow"


class XComBackend:
    """A minimal key→records store standing in for Airflow XCom (the inter-task carrier).

    Under real Airflow each Stage task pushes its output list to XCom keyed by the Stage name and
    every downstream task pulls its parents' outputs — exactly the wiring :class:`XComBackend`
    models. In the in-process path this is a plain dict; the Airflow DAG path uses Airflow's real
    XCom (the task functions call ``ti.xcom_push`` / ``ti.xcom_pull``), so the *same* task callable
    works under both.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[StageOutput]] = {}

    def push(self, stage_name: str, records: list[StageOutput]) -> None:
        self._store[stage_name] = records

    def pull(self, stage_name: str) -> list[StageOutput]:
        return self._store.get(stage_name, [])

    def gather(self, stage: Stage) -> list[StageOutput]:
        """A Stage's input = the concatenation of ALL its parents' XCom outputs (declared order)."""
        gathered: list[StageOutput] = []
        for parent in stage.depends_on:
            gathered.extend(self.pull(parent))
        return gathered

    def as_outputs(self) -> dict[str, list[StageOutput]]:
        return dict(self._store)


class AirflowRunner:
    """Runs a Pipeline's Stages under Airflow (or an in-process Airflow-shaped executor).

    Reuses :class:`~latence_core.runner.LocalRunner`'s Stage-execution and report-building seams so
    the Stage semantics are byte-identical to the local Runner; only scheduling and run-state differ
    (Airflow's, not files-on-Storage — ADR-0010). Two entry points:

    * :meth:`run` — execute the DAG **in process** in topological order, driving the same per-Stage
      task callable Airflow would, with an :class:`XComBackend` for inter-task hand-off. Needs no
      Airflow install, so it is the deterministic offline proof (the S12 end-to-end test path).
    * :meth:`build_dag` — construct a real Airflow ``DAG`` of ``PythonOperator`` tasks (one per
      Stage), for an adopter to drop into their Airflow deployment. Imports ``airflow`` lazily.
    """

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        # The LocalRunner is reused purely as the Stage-execution + report-building engine — we
        # never call its file-state `run()`. ADR-0003 "same Stages, pluggable Runner": no fork
        # of the Stage dispatch, only a different scheduler/state around it.
        self._engine = LocalRunner(registry)

    # -- in-process execution (the offline, deterministic proof) ------------

    def run(self, pipeline: Pipeline, run_id: str = "run-0001") -> QualityReport:
        """Execute the Pipeline's Stages in topological order via the Airflow-shaped task path.

        Airflow owns ordering and state here (modelled by the topological driver + XCom); there is
        no local checkpoint/manifest/WAL. Screening sidecars and Export outputs are still written to
        Storage (they are *data artifacts*, not run state), so the shared report builder produces
        the same Quality Report — now stamped ``runner="airflow"`` on every Stage.
        """
        # This path never calls the engine's ``run()``, so it makes the two refusals itself: an
        # unfilled template placeholder, and a config key ADR-0068 removed.
        pipeline.require_runnable()
        storage = Storage.for_uri(pipeline.storage_uri)
        # The Export Stage writes into this run's export dir; reuse the engine's own layout so the
        # data artifacts land where the report builder expects them (public run_dir, H-F1).
        storage.makedirs(self._engine.run_dir(pipeline, run_id))

        ordered = topological_order(list(pipeline.stages))
        xcom = XComBackend()
        stage_metrics: list[StageMetrics] = []
        for stage in ordered:
            stage_metrics.append(self._run_stage_task(pipeline, run_id, stage, storage, xcom))

        outputs = xcom.as_outputs()
        # H-F1: build the Quality Report through the engine's PUBLIC QualityReportBuilder — the same
        # report seam the local Runner uses, reused through the interface, not a private reach into
        # ``_build_report`` / ``_final_records`` / ``_report_uri``. Airflow's XCom holds every
        # Stage's output (nothing released), so the builder reads straight from ``outputs`` — the
        # #69 checkpoint fallback is only for the local Runner's released entries and never taken.
        builder = self._engine.quality_report_builder
        final_records = builder.final_records(
            ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        report = builder.build(
            pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
        )
        # The report is a data artifact of the run; persist it alongside the exports (the engine's
        # own report location), so an adopter reads it the same way regardless of Runner.
        storage.write_atomic(
            self._engine.report_uri(pipeline, run_id),
            report.to_json().encode("utf-8"),
        )
        return report

    def _run_stage_task(
        self,
        pipeline: Pipeline,
        run_id: str,
        stage: Stage,
        storage: Storage,
        xcom: XComBackend,
    ) -> StageMetrics:
        """The in-process driver's per-Stage task body — the same one the operator callable runs."""
        records, metrics = _execute(
            self._engine, pipeline, run_id, stage, storage, xcom.gather(stage),
            {parent: xcom.pull(parent) for parent in stage.depends_on},
        )
        xcom.push(stage.name, records)
        return metrics

    # -- real Airflow DAG construction (for an adopter's deployment) --------

    def build_dag(
        self,
        pipeline: Pipeline,
        *,
        dag_id: str | None = None,
        run_id: str = "run-0001",
        default_args: dict[str, Any] | None = None,
        **dag_kwargs: Any,
    ) -> Any:
        """Build a real Airflow ``DAG`` of one ``PythonOperator`` per Stage plus a report/commit
        task (adopter path).

        Wires one operator per Stage by the Pipeline's ``depends_on`` edges so Airflow schedules the
        Stages in the right order and owns their state/retries (ADR-0010). Each Stage operator's
        callable is :meth:`_airflow_stage_callable`, which runs the Stage through the same shared
        engine and pushes/pulls via Airflow's real XCom — so the Stage semantics match the local
        Runner.

        A final ``<dag_id>.report`` task (:meth:`_airflow_report_callable`) is wired **downstream of
        every sink Stage** (issue #44.4): it gathers each Stage's records + metrics from XCom,
        builds the Quality Report through the same shared public ``QualityReportBuilder`` seam the
        ``run()`` path uses, and persists it — the explicit commit of the run's data artifacts.
        Because it is
        downstream of every Stage, Airflow only schedules it once all Stages have succeeded, so a
        mid-DAG Stage failure fails the DAG run *before* the report/commit task runs and leaves the
        last committed Corpus Version intact (never torn). Before this task the real DAG emitted no
        Quality Report and had no commit step; the ADR/`run()`-path claim of a "commit downstream of
        every Stage" now actually holds on the adopter path too.

        ``airflow`` is imported here (lazily) so the rest of the adapter — and the in-process
        proof — needs no Airflow install (ADR-0016).
        """
        # A DAG built from a template would carry the unfilled placeholder into every task, and
        # one built with a removed config key would run tasks that ignore it.
        pipeline.require_runnable()
        try:
            from airflow import DAG
            from airflow.operators.python import PythonOperator
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'apache-airflow' package is required to build an Airflow DAG. "
                "Install it with: pip install latence-runner-airflow[airflow]"
            )
            raise ImportError(msg) from exc

        args: dict[str, Any] = {"retries": 2, **(default_args or {})}
        dag = DAG(
            dag_id=dag_id or f"latence-{pipeline.name}",
            default_args=args,
            catchup=False,
            **dag_kwargs,
        )
        operators: dict[str, Any] = {}
        for stage in pipeline.stages:
            operators[stage.name] = PythonOperator(
                task_id=stage.name,
                python_callable=self._airflow_stage_callable(pipeline, run_id, stage),
                dag=dag,
            )
        for stage in pipeline.stages:
            for parent in stage.depends_on:
                operators[parent] >> operators[stage.name]

        # The report/commit task (issue #44.4): downstream of every sink Stage, so Airflow runs it
        # only after all Stages succeed. It builds + persists the Quality Report (the commit of the
        # run's data artifacts). A mid-DAG failure fails the run before it, so no torn commit.
        ordered = topological_order(list(pipeline.stages))
        depended_on = {p for s in ordered for p in s.depends_on}
        sinks = [s.name for s in ordered if s.name not in depended_on]
        report_op = PythonOperator(
            task_id="report",
            python_callable=self._airflow_report_callable(pipeline, run_id),
            dag=dag,
        )
        for sink in sinks:
            operators[sink] >> report_op
        return dag

    def _airflow_stage_callable(
        self, pipeline: Pipeline, run_id: str, stage: Stage
    ) -> Callable[..., None]:
        """A Stage's Airflow task function: pull parents from XCom, run the Stage, push the output.

        Uses Airflow's real XCom (via the task-instance ``ti`` Airflow injects) as the inter-task
        carrier — so Airflow owns the run state, not local checkpoint files (ADR-0010). Records are
        serialized to their contract JSON for XCom transport and re-validated on pull, so the
        typed contract survives the Airflow boundary intact. The Stage also pushes its
        :class:`~latence_core.quality.StageMetrics` (records in/out, duration, runner name, and the
        endpoint-retry tally its Provider reported) under a ``"metrics"`` key so the downstream
        report/commit task can assemble the Quality Report from XCom — the same metrics the
        in-process :meth:`run` records (issue #44.4).
        """
        engine = self._engine

        def _task(**context: Any) -> None:
            ti = context["ti"]
            storage = Storage.for_uri(pipeline.storage_uri)
            incoming: list[StageOutput] = []
            # #69: keep each parent's records keyed so a corpus-level accumulator can stream its
            # input from them (the engine reads this map for accumulators; ``incoming`` for every
            # other Stage). The XCom store is Airflow's substrate — nothing is released.
            parent_outputs: dict[str, list[StageOutput]] = {}
            for parent in stage.depends_on:
                raw = ti.xcom_pull(task_ids=parent) or []
                parent_stage = _stage_by_name(pipeline, parent)
                parent_records = _deserialize(parent_stage, pipeline, raw)
                parent_outputs[parent] = parent_records
                incoming.extend(parent_records)
            records, metrics = _execute(
                engine, pipeline, run_id, stage, storage, incoming, parent_outputs
            )
            ti.xcom_push(key="return_value", value=[r.model_dump_json() for r in records])
            ti.xcom_push(key="metrics", value=metrics.model_dump_json())

        return _task

    def _airflow_report_callable(
        self, pipeline: Pipeline, run_id: str
    ) -> Callable[..., None]:
        """The report/commit task's Airflow function: build + persist the Quality Report (#44.4).

        Runs after every Stage (wired downstream of every sink), so all Stages have succeeded when
        it fires — a mid-DAG failure fails the run before this task, so the report/commit only ever
        reflects a complete run (no torn commit). Gathers each Stage's records and metrics from
        XCom, rebuilds the same ``outputs`` / ``stage_metrics`` the in-process :meth:`run` builds,
        and calls the shared public ``QualityReportBuilder`` seam — so the Quality Report is
        byte-identical in shape to the local Runner's, then persists it to the engine's own report
        location (the commit of the run's data artifacts).
        """
        engine = self._engine

        def _task(**context: Any) -> None:
            ti = context["ti"]
            storage = Storage.for_uri(pipeline.storage_uri)
            ordered = topological_order(list(pipeline.stages))
            outputs: dict[str, list[StageOutput]] = {}
            stage_metrics: list[StageMetrics] = []
            for stage in ordered:
                raw = ti.xcom_pull(task_ids=stage.name) or []
                outputs[stage.name] = _deserialize(stage, pipeline, raw)
                raw_metrics = ti.xcom_pull(task_ids=stage.name, key="metrics")
                if raw_metrics is not None:
                    stage_metrics.append(StageMetrics.model_validate_json(raw_metrics))
            # H-F1: build + persist through the engine's PUBLIC QualityReportBuilder + report_uri —
            # the same shared report seam ``run()`` uses, so the report is byte-identical in shape
            # to the local Runner's, with no reach past the interface.
            builder = engine.quality_report_builder
            final_records = builder.final_records(
                ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            report = builder.build(
                pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
            )
            storage.write_atomic(
                engine.report_uri(pipeline, run_id),
                report.to_json().encode("utf-8"),
            )

        return _task


def _execute(
    engine: LocalRunner,
    pipeline: Pipeline,
    run_id: str,
    stage: Stage,
    storage: Storage,
    incoming: list[StageOutput],
    parent_outputs: dict[str, list[StageOutput]],
) -> tuple[list[StageOutput], StageMetrics]:
    """Run ONE Stage through the shared executor. The single per-Stage task body (C3).

    Both entry points — the in-process driver and the real Airflow ``PythonOperator`` callable —
    go through here, so there is exactly one copy of "what a Stage task does" in this adapter
    instead of the three that had drifted apart.

    Everything that used to be re-derived by hand is now derived by the engine's
    :class:`~latence_core.stage_execution.StageExecution`: the Provider's device is routed against
    its declared profile, a GPU-only Provider on a CPU-only host is **skipped-with-flag** (an empty
    stream + ``skipped=True``) exactly as under the local Runner rather than crashing this DAG, and
    the returned :class:`~latence_core.quality.StageMetrics` carries EVERY column — the profile
    (``compute``/``model_id``/``license``), the device routing, the schema-induction tally and the
    relation fan-out — not the seven this adapter used to know how to fill in. It cannot forget a
    column because it never names one.

    XCom is a materialised substrate by construction (a task's return value is a value), so the
    Stage's lazily-streamed records are collected here — at the adapter, which is where the
    substrate's shape belongs, not in the shared executor.

    The Stage's INPUT is resolved by the engine's own
    :meth:`~latence_core.runner.LocalRunner.stage_inputs` over this task's parent map, so which
    shape each Capability receives (a gathered list, a lazy stream, one stream per carrier, an
    accumulator's per-carrier gather), how ``records_in`` is counted and which wiring rules apply
    are the Capability's declaration — not a second copy of that table living in this adapter.
    Every parent's records are present in ``parent_outputs`` (nothing is released under XCom), so
    the resolution reads the map and never the local Runner's checkpoint fallback.
    """
    ctx = StageContext(
        storage=storage,
        pipeline=pipeline,
        run_id=run_id,
        inputs=engine.stage_inputs(
            stage, parent_outputs, storage=storage, pipeline=pipeline, run_id=run_id
        ),
        runner=RUNNER_NAME,
    )
    outcome = engine.stage_execution.execute(stage, ctx)
    records = list(outcome.records)
    return records, outcome.metrics(records_in=len(incoming))


def _stage_by_name(pipeline: Pipeline, name: str) -> Stage:
    for stage in pipeline.stages:
        if stage.name == name:
            return stage
    msg = f"Unknown Stage {name!r} in pipeline {pipeline.name!r}."
    raise KeyError(msg)


def _deserialize(stage: Stage, pipeline: Pipeline, raw: list[str]) -> list[StageOutput]:
    """Re-validate a parent Stage's XCom-transported JSON back into typed records.

    Reuses the owned per-Stage output-type resolution + checkpoint decode
    (:mod:`latence_core.run_store`) so an XCom-carried record round-trips through the exact
    same typed contract the local Runner's checkpoints use — the contract survives the Airflow
    boundary, no bespoke (de)serialization and no private reach-through (Wave 2-E).
    """
    if not raw:
        return []
    text = "\n".join(raw)
    record_type = stage_output_type(stage, pipeline)
    return decode_records(text, record_type)
