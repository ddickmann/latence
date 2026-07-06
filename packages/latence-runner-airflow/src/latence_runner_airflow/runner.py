"""Airflow Runner adapter — the same Stages under a second Runner (S12, ADR-0003, ADR-0010).

The framework core is a library of typed Stages; execution is delegated to interchangeable Runners
(ADR-0003). This is the proof of that seam's generality: an adapter that runs the **identical**
Pipeline Stages inside **Apache Airflow**, so an adopter keeps their own orchestrator instead of
adopting ours. The Stage *semantics* are not re-implemented — the adapter reuses
:class:`latence_core.runner.LocalRunner`'s :meth:`~latence_core.runner.LocalRunner._execute_stage`
and :meth:`~latence_core.runner.LocalRunner._build_report` seams verbatim (no fork). What differs is
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
**task-level retries** re-run a whole Stage task on an unhandled failure. Because a Corpus Version
is only committed *after* the DAG run finishes successfully (the commit is downstream of every Stage
task), a mid-DAG endpoint failure — retried or ultimately failed — leaves the last committed Corpus
Version intact (S12 AC): a failed Airflow task fails the DAG run before any commit, never torn.

``airflow`` is a heavy optional dependency isolated in THIS adapter package (ADR-0016), imported
lazily only in :meth:`AirflowRunner.build_dag`; the in-process execution path
(:meth:`AirflowRunner.run`) needs no Airflow install, so the same Stage-under-a-second-Runner proof
runs offline and deterministically in CI.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from latence_core.pipeline import Pipeline
from latence_core.quality import StageMetrics
from latence_core.runner import LocalRunner, StageOutput, topological_order
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
        storage = Storage.for_uri(pipeline.storage_uri)
        # The Export Stage writes into this run's export dir; reuse the engine's own layout so the
        # data artifacts land where the report builder expects them.
        storage.makedirs(self._engine._run_dir(pipeline, run_id))  # noqa: SLF001

        ordered = topological_order(list(pipeline.stages))
        xcom = XComBackend()
        stage_metrics: list[StageMetrics] = []
        for stage in ordered:
            stage_metrics.append(self._run_stage_task(pipeline, run_id, stage, storage, xcom))

        outputs = xcom.as_outputs()
        final_records = self._engine._final_records(ordered, outputs)  # noqa: SLF001
        report = self._engine._build_report(  # noqa: SLF001
            pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
        )
        # The report is a data artifact of the run; persist it alongside the exports (the engine's
        # own report location), so an adopter reads it the same way regardless of Runner.
        storage.write_atomic(
            self._engine._report_uri(pipeline, run_id),  # noqa: SLF001
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
        """The per-Stage task body — identical whether driven in-process or by an Airflow operator.

        Gathers the Stage's inputs from its parents' XCom, runs the Stage through the shared engine
        (identical semantics + endpoint-retry accounting), pushes the output to XCom, and returns
        the Stage's metrics stamped with this Runner's name. A raised exception propagates so the
        surrounding scheduler (Airflow's task retry, or the in-process driver) handles it — the
        adapter never swallows a Stage failure, so a corrupt commit is impossible.
        """
        incoming = xcom.gather(stage)
        started = time.perf_counter()
        own_output = self._engine._execute_stage(  # noqa: SLF001
            stage, storage, pipeline, run_id, incoming
        )
        duration = time.perf_counter() - started
        xcom.push(stage.name, own_output)
        return StageMetrics(
            name=stage.name,
            capability=stage.capability.value,
            provider=stage.provider,
            records_in=len(incoming),
            records_out=len(own_output),
            duration_seconds=round(duration, 6),
            runner=RUNNER_NAME,
            endpoint_retries=self._engine._drain_retry_count(),  # noqa: SLF001
        )

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
        """Build a real Airflow ``DAG`` of one ``PythonOperator`` per Stage (adopter path).

        Wires the operators by the Pipeline's ``depends_on`` edges so Airflow schedules the Stages
        in the right order and owns their state/retries (ADR-0010). Each operator's callable is
        :meth:`_airflow_stage_callable`, which runs the Stage through the same shared engine and
        pushes/pulls via Airflow's real XCom — so the Stage semantics match the local Runner.
        ``airflow`` is imported here (lazily) so the rest of the adapter — and the in-process
        proof — needs no Airflow install (ADR-0016).
        """
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
        return dag

    def _airflow_stage_callable(
        self, pipeline: Pipeline, run_id: str, stage: Stage
    ) -> Callable[..., None]:
        """A Stage's Airflow task function: pull parents from XCom, run the Stage, push the output.

        Uses Airflow's real XCom (via the task-instance ``ti`` Airflow injects) as the inter-task
        carrier — so Airflow owns the run state, not local checkpoint files (ADR-0010). Records are
        serialized to their contract JSON for XCom transport and re-validated on pull, so the
        typed contract survives the Airflow boundary intact.
        """
        engine = self._engine

        def _task(**context: Any) -> None:
            ti = context["ti"]
            storage = Storage.for_uri(pipeline.storage_uri)
            incoming: list[StageOutput] = []
            for parent in stage.depends_on:
                raw = ti.xcom_pull(task_ids=parent) or []
                parent_stage = _stage_by_name(pipeline, parent)
                incoming.extend(_deserialize(engine, parent_stage, pipeline, raw))
            own_output = engine._execute_stage(  # noqa: SLF001
                stage, storage, pipeline, run_id, incoming
            )
            ti.xcom_push(key="return_value", value=[r.model_dump_json() for r in own_output])

        return _task


def _stage_by_name(pipeline: Pipeline, name: str) -> Stage:
    for stage in pipeline.stages:
        if stage.name == name:
            return stage
    msg = f"Unknown Stage {name!r} in pipeline {pipeline.name!r}."
    raise KeyError(msg)


def _deserialize(
    engine: LocalRunner, stage: Stage, pipeline: Pipeline, raw: list[str]
) -> list[StageOutput]:
    """Re-validate a parent Stage's XCom-transported JSON back into typed records.

    Reuses the engine's own per-Stage output-type resolution + checkpoint reader so an XCom-carried
    record round-trips through the exact same typed contract the local Runner's checkpoints use —
    the contract survives the Airflow boundary, no bespoke (de)serialization.
    """
    if not raw:
        return []
    text = "\n".join(raw)
    record_type = engine._output_type(stage, pipeline)  # noqa: SLF001
    return engine._read_checkpoint_text(text, record_type)  # noqa: SLF001
