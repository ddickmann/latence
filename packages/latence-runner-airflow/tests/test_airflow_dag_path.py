"""The real-adopter DAG path: report/commit task + DAG structure (issue #44.4).

The in-process ``run()`` proof (``test_airflow_runner.py``) drives the per-Stage task *callable*
through a topological driver. This module exercises the **actual adopter DAG path** — the task
callables :meth:`build_dag` wires into Airflow operators, driven through a faithful fake of
Airflow's XCom (per-task-id, per-key push/pull) — WITHOUT needing Airflow installed, so it runs
offline and deterministically in CI (ADR-0016: apache-airflow is intentionally out of the CI env).

It covers what the DAG path previously lacked (issue #44.4): the report/commit task that builds and
persists the Quality Report downstream of every Stage, and the explicit commit step. A separate
lane, gated on ``apache-airflow`` actually being importable, asserts the real ``DAG`` structure:
one operator per Stage plus the ``report`` task wired downstream of the sinks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner, topological_order
from latence_core.storage import Storage
from latence_runner_airflow import RUNNER_NAME, AirflowRunner

_HAS_AIRFLOW = importlib.util.find_spec("airflow") is not None


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: dag-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:
      labels:
        WORD:
          terms: [framework, folder, records, Notes]
          confidence: 0.9
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


class _FakeTI:
    """A faithful stand-in for Airflow's task-instance XCom.

    Airflow keys every pushed value by ``(task_id, key)`` and ``xcom_pull(task_ids=..., key=...)``
    reads it back; the default push key is ``"return_value"`` and a bare ``xcom_pull(task_ids=t)``
    reads that default. This models exactly that, so the same task callables the DAG runs work here.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], Any] = {}
        self.current_task_id: str = ""

    def xcom_push(self, key: str, value: Any) -> None:
        self._store[(self.current_task_id, key)] = value

    def xcom_pull(self, task_ids: str, key: str = "return_value") -> Any:
        return self._store.get((task_ids, key))


def _drive_dag_callables(runner: AirflowRunner, pipeline: Pipeline, run_id: str) -> None:
    """Run the DAG's real task callables (Stage tasks then the report/commit task) via a fake TI.

    This is the adopter DAG path minus Airflow's scheduler: each Stage's callable, in topological
    order (what the ``depends_on`` wiring guarantees), then the report/commit callable wired
    downstream of the sinks — exactly the operators :meth:`build_dag` constructs.
    """
    ti = _FakeTI()
    ordered = topological_order(list(pipeline.stages))
    for stage in ordered:
        ti.current_task_id = stage.name
        callable_ = runner._airflow_stage_callable(pipeline, run_id, stage)  # noqa: SLF001
        callable_(ti=ti)
    ti.current_task_id = "report"
    report_callable = runner._airflow_report_callable(pipeline, run_id)  # noqa: SLF001
    report_callable(ti=ti)


def test_dag_path_report_commit_task_persists_quality_report(
    docs_dir: Path, tmp_path: Path
) -> None:
    """Regression (issue #44.4): the DAG path's report/commit task emits the Quality Report.

    Before the fix ``build_dag`` had one task per Stage and NO report/commit task, so the real
    adopter DAG produced exports but never a Quality Report and never an explicit commit. Driving
    the DAG's own task callables must now persist a Quality Report stamped ``runner="airflow"`` on
    every Stage — the same artifact the in-process ``run()`` writes.
    """
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs_dir)

    _drive_dag_callables(AirflowRunner(), pipeline, run_id="dag")

    s = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/dag/quality-report.json"
    assert s.exists(report_uri), "the report/commit task must persist a Quality Report"
    report = json.loads(s.read_text(report_uri))
    stage_runners = {stage["runner"] for stage in report["stages"]}
    assert stage_runners == {RUNNER_NAME}
    stage_names = {stage["name"] for stage in report["stages"]}
    assert stage_names == {"source", "parse", "chunk", "entities", "export"}
    assert report["contracts_complete"] is True
    # Exports were committed alongside the report.
    assert s.exists(f"{store}/_latence/runs/dag/export/records.jsonl")


def test_dag_path_report_matches_in_process_run(docs_dir: Path, tmp_path: Path) -> None:
    """The DAG path's committed report is equivalent to the in-process ``run()`` report.

    Same Stages, same corpus, only the driver differs (real task callables via XCom vs the
    topological in-process driver) — so the committed Quality Report must carry the same per-Stage
    metrics and document count. Proves the report/commit task is not a bespoke second report path.
    """
    dag_store = f"file://{tmp_path}/dag"
    run_store = f"file://{tmp_path}/run"

    _drive_dag_callables(AirflowRunner(), _pipeline(dag_store, docs_dir), run_id="r")
    in_process_report = AirflowRunner().run(_pipeline(run_store, docs_dir), run_id="r")

    s = Storage.for_uri(dag_store)
    dag_report = json.loads(s.read_text(f"{dag_store}/_latence/runs/r/quality-report.json"))

    assert dag_report["document_count"] == in_process_report.document_count
    dag_out = {stage["name"]: stage["records_out"] for stage in dag_report["stages"]}
    run_out = {m.name: m.records_out for m in in_process_report.stages}
    assert dag_out == run_out
    assert {stage["runner"] for stage in dag_report["stages"]} == {RUNNER_NAME}


def test_dag_path_report_survives_endpoint_retries(docs_dir: Path, tmp_path: Path) -> None:
    """The DAG report/commit task carries the endpoint-retry tally through XCom (issue #44.4).

    A Stage's Provider reports retries; the Stage task pushes its metrics (incl. the retry count) to
    XCom and the report/commit task reads them back — so a flaky endpoint is visible in the DAG
    path's Quality Report, not just the in-process one.
    """
    from latence_core.capability import ProviderRegistry
    from latence_core.contracts import ChunkRecord, EntityMention

    class _FlakyEndpointExtractor:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self.endpoint_retries = 0

        def extract(self, chunks: Any) -> Any:
            self.endpoint_retries = 2
            for chunk in chunks:
                if not isinstance(chunk, ChunkRecord):
                    continue
                prov = chunk.provenance.model_copy(
                    update={
                        "char_start": chunk.provenance.char_start or 0,
                        "char_end": (chunk.provenance.char_start or 0) + 1,
                    }
                )
                yield EntityMention(
                    record_id=f"{chunk.record_id}#entity-0",
                    document_record_id=chunk.document_record_id,
                    chunk_record_id=chunk.record_id,
                    index=0,
                    label="WORD",
                    text=chunk.content[:1] or "x",
                    confidence=0.9,
                    provenance=prov,
                    classification=chunk.classification,
                )
                return

    registry = ProviderRegistry()
    registry._loaded["entity.flaky_endpoint"] = _FlakyEndpointExtractor  # noqa: SLF001
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: flaky-dag
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.flaky_endpoint
    depends_on: [chunk]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )
    _drive_dag_callables(AirflowRunner(registry), pipeline, run_id="dag")
    s = Storage.for_uri(store)
    report = json.loads(s.read_text(f"{store}/_latence/runs/dag/quality-report.json"))
    entities = next(stage for stage in report["stages"] if stage["name"] == "entities")
    assert entities["endpoint_retries"] == 2
    parse = next(stage for stage in report["stages"] if stage["name"] == "parse")
    assert parse["endpoint_retries"] == 0


@pytest.mark.skipif(not _HAS_AIRFLOW, reason="apache-airflow not installed (out of CI per ADR-0016)")
def test_real_airflow_dag_has_report_task_downstream_of_sinks(
    docs_dir: Path, tmp_path: Path
) -> None:
    """Real-airflow lane (issue #44.4): the built DAG has a report task downstream of every sink.

    Skipped when apache-airflow is absent (the CI default). When present, it asserts the actual
    ``DAG`` structure: one operator per Stage PLUS a ``report`` task wired downstream of every sink
    Stage, so Airflow schedules the commit only after all Stages succeed.
    """
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs_dir)
    dag = AirflowRunner().build_dag(pipeline, dag_id="latence-test")

    task_ids = set(dag.task_dict.keys())
    assert task_ids == {"source", "parse", "chunk", "entities", "export", "report"}

    report_task = dag.get_task("report")
    # The sole sink here is ``export``; the report task must be immediately downstream of it.
    assert "export" in {t.task_id for t in report_task.upstream_list}
    # And the report task has no downstream — it is the terminal commit.
    assert report_task.downstream_list == []


@pytest.mark.skipif(not _HAS_AIRFLOW, reason="apache-airflow not installed (out of CI per ADR-0016)")
def test_real_airflow_dag_stage_and_report_callables_run(docs_dir: Path, tmp_path: Path) -> None:
    """Real-airflow lane: the DAG's operator callables execute end-to-end and commit a report.

    Runs each operator's ``python_callable`` in topological order through a real Airflow-shaped TI,
    proving the actual operators the adopter drops into Airflow produce exports + a committed
    Quality Report (not just that the DAG is shaped right).
    """
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs_dir)
    dag = AirflowRunner().build_dag(pipeline, dag_id="latence-test", run_id="dag")

    ti = _FakeTI()
    for stage in topological_order(list(pipeline.stages)):
        ti.current_task_id = stage.name
        dag.get_task(stage.name).python_callable(ti=ti)
    ti.current_task_id = "report"
    dag.get_task("report").python_callable(ti=ti)

    s = Storage.for_uri(store)
    assert s.exists(f"{store}/_latence/runs/dag/quality-report.json")
    assert s.exists(f"{store}/_latence/runs/dag/export/records.jsonl")


def test_local_runner_import_is_available() -> None:
    """Guard: the shared engine seam the report task reuses is the LocalRunner (no fork)."""
    assert issubclass(type(AirflowRunner()._engine), LocalRunner)  # noqa: SLF001
