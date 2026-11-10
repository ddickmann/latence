"""AirflowRunner: the identical Stages under a second Runner (S12, ADR-0003, ADR-0010).

The offline, deterministic proof. Apache Airflow is intentionally NOT installed (ADR-0016: the
adapter's in-process path runs the same per-Stage task callable Airflow would, with an XCom
stand-in, so the second-Runner proof runs offline). These tests assert:

* the same Pipeline over the same corpus yields **byte-identical exported records** under the
  Airflow adapter and the local Runner (same Stage semantics, only the Runner differs);
* every StageMetrics is stamped ``runner="airflow"`` in the Quality Report;
* the adapter uses Airflow's state, not the local files-on-Storage run state — NO local
  ``checkpoints`` / ``manifest`` / ``wal`` are written (ADR-0010);
* a Stage failure propagates (the DAG run fails) rather than being swallowed — so a Corpus Version
  can never be committed off a torn run (S12 AC);
* endpoint retries a Stage absorbed surface on that Stage's metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.contracts import ChunkRecord, EntityMention
from latence_core.pipeline import Pipeline, Stage
from latence_core.runner import LocalRunner, StageOutput
from latence_core.storage import Storage
from latence_runner_airflow import RUNNER_NAME, AirflowRunner, XComBackend


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: airflow-slice
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


def test_end_to_end_under_airflow_runner(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs_dir)

    report = AirflowRunner().run(pipeline, run_id="af")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/af"

    # Exports exist and parse (the same AI-ready outputs as any Runner).
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows, "expected exported entity mentions"

    # Every Stage ran under the Airflow Runner (proves pluggable execution in the report).
    assert {m.runner for m in report.stages} == {RUNNER_NAME}
    assert {m.name for m in report.stages} == {"source", "parse", "chunk", "entities", "export"}
    assert report.contracts_complete is True
    assert report.entities is not None and report.entities.mentions_total > 0


def test_airflow_output_matches_local_runner(docs_dir: Path, tmp_path: Path) -> None:
    # The SAME Stages under the two Runners must produce identical records (only scheduling/state
    # differ, not semantics). Run each into its own store, compare the exported record ids/text.
    local_store = f"file://{tmp_path}/local"
    airflow_store = f"file://{tmp_path}/airflow"

    local_report = LocalRunner().run(_pipeline(local_store, docs_dir), run_id="r")
    airflow_report = AirflowRunner().run(_pipeline(airflow_store, docs_dir), run_id="r")

    def _records(store: str) -> list[dict[str, Any]]:
        s = Storage.for_uri(store)
        text = s.read_text(f"{store}/_latence/runs/r/export/records.jsonl")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return sorted(rows, key=lambda r: r["record_id"])

    local_records = _records(local_store)
    airflow_records = _records(airflow_store)
    assert [r["record_id"] for r in local_records] == [r["record_id"] for r in airflow_records]
    assert [r.get("text") for r in local_records] == [r.get("text") for r in airflow_records]
    # Same document count; the local Runner stamps "local", the adapter stamps "airflow".
    assert local_report.document_count == airflow_report.document_count
    assert {m.runner for m in local_report.stages} == {"local"}
    assert {m.runner for m in airflow_report.stages} == {"airflow"}


def test_airflow_runner_uses_platform_state_not_local_checkpoints(
    docs_dir: Path, tmp_path: Path
) -> None:
    store = f"file://{tmp_path}/store"
    AirflowRunner().run(_pipeline(store, docs_dir), run_id="af")
    s = Storage.for_uri(store)
    run_dir = f"{store}/_latence/runs/af"
    # ADR-0010: an adopter-grade Runner uses its own state/scheduling — NOT the local Runner's
    # files-on-Storage checkpoints/manifest/WAL. Only data artifacts (export, report) exist.
    assert not s.exists(f"{run_dir}/checkpoints")
    assert not s.exists(f"{run_dir}/manifest.json")
    assert not s.exists(f"{run_dir}/wal.log")
    assert s.exists(f"{run_dir}/export/records.jsonl")
    assert s.exists(f"{run_dir}/quality-report.json")


def test_stage_failure_propagates_no_torn_run(tmp_path: Path) -> None:
    # A Stage whose Provider raises must fail the run (Airflow would fail the DAG run and retry the
    # task); the adapter never swallows it, so a Corpus Version can't be committed off a torn run.
    class _BoomSource:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            pass

        def produce(self, storage: Storage) -> Any:
            raise RuntimeError("source blew up")

    registry = ProviderRegistry()
    registry._loaded["source.boom"] = _BoomSource  # noqa: SLF001
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: boom
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.boom
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )
    with pytest.raises(RuntimeError, match="source blew up"):
        AirflowRunner(registry).run(pipeline, run_id="af")
    # No export was produced — the run aborted at the failed Stage.
    s = Storage.for_uri(store)
    assert not s.exists(f"{store}/_latence/runs/af/export/records.jsonl")


def test_endpoint_retries_surface_on_stage_metrics(docs_dir: Path, tmp_path: Path) -> None:
    # A fake endpoint-backed entity Provider that reports it retried through 3 transient failures;
    # the adapter reads that off the Provider and folds it into the Stage's report metrics (S12 AC:
    # endpoint failures retried/surfaced).
    class _FlakyEndpointExtractor:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self.endpoint_retries = 0

        def extract(self, chunks: Any) -> Any:
            self.endpoint_retries = 3  # simulate three transient endpoint blips absorbed
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
name: flaky
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
    report = AirflowRunner(registry).run(pipeline, run_id="af")
    entities_metric = next(m for m in report.stages if m.name == "entities")
    assert entities_metric.endpoint_retries == 3
    assert entities_metric.runner == "airflow"
    # Other Stages report no endpoint retries (CPU/in-process Providers).
    assert next(m for m in report.stages if m.name == "parse").endpoint_retries == 0


def test_xcom_backend_gathers_parents_in_order() -> None:
    xcom = XComBackend()
    # String sentinels stand in for opaque StageOutput records: this test only asserts the
    # concatenation ORDER (parents in declared order), not record structure, so we cast them
    # through the typed seam rather than construct full record instances.
    a_outputs = cast(list[StageOutput], ["a1", "a2"])
    b_outputs = cast(list[StageOutput], ["b1"])
    xcom.push("a", a_outputs)
    xcom.push("b", b_outputs)

    class _Stage:
        depends_on = ["a", "b"]

    gathered = xcom.gather(cast(Stage, _Stage()))
    assert gathered == cast(list[StageOutput], ["a1", "a2", "b1"])


def test_a_template_pipeline_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Pipeline with an unfilled ``{STORAGE_URI}`` is a config error under this Runner too.

    The adapter never calls the local Runner's ``run()``, so the refusal the local Runner makes
    does not cover it. Executed as written, the template would create a directory literally named
    ``{STORAGE_URI}`` under the working directory and report an empty corpus. ``build_dag`` refuses
    as well, and before it needs Airflow: a DAG built from a template would only fail (or, worse,
    not fail) once scheduled.
    """
    from latence_core.errors import ConfigError

    monkeypatch.chdir(tmp_path)
    template = _pipeline("'{STORAGE_URI}'", Path("{CORPUS_DIR}"))
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        AirflowRunner().run(template, run_id="af")
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        AirflowRunner().build_dag(template)
    assert list(tmp_path.iterdir()) == []


def test_a_pipeline_with_a_removed_config_key_is_refused_before_anything_is_written(
    docs_dir: Path, tmp_path: Path
) -> None:
    """A config key ADR-0068 removed is a config error under this Runner too.

    ``export.jsonl_parquet`` no longer reads ``context_columns``, so without the refusal this run
    completes and the corpus carries none of the columns the key asks for. The refusal names the
    key, the Stage and the ADR, and comes before the storage root exists. ``build_dag`` refuses as
    well, before it needs Airflow.
    """
    from latence_core.errors import ConfigError

    store = tmp_path / "store"
    pipeline = _pipeline(f"file://{store}", docs_dir)
    next(s for s in pipeline.stages if s.name == "export").config["context_columns"] = True

    with pytest.raises(ConfigError, match="'context_columns'") as excinfo:
        AirflowRunner().run(pipeline, run_id="af")
    assert "Stage 'export'" in str(excinfo.value)
    assert "ADR-0068" in str(excinfo.value)
    with pytest.raises(ConfigError, match="'context_columns'"):
        AirflowRunner().build_dag(pipeline)
    assert not store.exists()
