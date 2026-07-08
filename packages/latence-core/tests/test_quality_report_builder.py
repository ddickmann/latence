"""H-F1: the extracted QualityReportBuilder is a real, relocatable unit — byte-identical report.

The Quality Report builder used to live inside :class:`~latence_core.runner.LocalRunner` as
~830 lines of private methods that a second Runner could only reuse by reaching past the interface
(``# noqa: SLF001``). H-F1 lifted it into :mod:`latence_core.quality_report_builder`. These tests
prove the extraction met its guarantee:

* **Byte-identical**: a report rebuilt through a *freshly-constructed* public
  :class:`~latence_core.quality_report_builder.QualityReportBuilder` — against the engine's public
  reader seam (``stage_records`` / ``read_findings``) — is byte-for-byte identical to the one the
  local Runner emitted. No behaviour changed; only the wiring moved.
* **Relocatable unit**: the builder accepts its dependencies (a reader + a fan-out tally) and does
  not reach back into the Runner — it composes with a minimal fake reader that is not a
  ``LocalRunner`` at all.
* **The Airflow report path no longer reaches past the interface**: the report-building
  ``_build_report`` / ``_final_records`` / ``_report_uri`` ``SLF001`` reaches are gone from the
  adapter (grep-verified so the count cannot silently regress).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from latence_core.contracts import ScreeningFinding
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.quality_report_builder import QualityReportBuilder, StageRecordReader
from latence_core.runner import LocalRunner, StageOutput, topological_order
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    """A pipeline broad enough to populate many report sections (parse/chunk/entity/profiling).

    Source→Parse→Chunk→Entity→Redaction→Profiling→Export. The Redaction Stage carries a ``salt``
    so the co-occurrence fingerprinting path (relocated with the builder) is exercised with a
    non-default salt; the whole chain is a single spine so the DAG has one sink (Export).
    """
    return Pipeline.from_yaml_text(
        f"""
name: hf1-report
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
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
        ORG:
          terms: [Acme, Globex]
        PERSON:
          terms: [Alice, Bob]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [parse]
    config:
      salt: hf1-secret-salt
  - name: profile
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, entities]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [profile]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works at Acme. Bob also works at Acme. The team is productive.\n",
        encoding="utf-8",
    )
    (docs / "note.txt").write_text(
        "Acme announced a partnership with Globex. Alice leads the effort.\n",
        encoding="utf-8",
    )
    return docs


def test_report_is_byte_identical_when_rebuilt_through_a_fresh_public_builder(
    tmp_path: Path,
) -> None:
    """The persisted report == a report rebuilt via a freshly-constructed public builder.

    This is the H-F1 no-behaviour-change proof: run the pipeline, read the report the local Runner
    persisted, then rebuild the report through a *new* ``QualityReportBuilder`` constructed against
    the engine's public reader seam (``stage_records`` / ``read_findings``) + the engine's fan-out
    tally — and assert the serialized JSON is byte-for-byte identical. Nothing about the report's
    content changed; only where the building code lives moved.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)

    engine = LocalRunner()
    persisted_report = engine.run(pipeline, run_id="hf1")
    s = Storage.for_uri(store)
    on_disk = s.read_text(f"{store}/_latence/runs/hf1/quality-report.json")

    # Rebuild through a FRESH public builder — not the engine's own composed instance — using only
    # the public reader seam + the fan-out tally the run captured. The outputs dict is empty: every
    # Stage's records are read back from its on-disk checkpoint via the reader (#69 path).
    rebuilt_builder = QualityReportBuilder(engine, fanout_by_stage=engine._fanout_by_stage)  # noqa: SLF001
    ordered = topological_order(list(pipeline.stages))
    empty_outputs: dict[str, list[StageOutput]] = {}
    final_records = rebuilt_builder.final_records(
        ordered, empty_outputs, storage=s, pipeline=pipeline, run_id="hf1"
    )
    rebuilt = rebuilt_builder.build(
        pipeline,
        "hf1",
        s,
        ordered,
        list(persisted_report.stages),
        empty_outputs,
        final_records,
    )

    # Byte-identical JSON (the report is a shareable artifact; its bytes are the contract).
    assert rebuilt.to_json() == on_disk
    assert rebuilt.to_json() == persisted_report.to_json()
    # Sanity: the broad pipeline actually populated the sections we care about (not an all-None
    # report that would pass vacuously).
    assert persisted_report.parse is not None
    assert persisted_report.chunk is not None
    assert persisted_report.entities is not None
    assert persisted_report.redaction is not None
    assert persisted_report.profiling is not None


class _FakeReader:
    """A minimal :class:`StageRecordReader` that is NOT a ``LocalRunner``.

    Proves the builder is a relocatable unit: it depends only on the reader seam (records +
    findings), never on the concrete engine. This one serves an in-memory ``outputs`` map and no
    screening findings.
    """

    def __init__(self, outputs: dict[str, list[StageOutput]]) -> None:
        self._outputs = outputs

    def stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        yield from self._outputs.get(stage_name, [])

    def read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        return []


def test_builder_composes_with_a_minimal_fake_reader_no_runner_needed(
    tmp_path: Path,
) -> None:
    """The builder builds a report with a fake reader — it does not reach back into the Runner.

    A Source→Parse→Export pipeline whose Stage outputs come from a hand-built ``_FakeReader`` (not a
    ``LocalRunner``). The builder produces a valid ``QualityReport`` purely through the injected
    seam — the deletion test for the report concern: it is a real unit, not glued to the executor.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    ordered = topological_order(list(pipeline.stages))

    # Run once to obtain real Stage outputs, then hand them to the builder via a fake reader with an
    # EMPTY storage-backed path (the fake serves everything from memory).
    engine = LocalRunner()
    report_ref = engine.run(pipeline, run_id="ref")
    s = Storage.for_uri(store)
    outputs: dict[str, list[StageOutput]] = {
        stage.name: list(
            engine.stage_records(
                stage.name, {}, storage=s, pipeline=pipeline, run_id="ref"
            )
        )
        for stage in ordered
    }

    fake: StageRecordReader = _FakeReader(outputs)
    builder = QualityReportBuilder(fake)  # no fanout tally → every Stage contributes (0,0,0,0)
    final_records = builder.final_records(
        ordered, outputs, storage=s, pipeline=pipeline, run_id="ref"
    )
    report = builder.build(
        pipeline,
        "ref",
        s,
        ordered,
        list(report_ref.stages),
        outputs,
        final_records,
    )
    assert isinstance(report, QualityReport)
    # Same document count + completeness as the reference run — the builder reproduced the report
    # from the injected records alone.
    assert report.document_count == report_ref.document_count
    assert report.completeness.passed == report_ref.completeness.passed
    assert report.entities is not None
    assert report_ref.entities is not None
    assert report.entities.mentions_total == report_ref.entities.mentions_total


def test_airflow_report_path_has_no_slf001_reach_for_report_building() -> None:
    """The Airflow adapter builds its report through the public builder — no SLF001 report reaches.

    The audit's THEME F fix: the report-building private reaches
    (``_build_report`` / ``_final_records`` / ``_report_uri``) are gone from the Airflow adapter.
    Grep-verify the count is zero so it cannot silently regress; assert the adapter instead names
    the public ``quality_report_builder`` / ``report_uri`` interface.
    """
    adapter = (
        Path(__file__).resolve().parents[2]
        / "latence-runner-airflow"
        / "src"
        / "latence_runner_airflow"
        / "runner.py"
    )
    lines = adapter.read_text(encoding="utf-8").splitlines()

    # 1. No CODE line calls a report-building private method on the engine (a real reach-through),
    #    regardless of whether it carries a noqa. Match the actual call forms, not doc mentions.
    report_reaches = (
        "._build_report(",
        "._final_records(",
        "._report_uri(",
        "._run_dir(",
    )
    for i, line in enumerate(lines, start=1):
        code = line.split("#", 1)[0]  # ignore trailing comments/docstring-free code
        for reach in report_reaches:
            assert reach not in code, (
                f"Airflow adapter line {i} still reaches past the interface: {reach!r} in {line!r}"
            )

    # 2. No SLF001-suppressed reach remains on a report-building line (the reaches the issue named
    #    to kill; the execution-side reaches — _execute_stage etc. — may keep theirs).
    for i, line in enumerate(lines, start=1):
        if "SLF001" in line:
            assert not any(
                name in line
                for name in ("build_report", "final_records", "report_uri", "run_dir")
            ), f"Airflow adapter line {i} still has an SLF001 report reach: {line!r}"

    # 3. It uses the public report seam instead.
    text = "\n".join(lines)
    assert "quality_report_builder" in text
    assert ".report_uri(" in text
    assert ".run_dir(" in text
