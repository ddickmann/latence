"""Runner observability regression tests (#73): typed errors surfaced in spans + metrics.

Pins the wiring a revert of the Runner's #73 changes would break:

1. A Stage that raises a typed :class:`~latence_core.errors.ProviderError` emits a **failed
   ``stage_end`` span** stamped with the ``"provider"`` category (not ``"unknown"``), and the
   exception propagates (the Runner never swallows a Stage failure — no torn commit).
2. A mis-wired Pipeline (a carrier mismatch) raises a :class:`~latence_core.errors.ContractError`
   — a ``TypeError`` — so the pre-taxonomy ``except TypeError`` guards keep catching it, and the
   failed span is stamped ``"contract"``.
3. The failed span carries the error **message + category only, never the record content** that
   triggered it.
4. The failure is surfaced in the *persisted Quality Report* too, not only the span: the failed
   Stage's :class:`~latence_core.quality.StageMetrics` lands with ``error_count=1`` and the stable
   ``error_category`` tag — the report half of #73's "surfaced in the report + logs".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import latence_core.runner as runner_mod
import pytest
from latence_core.capability import ProviderRegistry
from latence_core.contracts import DocumentRecord, ParserInput
from latence_core.errors import ContractError, ProviderError
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage
from latence_core.tracing import CollectingSpanSink

_PII_SENTINEL = "patient-jane-doe-DOB-1970-01-01-unique-sentinel"


class _ExplodingParser:
    """A Parser Provider that raises a typed ProviderError mid-parse (a runtime provider fault)."""

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        # Touch the inputs so the failure is genuinely inside the Stage's work, then raise a
        # typed error whose message deliberately does NOT include any record content.
        _ = list(inputs)
        raise ProviderError("upstream model endpoint returned 503", retryable=True)
        yield  # pragma: no cover - unreachable, marks this a generator


class _InjectedRegistry(ProviderRegistry):
    """Resolves the exploding parser by name, else delegates to real entry-point discovery."""

    def load(self, name: str) -> type:
        if name == "parser.exploding":
            return _ExplodingParser
        return super().load(name)


def _pipeline(store: str, docs_dir: Path, *, parser: str = "parser.plaintext") -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: obs-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: confidential
      extensions: [txt]
  - name: parse
    capability: parse
    provider: {parser}
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


def _miswired_pipeline(store: str, docs_dir: Path) -> Pipeline:
    # Source feeds Export directly (no Parse between) — a raw ParserInput reaches Export, which
    # is a carrier-contract violation the Runner must reject with a ContractError.
    return Pipeline.from_yaml_text(
        f"""
name: miswired
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      extensions: [txt]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [source]
"""
    )


def test_provider_failure_emits_failed_span_with_provider_category(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(f"{_PII_SENTINEL}\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    runner = LocalRunner(registry=_InjectedRegistry(), span_sink=sink)

    # The typed ProviderError propagates — the Runner never swallows a Stage failure.
    with pytest.raises(ProviderError):
        runner.run(_pipeline(store, docs, parser="parser.exploding"), run_id="fail-run")

    # A failed stage_end span for parse, stamped with the typed "provider" category.
    failed = [s for s in sink.spans if s.event == "stage_end" and s.error_category is not None]
    assert len(failed) == 1
    span = failed[0]
    assert span.stage == "parse"
    assert span.error_category == "provider"
    assert span.error_message is not None and "503" in span.error_message
    # The parse checkpoint was never finalized (the Stage failed before commit) → no torn write.
    s = Storage.for_uri(store)
    assert not s.exists(f"{store}/_latence/runs/fail-run/checkpoints/parse.jsonl")


def test_failed_span_carries_no_record_content_adversarial(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(f"{_PII_SENTINEL}\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    runner = LocalRunner(registry=_InjectedRegistry(), span_sink=sink)
    with pytest.raises(ProviderError):
        runner.run(_pipeline(store, docs, parser="parser.exploding"), run_id="fail-run-2")
    for span in sink.spans:
        assert _PII_SENTINEL not in repr(span.as_dict())


def test_miswiring_raises_contract_error_and_stamps_contract_category(
    docs_dir: Path, tmp_path: Path
) -> None:
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    runner = LocalRunner(span_sink=sink)

    # A carrier mismatch is a ContractError — and, being a TypeError subclass, still catchable by
    # the pre-taxonomy ``except TypeError`` guards.
    with pytest.raises(ContractError):
        runner.run(_miswired_pipeline(store, docs_dir), run_id="miswired-run")
    with pytest.raises(TypeError):
        runner.run(_miswired_pipeline(store, docs_dir), run_id="miswired-run-b")

    failed = [s for s in sink.spans if s.event == "stage_end" and s.error_category is not None]
    assert failed and failed[0].stage == "export"
    assert failed[0].error_category == "contract"


def test_successful_stage_metrics_have_no_error(docs_dir: Path, tmp_path: Path) -> None:
    # A clean run records error_count=0 / error_category=None on every persisted StageMetrics,
    # so the new fields default correctly and a healthy report carries no spurious error marker.
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs_dir), run_id="ok-run")
    assert report.stages
    for m in report.stages:
        assert m.error_count == 0
        assert m.error_category is None
    # The report's inline render carries NO ERROR[...] marker on a clean run.
    assert "ERROR[" not in report.render()


def _load_persisted_report(store: str, run_id: str) -> QualityReport:
    s = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/quality-report.json"
    assert s.exists(uri), "a Quality Report must be persisted even on a failed run"
    return QualityReport.model_validate(json.loads(s.read_text(uri)))


def test_failed_stage_lands_error_count_in_persisted_report(tmp_path: Path) -> None:
    # #73 report half: a Stage failure persists a Quality Report whose failed Stage carries
    # error_count=1 and the stable "provider" category — the taxonomy is surfaced in the REPORT,
    # not only the transient span. This is the assertion a revert of the report-persist path fails:
    # without it no report is written on failure and error_count is never non-zero anywhere.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(f"{_PII_SENTINEL}\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_InjectedRegistry())

    with pytest.raises(ProviderError):
        runner.run(_pipeline(store, docs, parser="parser.exploding"), run_id="fail-report")

    report = _load_persisted_report(store, "fail-report")
    failed = [m for m in report.stages if m.error_count]
    assert len(failed) == 1
    assert failed[0].name == "parse"
    assert failed[0].error_count == 1
    assert failed[0].error_category == "provider"
    assert failed[0].records_out == 0
    # A healthy Stage in the same run still records error_count=0 (only the failed Stage is marked).
    ok = [m for m in report.stages if not m.error_count]
    assert ok and all(m.error_category is None for m in ok)
    # The failed category is legible in the human-readable render too.
    assert "ERROR[provider]" in report.render()


def test_miswiring_lands_contract_error_count_in_persisted_report(
    docs_dir: Path, tmp_path: Path
) -> None:
    # A carrier-mismatch ContractError also surfaces in the persisted report as a "contract"
    # error_count row on the offending Stage — the taxonomy tag, not a refactorable type name.
    store = f"file://{tmp_path}/store"
    with pytest.raises(ContractError):
        LocalRunner().run(_miswired_pipeline(store, docs_dir), run_id="miswired-report")

    report = _load_persisted_report(store, "miswired-report")
    failed = [m for m in report.stages if m.error_count]
    assert failed and failed[0].error_category == "contract"


def test_persisted_fail_report_carries_no_record_content_adversarial(tmp_path: Path) -> None:
    # The failed-run report is counts-only, exactly like a span: the PII sentinel that flowed
    # through the run appears in NO field of the persisted report (error_message is never stored).
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(f"{_PII_SENTINEL}\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    with pytest.raises(ProviderError):
        LocalRunner(registry=_InjectedRegistry()).run(
            _pipeline(store, docs, parser="parser.exploding"), run_id="fail-report-pii"
        )
    s = Storage.for_uri(store)
    raw = s.read_text(f"{store}/_latence/runs/fail-report-pii/quality-report.json")
    assert _PII_SENTINEL not in raw


def test_failed_report_does_not_block_resume(tmp_path: Path) -> None:
    # The FAIL report is not a resume gate: a run that fails at parse persists a FAIL report, then
    # a resume with a WORKING parser re-runs the failed Stage and overwrites the report with a
    # clean one (error_count back to 0). No-torn-write: the failed parse checkpoint never existed,
    # so the resume recomputes it exactly.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello world\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"

    with pytest.raises(ProviderError):
        LocalRunner(registry=_InjectedRegistry()).run(
            _pipeline(store, docs, parser="parser.exploding"), run_id="resume-run"
        )
    fail_report = _load_persisted_report(store, "resume-run")
    assert any(m.error_count for m in fail_report.stages)

    # Resume with a real parser (same run_id): source may resume from its checkpoint, parse re-runs.
    ok_report = LocalRunner().run(_pipeline(store, docs), run_id="resume-run")
    assert all(m.error_count == 0 for m in ok_report.stages)
    persisted = _load_persisted_report(store, "resume-run")
    assert all(m.error_count == 0 for m in persisted.stages)


def test_stage_batch_spans_emitted_across_batch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #73 batch seam: with a tiny checkpoint batch a multi-doc corpus crosses a batch boundary,
    # so the Runner emits a ``stage_batch`` span per durably-written batch (counts only) — proving
    # the batch span path actually executes with a tracer attached, not just the begin/end spans.
    # A revert of the ``tracer.stage_batch`` wiring in _write_checkpoint drops these spans.
    monkeypatch.setattr(runner_mod, "_CHECKPOINT_BATCH_RECORDS", 1)
    docs = tmp_path / "docs"
    docs.mkdir()
    # Three .txt docs → the parse Stage fans out to three DocumentRecords, so at batch size 1 the
    # parse checkpoint spans three batches (the pipeline's source restricts to `txt`).
    for i in range(3):
        (docs / f"d{i}.txt").write_text(f"document number {i} body text\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    LocalRunner(span_sink=sink).run(_pipeline(store, docs), run_id="batch-run")

    batch_spans = [s for s in sink.spans if s.event == "stage_batch"]
    assert batch_spans, "expected per-batch stage_batch spans with a tracer attached"
    # Every batch span carries the run's correlation id and counts-only fields (batch index + a
    # cumulative record count), never record content.
    for s in batch_spans:
        assert s.correlation_id == "batch-run"
        assert set(s.fields) == {"batch", "records"}
    # The parse Stage emitted one document per doc in the corpus; with batch size 1 it must span
    # three batches, and the cumulative record count is monotonic non-decreasing per Stage.
    parse_spans = [s for s in batch_spans if s.stage == "parse"]
    assert len(parse_spans) == 3
    counts = [int(s.fields["records"]) for s in parse_spans]
    assert counts == sorted(counts)
    assert counts[-1] == 3
