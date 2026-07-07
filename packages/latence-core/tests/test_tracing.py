"""Tracing-span seam regression tests (#73).

Pins the properties a revert of ``tracing.py`` (or its Runner wiring) would break:

1. A run-scoped **correlation id** stamps every span, so all spans of one run are greppable
   together and a downstream tracer can reconstruct the per-run tree.
2. Spans carry **counts + typed-error categories only, never record content** — the no-PII
   discipline. The adversarial test feeds a run whose document text is a unique PII-shaped
   sentinel and asserts that sentinel appears in **no** emitted span.
3. The default sink is **silent** (DEBUG on ``latence.trace``), so a normal run's logging is
   unchanged; an explicit :class:`CollectingSpanSink` captures the spans deterministically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from _pytest.logging import LogCaptureFixture
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.tracing import (
    CollectingSpanSink,
    LoggingSpanSink,
    RunTracer,
    Span,
)

# A unique, PII-shaped sentinel that appears ONLY in the document body. If it leaks into any
# span field, the no-PII invariant is broken. Made distinctive so a substring scan is exact.
_PII_SENTINEL = "SSN-000-11-2222-secretname-Zzyzx-unique-sentinel"


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: trace-slice
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
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


def test_span_as_dict_is_deterministic_and_omits_empty_optionals() -> None:
    span = Span(
        correlation_id="run-x",
        stage="parse",
        event="stage_end",
        fields={"records_out": 3, "capability": "parse"},
        duration_ms=1.5,
    )
    d = span.as_dict()
    # Fixed leading keys, then sorted fields; no error keys on a clean end.
    assert list(d)[:3] == ["correlation_id", "stage", "event"]
    assert d["duration_ms"] == 1.5
    assert "error_category" not in d and "error_message" not in d
    # Field keys are emitted in sorted order (determinism).
    field_keys = [k for k in d if k not in {"correlation_id", "stage", "event", "duration_ms"}]
    assert field_keys == sorted(field_keys)


def test_run_stamps_every_span_with_the_run_id_correlation(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    LocalRunner(span_sink=sink).run(_pipeline(store, docs_dir), run_id="trace-run-1")
    assert sink.spans, "the run must have emitted spans"
    # Every span shares the run-scoped correlation id.
    assert {s.correlation_id for s in sink.spans} == {"trace-run-1"}
    # Every executed Stage emitted a begin and an end span (fresh run, nothing from checkpoint).
    begins = {s.stage for s in sink.spans if s.event == "stage_begin"}
    ends = {s.stage for s in sink.spans if s.event == "stage_end"}
    assert begins == ends == {"source", "parse", "export"}


def test_spans_carry_no_record_content_adversarial(tmp_path: Path) -> None:
    # A document whose entire body is the PII sentinel — the adversarial input.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leak.txt").write_text(f"{_PII_SENTINEL}\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    sink = CollectingSpanSink()
    LocalRunner(span_sink=sink).run(_pipeline(store, docs), run_id="leak-run")

    assert sink.spans
    for span in sink.spans:
        blob = repr(span.as_dict())
        assert _PII_SENTINEL not in blob, f"PII leaked into a span: {span.event}/{span.stage}"
        # Every field value is a safe low-cardinality scalar (count/name), never a text body.
        for value in span.fields.values():
            assert isinstance(value, (str, int, float, bool))
        # The only strings a span carries are the stable stage/capability/provider names —
        # none is the document body.
        for value in span.fields.values():
            if isinstance(value, str):
                assert _PII_SENTINEL not in value


def test_default_sink_is_silent_at_info_level(
    docs_dir: Path, tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    # With no sink supplied, the default LoggingSpanSink logs at DEBUG on ``latence.trace``.
    # At INFO capture level a normal run emits no trace records — stdout/logging is unchanged.
    store = f"file://{tmp_path}/store"
    with caplog.at_level(logging.INFO, logger="latence.trace"):
        LocalRunner().run(_pipeline(store, docs_dir), run_id="silent-run")
    assert [r for r in caplog.records if r.name == "latence.trace"] == []


def test_logging_sink_emits_structured_records_at_debug(caplog: LogCaptureFixture) -> None:
    sink = LoggingSpanSink()
    tracer = RunTracer("run-dbg", sink)
    with caplog.at_level(logging.DEBUG, logger="latence.trace"):
        tracer.stage_begin("parse", capability="parse", provider="parser.plaintext", records_in=2)
        tracer.stage_end(
            "parse",
            capability="parse",
            provider="parser.plaintext",
            records_in=2,
            records_out=2,
            duration_ms=0.5,
        )
    trace_records = [r for r in caplog.records if r.name == "latence.trace"]
    assert len(trace_records) == 2
    # The correlation id + counts ride as structured ``extra`` fields, not an opaque string —
    # read them off the record's ``__dict__`` (where ``logging`` merges ``extra``), which is
    # both type-clean (LogRecord has no static ``correlation_id`` attr) and lint-clean.
    begin = trace_records[0].__dict__
    assert begin["correlation_id"] == "run-dbg"
    assert begin["records_in"] == 2
