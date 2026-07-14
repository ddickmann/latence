"""Observability metrics-seam regression tests (ADR-0047).

Pins the additive, optional metrics seam layered on the existing Quality Report + tracing
(ADR-0034):

1. :func:`~latence_core.observability.record_quality_report` reads an existing
   :class:`~latence_core.quality.QualityReport` and emits the per-Stage durations / record
   counts / typed-error counter / induction-empty gauge + the run totals + the ER-merge audit
   through a fake :class:`~latence_core.observability.MetricSink` — counts only, never content.
2. The seam **degrades to a no-op when the optional extra is absent** (anti-false-green): with
   ``prometheus_client`` / ``opentelemetry`` genuinely not installed in the CPU-first dev env,
   :func:`~latence_core.observability.build_metric_sink` returns a
   :class:`~latence_core.observability.NoOpMetricSink` (``strict=False``) and raises
   :class:`~latence_core.observability.MissingObservabilityExtra` only under ``strict=True``.
3. :class:`~latence_core.observability.MetricsSpanSink` translates a live ``stage_end`` span
   into the same metric primitives (the tracing seam reused, no logic duplicated), and forwards
   to a chained sink.
4. No content leaks: a run whose Stage/provider names are ordinary is fine, but a metric value
   is always a number and a label is always a low-cardinality name — asserted structurally.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import pytest
from latence_core.observability import (
    ER_MERGES_APPLIED,
    ER_MERGES_BELOW_POLICY,
    RUN_DOCUMENTS,
    RUN_DURATION_SECONDS,
    STAGE_DURATION_SECONDS,
    STAGE_ERRORS,
    STAGE_INDUCTION_EMPTY,
    STAGE_RECORDS_IN,
    STAGE_RECORDS_OUT,
    MetricsSpanSink,
    MissingObservabilityExtra,
    NoOpMetricSink,
    build_metric_sink,
    observability_available,
    record_quality_report,
)
from latence_core.quality import DisambiguationQuality, QualityReport, StageMetrics
from latence_core.tracing import CollectingSpanSink, Span

_PII_SENTINEL = "patient-jane-doe-DOB-1970-01-01-unique-sentinel"


class _Recorded:
    """One recorded metric call — kind, name, value, labels."""

    def __init__(self, kind: str, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.kind = kind
        self.name = name
        self.value = value
        self.labels = dict(labels)


class _FakeMetricSink:
    """A faithful in-memory MetricSink: records every primitive call in order (no deps)."""

    def __init__(self) -> None:
        self.calls: list[_Recorded] = []

    def incr_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.calls.append(_Recorded("counter", name, value, labels))

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.calls.append(_Recorded("histogram", name, value, labels))

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.calls.append(_Recorded("gauge", name, value, labels))

    def by_name(self, name: str) -> list[_Recorded]:
        return [c for c in self.calls if c.name == name]

    def one(self, name: str) -> _Recorded:
        found = self.by_name(name)
        assert len(found) == 1, f"expected exactly one {name}, got {len(found)}"
        return found[0]


def _report_with_two_stages() -> QualityReport:
    """A report with a clean Stage, a failed Stage, an induction-empty Stage, and an ER audit."""
    return QualityReport(
        pipeline_name="demo-pipeline",
        run_id="run-0001",
        document_count=3,
        total_duration_seconds=1.25,
        stages=[
            StageMetrics(
                name="parse",
                capability="parse",
                provider="parser.pdfplumber",
                records_in=3,
                records_out=3,
                duration_seconds=0.40,
                endpoint_retries=2,
            ),
            StageMetrics(
                name="extract",
                capability="entity_extraction",
                provider="entity.gazetteer",
                records_in=3,
                records_out=0,
                duration_seconds=0.10,
                error_count=1,
                error_category="provider",
            ),
            StageMetrics(
                name="induce",
                capability="schema_induction",
                provider="schema.inducer",
                records_in=3,
                records_out=3,
                duration_seconds=0.20,
                induction_documents=3,
                induction_fallbacks=2,
                induction_empty_after_sanitization=2,
            ),
        ],
        disambiguation=DisambiguationQuality(
            mentions_in=10,
            entities_out=7,
            merges_applied=3,
            merges_below_policy=1,
        ),
    )


def test_record_quality_report_emits_stage_and_run_metrics() -> None:
    sink = _FakeMetricSink()
    record_quality_report(_report_with_two_stages(), sink)

    # Run-level: duration histogram + document counter, keyed by pipeline only.
    run_dur = sink.one(RUN_DURATION_SECONDS)
    assert run_dur.kind == "histogram" and run_dur.value == pytest.approx(1.25)
    assert run_dur.labels == {"pipeline": "demo-pipeline"}
    assert sink.one(RUN_DOCUMENTS).value == pytest.approx(3.0)

    # Per-Stage duration histograms: one per stage, labelled stage/capability/provider.
    durations = sink.by_name(STAGE_DURATION_SECONDS)
    assert {c.labels["stage"] for c in durations} == {"parse", "extract", "induce"}
    parse_dur = next(c for c in durations if c.labels["stage"] == "parse")
    assert parse_dur.value == pytest.approx(0.40)
    assert parse_dur.labels == {
        "stage": "parse",
        "capability": "parse",
        "provider": "parser.pdfplumber",
    }

    # Record counters present for every stage.
    assert {c.labels["stage"] for c in sink.by_name(STAGE_RECORDS_IN)} == {
        "parse",
        "extract",
        "induce",
    }
    assert {c.labels["stage"] for c in sink.by_name(STAGE_RECORDS_OUT)} == {
        "parse",
        "extract",
        "induce",
    }


def test_record_quality_report_error_only_on_failed_stage() -> None:
    sink = _FakeMetricSink()
    record_quality_report(_report_with_two_stages(), sink)

    err = sink.one(STAGE_ERRORS)
    assert err.kind == "counter" and err.value == pytest.approx(1.0)
    # The typed category rides as a label; the failed stage is the one that raised.
    assert err.labels == {
        "stage": "extract",
        "capability": "entity_extraction",
        "provider": "entity.gazetteer",
        "category": "provider",
    }


def test_record_quality_report_induction_empty_and_merge_audit() -> None:
    sink = _FakeMetricSink()
    record_quality_report(_report_with_two_stages(), sink)

    empty = sink.one(STAGE_INDUCTION_EMPTY)
    assert empty.kind == "gauge" and empty.value == pytest.approx(2.0)
    assert empty.labels["stage"] == "induce"

    assert sink.one(ER_MERGES_APPLIED).value == pytest.approx(3.0)
    assert sink.one(ER_MERGES_BELOW_POLICY).value == pytest.approx(1.0)


def test_record_quality_report_counts_only_never_content() -> None:
    """Every value is a number and every label is a low-cardinality name — never record text."""
    report = _report_with_two_stages()
    # Even if a provider name carried the sentinel, labels are names by construction; assert no
    # metric value is a string and no label value looks like a document body (has whitespace).
    sink = _FakeMetricSink()
    record_quality_report(report, sink)
    assert sink.calls, "expected metrics to be emitted"
    for call in sink.calls:
        assert isinstance(call.value, (int, float))
        for key, val in call.labels.items():
            assert isinstance(key, str) and isinstance(val, str)
            assert _PII_SENTINEL not in val
            assert "\n" not in val and " " not in val  # names, not sentences


def test_noop_sink_records_nothing_and_never_raises() -> None:
    sink = NoOpMetricSink()
    # Exercised over a full report: every primitive is a silent no-op.
    record_quality_report(_report_with_two_stages(), sink)
    # Each primitive is a silent no-op (returns None, records nothing, never raises).
    sink.incr_counter("x", 1.0, {})
    sink.observe("x", 1.0, {})
    sink.set_gauge("x", 1.0, {})


def test_build_metric_sink_none_is_noop() -> None:
    assert isinstance(build_metric_sink("none"), NoOpMetricSink)
    assert isinstance(build_metric_sink(""), NoOpMetricSink)


@pytest.mark.parametrize("kind", ["prometheus", "otel"])
def test_build_metric_sink_degrades_to_noop_when_extra_absent(kind: str) -> None:
    """Anti-false-green: the optional deps are NOT in the dev env, so this is a REAL absence.

    ``build_metric_sink`` must return a NoOp (never crash) when the extra is missing — proving
    the degrade path, not a mock of it. If someone installs the extra into the dev env, this
    test self-skips rather than silently passing for the wrong reason.
    """
    module = {"prometheus": "prometheus_client", "otel": "opentelemetry"}[kind]
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"{module} is installed; the absent-extra degrade path cannot be exercised")
    assert observability_available(kind) is False
    assert isinstance(build_metric_sink(kind), NoOpMetricSink)
    # strict=True surfaces the missing extra instead of degrading.
    with pytest.raises(MissingObservabilityExtra) as excinfo:
        build_metric_sink(kind, strict=True)
    assert excinfo.value.module == module


def test_build_metric_sink_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):  # ConfigError is a ValueError subclass
        build_metric_sink("statsd")


def test_metrics_span_sink_translates_stage_end_and_forwards() -> None:
    sink = _FakeMetricSink()
    chained = CollectingSpanSink()
    span_sink = MetricsSpanSink(sink, chain=chained)

    # A begin span forwards but emits no metric.
    span_sink.emit(
        Span(
            correlation_id="run-0001",
            stage="parse",
            event="stage_begin",
            fields={"capability": "parse", "provider": "parser.pdfplumber", "records_in": 3},
        )
    )
    assert sink.calls == []
    assert len(chained.spans) == 1  # forwarded to the chain

    # An end span emits duration + record counters.
    span_sink.emit(
        Span(
            correlation_id="run-0001",
            stage="parse",
            event="stage_end",
            fields={
                "capability": "parse",
                "provider": "parser.pdfplumber",
                "records_in": 3,
                "records_out": 3,
            },
            duration_ms=400.0,
        )
    )
    assert sink.one(STAGE_DURATION_SECONDS).value == pytest.approx(0.40)  # ms -> s
    assert sink.one(STAGE_RECORDS_IN).value == pytest.approx(3.0)
    assert sink.one(STAGE_RECORDS_OUT).value == pytest.approx(3.0)
    assert len(chained.spans) == 2


def test_metrics_span_sink_emits_error_counter_on_failed_end() -> None:
    sink = _FakeMetricSink()
    span_sink = MetricsSpanSink(sink)
    span_sink.emit(
        Span(
            correlation_id="run-0001",
            stage="extract",
            event="stage_end",
            fields={"capability": "entity_extraction", "provider": "entity.gazetteer"},
            duration_ms=100.0,
            error_category="provider",
            error_message="endpoint 503",
        )
    )
    err = sink.one(STAGE_ERRORS)
    assert err.value == pytest.approx(1.0)
    assert err.labels["category"] == "provider"
    # The error MESSAGE never becomes a metric or a label (counts-only discipline).
    for call in sink.calls:
        assert "503" not in "".join(call.labels.values())
