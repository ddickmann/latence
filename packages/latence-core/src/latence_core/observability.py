"""Optional metrics + OpenTelemetry seam (ADR-0047), layered on the Quality Report + tracing.

ADR-0034 already gave ``latence-core`` two dependency-free observability primitives: a typed
error taxonomy (:mod:`latence_core.errors`) and a thin per-Stage tracing seam
(:mod:`latence_core.tracing`). This module adds the *export* half an operator asks for in
production — **Prometheus counters/histograms** and/or an **OpenTelemetry meter** — without
breaking the CPU-first, offline, thin-core promise (ADR-0016):

* The metric *shape* (what is emitted, from which fields) lives here behind a tiny
  :class:`MetricSink` Protocol. The Protocol has three primitives — counter / histogram /
  gauge — and nothing else; :func:`record_quality_report` is the single, dep-free function
  that walks a :class:`~latence_core.quality.QualityReport` and its
  :class:`~latence_core.quality.StageMetrics` and calls those primitives. It **reads** the
  existing report (per-Stage durations, record counts, the typed ``error_category``, the
  induction-empty flag, the ER merge audit) — it never recomputes or duplicates them.
* The *backends* — :class:`PrometheusMetricSink`, :class:`OpenTelemetryMetricSink` — **lazily
  import** their optional package (``prometheus_client`` / ``opentelemetry``) at construction,
  exactly the provider-optional pattern the heavy Provider packages use (ADR-0016). Neither is
  a ``latence-core`` runtime dependency: they live in the ``observability`` optional-dependency
  group. When the extra is absent, :func:`build_metric_sink` **degrades to a
  :class:`NoOpMetricSink`** (a warning, never a crash), so a default install that never opts in
  carries no ``opentelemetry-*`` / ``prometheus-client`` tree and a run behaves identically.
* :class:`MetricsSpanSink` reuses the *existing* tracing seam: it is a
  :class:`~latence_core.tracing.SpanSink` that translates live ``stage_end`` spans into the
  same metric primitives, so an operator who wants live per-Stage metrics passes
  ``LocalRunner(span_sink=MetricsSpanSink(sink))`` — no Runner change, no metric logic
  duplicated (it forwards to the same :class:`MetricSink`).

Counts only, never content — the same no-PII discipline the tracing seam and the Quality Report
keep. Every value emitted is a duration, a record count, or an error/merge tally; every label is
a Stage/capability/provider/category/pipeline *name*. A document body, a PII surface, or a secret
never enters a metric or a label (asserted adversarially in the seam's regression test).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from latence_core.errors import ConfigError

if TYPE_CHECKING:
    from latence_core.quality import QualityReport
    from latence_core.tracing import Span, SpanSink

logger = logging.getLogger("latence.observability")

# --- Metric names (stable, Prometheus-friendly: lower_snake, unit-suffixed) ---------------
# One place so the docs, the Prometheus backend, and the OTel backend agree byte-for-byte.
STAGE_DURATION_SECONDS = "latence_stage_duration_seconds"
STAGE_RECORDS_IN = "latence_stage_records_in_total"
STAGE_RECORDS_OUT = "latence_stage_records_out_total"
STAGE_ERRORS = "latence_stage_errors_total"
STAGE_ENDPOINT_RETRIES = "latence_stage_endpoint_retries_total"
STAGE_INDUCTION_EMPTY = "latence_stage_induction_empty_documents"
RUN_DURATION_SECONDS = "latence_run_duration_seconds"
RUN_DOCUMENTS = "latence_run_documents_total"
ER_MERGES_APPLIED = "latence_er_merges_applied_total"
ER_MERGES_BELOW_POLICY = "latence_er_merges_below_policy_total"

# The fixed label schema per metric name. A backend that needs declared label keys up front
# (Prometheus) reads this; :func:`record_quality_report` passes labels that conform to it. The
# stage metrics share ``(stage, capability, provider)``; the error metric adds the typed
# ``category``; the run/ER metrics are keyed by ``pipeline`` only. Low-cardinality, name-only —
# never a record value.
_STAGE_LABELS: tuple[str, ...] = ("stage", "capability", "provider")
_RUN_LABELS: tuple[str, ...] = ("pipeline",)
METRIC_LABELS: dict[str, tuple[str, ...]] = {
    STAGE_DURATION_SECONDS: _STAGE_LABELS,
    STAGE_RECORDS_IN: _STAGE_LABELS,
    STAGE_RECORDS_OUT: _STAGE_LABELS,
    STAGE_ERRORS: (*_STAGE_LABELS, "category"),
    STAGE_ENDPOINT_RETRIES: _STAGE_LABELS,
    STAGE_INDUCTION_EMPTY: _STAGE_LABELS,
    RUN_DURATION_SECONDS: _RUN_LABELS,
    RUN_DOCUMENTS: _RUN_LABELS,
    ER_MERGES_APPLIED: _RUN_LABELS,
    ER_MERGES_BELOW_POLICY: _RUN_LABELS,
}

# Human-readable help strings (Prometheus HELP lines / OTel descriptions).
METRIC_HELP: dict[str, str] = {
    STAGE_DURATION_SECONDS: "Wall-clock duration of a pipeline Stage, in seconds.",
    STAGE_RECORDS_IN: "Records a Stage consumed.",
    STAGE_RECORDS_OUT: "Records a Stage produced.",
    STAGE_ERRORS: "Stage failures, labelled by the typed error category.",
    STAGE_ENDPOINT_RETRIES: "Transient endpoint retries a Stage absorbed before succeeding.",
    STAGE_INDUCTION_EMPTY: "Documents an induction Stage degraded to the bare config-label floor.",
    RUN_DURATION_SECONDS: "Total wall-clock duration of a pipeline run, in seconds.",
    RUN_DOCUMENTS: "Documents a pipeline run processed.",
    ER_MERGES_APPLIED: "Entity-resolution merges applied (the audited merge log).",
    ER_MERGES_BELOW_POLICY: "Entity-resolution merges recorded but NOT applied (below policy).",
}


class MetricSink(Protocol):
    """The thin seam a metrics backend implements — three primitives, nothing else.

    Deliberately minimal so :func:`record_quality_report` (and :class:`MetricsSpanSink`) drive
    *any* backend — Prometheus, OpenTelemetry, a test fake, or the no-op — through the same
    calls. ``labels`` are always low-cardinality *names* (Stage/capability/provider/category/
    pipeline), never a record value.
    """

    def incr_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Add ``value`` (>= 0) to a monotonic counter."""
        ...

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Record one observation of ``value`` into a histogram/distribution."""
        ...

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        """Set a gauge to ``value``."""
        ...


class NoOpMetricSink:
    """The graceful-degrade sink: every primitive is a no-op.

    Returned by :func:`build_metric_sink` when the requested extra is not installed (or the
    caller asked for ``"none"``), so wiring metrics in is always safe: an operator who has not
    installed ``latence-core[observability]`` gets exactly the pre-metrics behaviour, no import
    of any optional package, no error.
    """

    def incr_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        return None

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        return None

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        return None


def record_quality_report(report: QualityReport, sink: MetricSink) -> None:
    """Emit the run's metrics from an existing :class:`~latence_core.quality.QualityReport`.

    The single, dependency-free translation from the report the Runner already built to the
    :class:`MetricSink` primitives. It **reads** the report — it never recomputes a duration or
    a count, and never touches record content:

    * per Stage (:class:`~latence_core.quality.StageMetrics`): the duration histogram, the
      records-in / records-out counters, the endpoint-retry counter, and — only on a failed
      Stage — the error counter labelled with the stable typed ``error_category``; and — only
      for an induction Stage that degraded — the induction-empty gauge;
    * per run: the total-duration histogram and the document counter;
    * the ER merge audit (when the run had a Disambiguation Stage): merges applied vs recorded
      below policy — the no-silent-over-merge audit, surfaced as counters.

    Safe to call with any sink; with :class:`NoOpMetricSink` it is inert.
    """
    run_labels = {"pipeline": report.pipeline_name}
    sink.observe(RUN_DURATION_SECONDS, report.total_duration_seconds, run_labels)
    sink.incr_counter(RUN_DOCUMENTS, float(report.document_count), run_labels)

    for stage in report.stages:
        labels = {
            "stage": stage.name,
            "capability": stage.capability,
            "provider": stage.provider,
        }
        sink.observe(STAGE_DURATION_SECONDS, stage.duration_seconds, labels)
        sink.incr_counter(STAGE_RECORDS_IN, float(stage.records_in), labels)
        sink.incr_counter(STAGE_RECORDS_OUT, float(stage.records_out), labels)
        if stage.endpoint_retries:
            sink.incr_counter(STAGE_ENDPOINT_RETRIES, float(stage.endpoint_retries), labels)
        if stage.error_count:
            err_labels = {**labels, "category": stage.error_category or "unknown"}
            sink.incr_counter(STAGE_ERRORS, float(stage.error_count), err_labels)
        if stage.induction_empty_after_sanitization:
            sink.set_gauge(
                STAGE_INDUCTION_EMPTY,
                float(stage.induction_empty_after_sanitization),
                labels,
            )

    disambiguation = report.disambiguation
    if disambiguation is not None:
        sink.incr_counter(ER_MERGES_APPLIED, float(disambiguation.merges_applied), run_labels)
        sink.incr_counter(
            ER_MERGES_BELOW_POLICY, float(disambiguation.merges_below_policy), run_labels
        )


class MetricsSpanSink:
    """A :class:`~latence_core.tracing.SpanSink` that emits metrics from live Stage spans.

    Layered on the *existing* tracing seam (ADR-0034): an operator who wants live per-Stage
    metrics (a Stage finishing pushes its duration/counts immediately, not only at the end-of-run
    report) passes ``LocalRunner(span_sink=MetricsSpanSink(sink))``. It translates a
    ``stage_end`` span into the same :class:`MetricSink` primitives
    :func:`record_quality_report` uses — no metric logic is duplicated. An optional ``chain``
    sink is emitted first, so metrics can ride alongside the default
    :class:`~latence_core.tracing.LoggingSpanSink` (or a :class:`CollectingSpanSink` in a test).
    Only ``stage_end`` carries the terminal counts/duration, so ``stage_begin`` / ``stage_batch``
    are forwarded-only (no metric).
    """

    def __init__(self, sink: MetricSink, *, chain: SpanSink | None = None) -> None:
        self._sink = sink
        self._chain = chain

    def emit(self, span: Span) -> None:
        if self._chain is not None:
            self._chain.emit(span)
        if span.event != "stage_end":
            return
        labels = {
            "stage": span.stage,
            "capability": str(span.fields.get("capability", "")),
            "provider": str(span.fields.get("provider", "")),
        }
        if span.duration_ms is not None:
            self._sink.observe(STAGE_DURATION_SECONDS, span.duration_ms / 1000.0, labels)
        records_in = span.fields.get("records_in")
        if isinstance(records_in, (int, float)) and not isinstance(records_in, bool):
            self._sink.incr_counter(STAGE_RECORDS_IN, float(records_in), labels)
        records_out = span.fields.get("records_out")
        if isinstance(records_out, (int, float)) and not isinstance(records_out, bool):
            self._sink.incr_counter(STAGE_RECORDS_OUT, float(records_out), labels)
        if span.error_category is not None:
            self._sink.incr_counter(STAGE_ERRORS, 1.0, {**labels, "category": span.error_category})


class MissingObservabilityExtra(ImportError):
    """Raised when a metrics backend is requested but its optional package is not installed.

    Carries the ``kind`` (``"prometheus"`` / ``"otel"``) and the missing ``module`` so a caller
    can print an actionable install hint. :func:`build_metric_sink` raises it only in
    ``strict=True`` mode; the default degrades to :class:`NoOpMetricSink`.
    """

    def __init__(self, kind: str, module: str) -> None:
        super().__init__(
            f"the '{module}' package is required for the '{kind}' metrics sink but is not "
            f"installed; install the optional group: pip install 'latence-core[observability]'"
        )
        self.kind = kind
        self.module = module


def _conform(name: str, labels: Mapping[str, str]) -> dict[str, str]:
    """Coerce ``labels`` to a metric's declared schema (missing key -> "", extras dropped).

    Prometheus fixes a metric's label *keys* at creation, so every call for one metric name
    must present the same keys; this makes that true regardless of what the caller passed.
    """
    schema = METRIC_LABELS.get(name)
    if schema is None:
        return dict(labels)
    return {key: labels.get(key, "") for key in schema}


class PrometheusMetricSink:
    """A :class:`MetricSink` backed by ``prometheus_client`` (optional extra).

    Lazily imports ``prometheus_client`` at construction — absent it raises
    :class:`MissingObservabilityExtra` (the provider-optional pattern; the package is only in the
    ``observability`` optional group, never a core dependency). Metrics are created on first use
    against a :class:`~prometheus_client.CollectorRegistry` (a fresh, private one by default so a
    batch run's metrics do not leak into the global registry) with the declared label schema.
    Because the pipeline is a **batch** Job (not a long-lived server), the pull model needs a
    bridge: call :meth:`write_textfile` to dump the registry for the node-exporter textfile
    collector, or push the registry to a Pushgateway (both shown in ``docs/observability.md``).
    """

    def __init__(self, registry: Any | None = None) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError as exc:  # pragma: no cover - exercised only where the extra is absent
            raise MissingObservabilityExtra("prometheus", "prometheus_client") from exc
        self._registry: Any = registry if registry is not None else CollectorRegistry()
        self._Counter = Counter
        self._Gauge = Gauge
        self._Histogram = Histogram
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    @property
    def registry(self) -> Any:
        """The underlying ``CollectorRegistry`` (to scrape, push, or write to a textfile)."""
        return self._registry

    def _metric(self, cache: dict[str, Any], factory: Any, name: str) -> Any:
        metric = cache.get(name)
        if metric is None:
            metric = factory(
                name,
                METRIC_HELP.get(name, name),
                labelnames=METRIC_LABELS.get(name, ()),
                registry=self._registry,
            )
            cache[name] = metric
        return metric

    def incr_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._metric(self._counters, self._Counter, name).labels(**_conform(name, labels)).inc(
            value
        )

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._metric(self._histograms, self._Histogram, name).labels(
            **_conform(name, labels)
        ).observe(value)

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self._metric(self._gauges, self._Gauge, name).labels(**_conform(name, labels)).set(value)

    def write_textfile(self, path: str) -> None:
        """Dump the registry to ``path`` for the Prometheus node-exporter textfile collector."""
        from prometheus_client import write_to_textfile

        write_to_textfile(path, self._registry)


class OpenTelemetryMetricSink:
    """A :class:`MetricSink` backed by an OpenTelemetry :class:`~opentelemetry.metrics.Meter`.

    Lazily imports ``opentelemetry`` at construction — absent it raises
    :class:`MissingObservabilityExtra`. Instruments are created on first use from the meter;
    labels become OTel attributes. The caller configures the SDK's ``MeterProvider`` /
    exporter (OTLP to a collector, Prometheus reader, …) at the edge — this sink only records,
    keeping ``latence-core`` free of any exporter/transport choice (``docs/observability.md``
    shows a collector config).
    """

    def __init__(self, meter: Any | None = None) -> None:
        try:
            from opentelemetry import metrics
        except ImportError as exc:  # pragma: no cover - exercised only where the extra is absent
            raise MissingObservabilityExtra("otel", "opentelemetry") from exc
        self._meter: Any = meter if meter is not None else metrics.get_meter("latence")
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    def incr_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        counter = self._counters.get(name)
        if counter is None:
            counter = self._meter.create_counter(name, description=METRIC_HELP.get(name, name))
            self._counters[name] = counter
        counter.add(value, dict(labels))

    def observe(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        histogram = self._histograms.get(name)
        if histogram is None:
            histogram = self._meter.create_histogram(name, description=METRIC_HELP.get(name, name))
            self._histograms[name] = histogram
        histogram.record(value, dict(labels))

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        gauge = self._gauges.get(name)
        if gauge is None:
            # ``create_gauge`` is the synchronous gauge (OTel SDK >= 1.23); fall back to a
            # histogram on an older SDK so the seam degrades rather than crashes.
            create_gauge = getattr(self._meter, "create_gauge", None)
            gauge = (
                create_gauge(name, description=METRIC_HELP.get(name, name))
                if create_gauge is not None
                else self._meter.create_histogram(name, description=METRIC_HELP.get(name, name))
            )
            self._gauges[name] = gauge
        setter = getattr(gauge, "set", None)
        if setter is not None:
            setter(value, dict(labels))
        else:  # pragma: no cover - only on the histogram fallback path
            gauge.record(value, dict(labels))


def observability_available(kind: str) -> bool:
    """Is the optional package for metrics backend ``kind`` importable in this environment?

    A cheap :func:`importlib.util.find_spec` probe (no import side effects), so a caller can
    branch on availability without catching :class:`MissingObservabilityExtra`. ``"none"`` is
    always available (the no-op needs no package).
    """
    module = _MODULE_FOR_KIND.get(kind.lower())
    if module is None:
        return kind.lower() in _NOOP_KINDS
    return importlib.util.find_spec(module) is not None


_NOOP_KINDS: frozenset[str] = frozenset({"none", "noop", ""})
_MODULE_FOR_KIND: dict[str, str] = {
    "prometheus": "prometheus_client",
    "otel": "opentelemetry",
    "opentelemetry": "opentelemetry",
}
_CTOR_FOR_KIND: dict[str, type[MetricSink]] = {
    "prometheus": PrometheusMetricSink,
    "otel": OpenTelemetryMetricSink,
    "opentelemetry": OpenTelemetryMetricSink,
}


def build_metric_sink(kind: str, *, strict: bool = False) -> MetricSink:
    """Build a :class:`MetricSink` for ``kind``, degrading to no-op when the extra is absent.

    ``kind`` is ``"none"`` / ``"prometheus"`` / ``"otel"`` (case-insensitive). ``"none"`` always
    returns a :class:`NoOpMetricSink`. For a real backend whose optional package is **not
    installed**, the default (``strict=False``) logs a warning and returns
    :class:`NoOpMetricSink` — so wiring metrics into a pipeline is always safe on a default
    install; ``strict=True`` raises :class:`MissingObservabilityExtra` instead (for a deploy
    that means to fail if its observability stack is missing). An unknown ``kind`` is a
    :class:`~latence_core.errors.ConfigError`.
    """
    normalized = kind.lower()
    if normalized in _NOOP_KINDS:
        return NoOpMetricSink()
    ctor = _CTOR_FOR_KIND.get(normalized)
    module = _MODULE_FOR_KIND.get(normalized)
    if ctor is None or module is None:
        raise ConfigError(
            f"unknown metrics sink kind {kind!r}; expected one of "
            f"'none', 'prometheus', 'otel'"
        )
    if importlib.util.find_spec(module) is None:
        if strict:
            raise MissingObservabilityExtra(normalized, module)
        logger.warning(
            "metrics backend %r requested but %r is not installed; metrics disabled (no-op). "
            "Install the optional group: pip install 'latence-core[observability]'",
            normalized,
            module,
        )
        return NoOpMetricSink()
    return ctor()
