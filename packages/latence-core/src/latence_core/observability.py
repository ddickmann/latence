"""Optional metrics, OpenTelemetry + OpenLineage seams, layered on the Quality Report + tracing.

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

The module's second half (#185) is the **lineage** export: the Runner emits standard OpenLineage
``RunEvent``\ s — START/COMPLETE/FAIL per Stage, with the datasets each Stage reads and writes and
a ``schema`` facet from its carrier contract — so any OpenLineage consumer (Marquez, Atlan,
Collibra, Dataplex) sees a latence run in the catalog the platform team already governs with. Same
posture: off by default, stdlib-only on the wire (no ``openlineage-python`` dependency), and an
emission failure is logged, never fatal.

Counts only, never content — the same no-PII discipline the tracing seam and the Quality Report
keep. Every value emitted is a duration, a record count, an error/merge tally, a
Stage/capability/provider/category/pipeline *name*, a dataset URI, or a schema *field name*. A
document body, a PII surface, a secret, or an exception message never enters a metric, a label, or
a lineage facet (asserted adversarially in both seams' regression tests).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, get_args, get_origin

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


# =========================================================================================
# OpenLineage run events (#185) — the standard-vocabulary lineage seam
# =========================================================================================
#
# One emitter buys every OpenLineage consumer at once (Marquez, Atlan, Collibra, Dataplex):
# the Runner emits a START / COMPLETE / FAIL ``RunEvent`` per Stage — plus a parent event for
# the run as a whole — with the Stage's input/output **datasets** (the Storage URIs it actually
# reads and writes) carrying a ``schema`` facet derived from the Stage's carrier contract.
#
# Three rules this section is built around, each pinned by a regression test:
#
# * **Counts only, never content** (ADR-0034). A facet carries record counts, a duration, a
#   Stage/capability/provider *name*, a dataset URI, and a *field name + type* schema — never a
#   record's text, a PII surface, a secret, or an exception message. In particular the standard
#   ``errorMessage`` run facet is **deliberately not emitted**: an exception message can quote a
#   document. A failed run carries the stable typed error *category* instead.
# * **Off by default, zero-cost when off.** ``LocalRunner`` takes ``lineage=None`` by default and
#   every emission site is guarded, so a default run builds no event and opens no socket.
# * **Emission never fails a run.** Every transport call (and the event-building around it) is
#   wrapped: a dead endpoint, a 500, a serialisation bug — all are logged and swallowed. Lineage
#   is observability, not a correctness gate.
#
# ``latence-core`` stays near-zero-dependency (ADR-0016): the wire format is built with ``json``
# and shipped with ``urllib`` — the ``openlineage-python`` client is NOT a dependency.

OPENLINEAGE_SPEC_VERSION = "2-0-2"
_OL_SPEC_BASE = f"https://openlineage.io/spec/{OPENLINEAGE_SPEC_VERSION}/OpenLineage.json"

#: ``schemaURL`` every emitted RunEvent declares (the OpenLineage spec version it conforms to).
RUN_EVENT_SCHEMA_URL = f"{_OL_SPEC_BASE}#/$defs/RunEvent"
_SCHEMA_FACET_URL = f"{_OL_SPEC_BASE}#/$defs/SchemaDatasetFacet"
_PARENT_FACET_URL = f"{_OL_SPEC_BASE}#/$defs/ParentRunFacet"
_NOMINAL_TIME_FACET_URL = f"{_OL_SPEC_BASE}#/$defs/NominalTimeRunFacet"
_JOB_TYPE_FACET_URL = (
    "https://openlineage.io/spec/facets/2-0-3/JobTypeJobFacet.json#/$defs/JobTypeJobFacet"
)

#: ``producer`` stamped on every event + facet — who generated this lineage.
LINEAGE_PRODUCER = "https://github.com/ddickmann/latence-framework"
_LATENCE_FACET_URL = f"{LINEAGE_PRODUCER}#/facets/LatenceRunFacet"

#: The default OpenLineage *namespace* a run's jobs live in when the operator names none.
DEFAULT_LINEAGE_NAMESPACE = "latence"

#: The ``eventType`` values this emitter produces (a subset of the spec's enum).
_EVENT_TYPES = frozenset({"START", "RUNNING", "COMPLETE", "ABORT", "FAIL", "OTHER"})

_LINEAGE_UUID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def lineage_run_uuid(namespace: str, pipeline: str, run_id: str, stage: str | None = None) -> str:
    """The UUID a run (or one Stage of it) is known by in the lineage graph.

    OpenLineage requires ``run.runId`` to be a **UUID** — latence run ids are operator-chosen
    strings like ``"run-0001"``, so they are hashed into a v5 UUID. Deterministic by
    construction: the same ``(namespace, pipeline, run_id, stage)`` always yields the same UUID,
    so a *resumed* run continues the same lineage run rather than forking a new one in the
    catalog, and a test can assert the id without freezing a clock or a random seed.
    """
    key = f"{namespace}/{pipeline}/{run_id}" + (f"/{stage}" if stage is not None else "")
    return str(uuid.uuid5(_LINEAGE_UUID_NAMESPACE, key))


# The carrier-field type mapping: a Python annotation -> the coarse type name that goes in the
# OpenLineage schema facet. Names + coarse types only — a schema facet describes *shape*, so no
# value ever passes through here.
_SCALAR_TYPE_NAMES: dict[object, str] = {
    str: "string",
    bytes: "binary",
    bool: "boolean",
    int: "integer",
    float: "double",
}


def _field_type_name(annotation: object) -> str:
    """Coarse OpenLineage type name for a pydantic field annotation (shape, never a value)."""
    origin = get_origin(annotation)
    if origin is None:
        scalar = _SCALAR_TYPE_NAMES.get(annotation)
        if scalar is not None:
            return scalar
        if isinstance(annotation, type):
            return "struct" if hasattr(annotation, "model_fields") else "string"
        return "string"
    args = [a for a in get_args(annotation) if a is not type(None)]
    if origin in (list, tuple, set, frozenset, Sequence):
        return "array"
    if origin in (dict, Mapping):
        return "map"
    if len(args) == 1:  # Optional[X] / Annotated[X, ...] collapse to X
        return _field_type_name(args[0])
    return "string"


def schema_fields_of(carrier: type[Any] | None) -> tuple[tuple[str, str], ...]:
    """The ``(field name, type)`` pairs of a Stage's carrier — the schema facet's payload.

    Reads the carrier contract's declared fields (a pydantic model's ``model_fields``), so the
    schema a consumer sees is the *contract*, not a sample of the data: no record is inspected,
    no value is read. A carrier without declared fields yields an empty tuple (the facet is then
    omitted rather than emitted empty).
    """
    fields = getattr(carrier, "model_fields", None)
    if not isinstance(fields, dict):
        return ()
    return tuple(
        (name, _field_type_name(getattr(info, "annotation", None)))
        for name, info in sorted(fields.items())
    )


@dataclass(frozen=True)
class LineageDataset:
    """One dataset in the lineage graph: where it lives + the shape of what is in it.

    ``namespace`` / ``name`` follow the OpenLineage dataset naming convention (a local path is
    ``file`` + the absolute path; an object store is ``<scheme>://<bucket>`` + the key). ``fields``
    is the carrier's ``(name, type)`` schema — never a value.
    """

    namespace: str
    name: str
    fields: tuple[tuple[str, str], ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"namespace": self.namespace, "name": self.name}
        if self.fields:
            out["facets"] = {
                "schema": {
                    "_producer": LINEAGE_PRODUCER,
                    "_schemaURL": _SCHEMA_FACET_URL,
                    "fields": [{"name": n, "type": t} for n, t in self.fields],
                }
            }
        return out


def dataset_for_uri(uri: str, carrier: type[Any] | None = None) -> LineageDataset:
    """Map a latence Storage URI onto the OpenLineage dataset naming convention.

    ``file:///data/x.jsonl`` (or a bare path) -> namespace ``file``, name ``/data/x.jsonl``;
    ``s3://bucket/key`` -> namespace ``s3://bucket``, name ``key``. The optional ``carrier`` adds
    the schema facet. A URI is a *location*, not content — this is exactly the identity a catalog
    needs to stitch latence's outputs into its graph.
    """
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme in ("", "file"):
        namespace = "file"
        name = parsed.path or uri
    else:
        namespace = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else parsed.scheme
        name = parsed.path.lstrip("/")
    return LineageDataset(namespace=namespace, name=name or "/", fields=schema_fields_of(carrier))


def validate_run_event(event: Mapping[str, Any]) -> None:
    """Reject an event the OpenLineage spec rejects — raise :class:`ConfigError`.

    These are the **required/typed constraints of the OpenLineage 2-0-2 JSON Schema** itself
    (``RunEvent``/``BaseEvent``/``Run``/``Job``/``Dataset``/``BaseFacet``), checked locally so an
    offline test is never a *weaker* consumer than a real one — a fake that accepts what a catalog
    rejects manufactures false green. Note the direction: this is strictly no laxer than a real
    backend, and in places stricter — Marquez 0.51.1's ``POST /api/v1/lineage`` was measured to
    accept a non-UUID ``runId`` and a naive ``eventTime`` that the spec forbids and that a
    schema-validating consumer (the OpenLineage client's own models, a hosted catalog) rejects.
    Being stricter can only produce a false *red*, which the live-Marquez check
    (``scripts/verify-openlineage-marquez.py``) disproves.

    * ``eventType`` is one of the spec's enum values;
    * ``eventTime`` is ISO-8601 **with a timezone offset** (a naive timestamp is rejected);
    * ``run.runId`` is a real UUID (this is the constraint latence run ids must be hashed for);
    * ``job.namespace`` / ``job.name`` are non-empty;
    * ``producer`` and ``schemaURL`` are present;
    * every input/output dataset has a non-empty namespace + name;
    * every facet object carries its required ``_producer`` and ``_schemaURL`` keys.
    """
    event_type = event.get("eventType")
    if event_type not in _EVENT_TYPES:
        raise ConfigError(f"invalid OpenLineage eventType {event_type!r}")
    event_time = event.get("eventTime")
    if not isinstance(event_time, str):
        raise ConfigError("OpenLineage eventTime is required")
    try:
        parsed_time = datetime.fromisoformat(event_time)
    except ValueError as exc:
        raise ConfigError(f"OpenLineage eventTime is not ISO-8601: {event_time!r}") from exc
    if parsed_time.tzinfo is None:
        raise ConfigError(f"OpenLineage eventTime needs a timezone offset: {event_time!r}")
    for key in ("producer", "schemaURL"):
        if not event.get(key):
            raise ConfigError(f"OpenLineage event is missing {key!r}")
    run = event.get("run")
    if not isinstance(run, Mapping):
        raise ConfigError("OpenLineage event is missing 'run'")
    try:
        uuid.UUID(str(run.get("runId")))
    except ValueError as exc:
        raise ConfigError(
            f"OpenLineage run.runId must be a UUID, got {run.get('runId')!r}"
        ) from exc
    job = event.get("job")
    if not isinstance(job, Mapping) or not job.get("namespace") or not job.get("name"):
        raise ConfigError("OpenLineage job needs a non-empty namespace + name")
    for side in ("inputs", "outputs"):
        for dataset in event.get(side) or ():
            if not isinstance(dataset, Mapping):
                raise ConfigError(f"OpenLineage {side} entry must be an object")
            if not dataset.get("namespace") or not dataset.get("name"):
                raise ConfigError(f"OpenLineage {side} dataset needs a namespace + name")
            _validate_facets(dataset.get("facets"), where=f"{side} dataset")
    _validate_facets(run.get("facets"), where="run")
    _validate_facets(job.get("facets"), where="job")


def _validate_facets(facets: Any, *, where: str) -> None:
    if facets is None:
        return
    if not isinstance(facets, Mapping):
        raise ConfigError(f"OpenLineage {where} facets must be an object")
    for name, facet in facets.items():
        if not isinstance(facet, Mapping):
            raise ConfigError(f"OpenLineage {where} facet {name!r} must be an object")
        for required in ("_producer", "_schemaURL"):
            if not facet.get(required):
                raise ConfigError(f"OpenLineage {where} facet {name!r} is missing {required!r}")


class LineageTransport(Protocol):
    """Where a built OpenLineage event goes — the one seam the emitter writes through."""

    def send(self, event: Mapping[str, Any]) -> None: ...


class CollectingLineageTransport:
    """An in-memory transport that keeps every event — the test + local-inspection transport.

    It is deliberately **no laxer than a real backend**: each event is run through
    :func:`validate_run_event` (the OpenLineage JSON-Schema constraints) before it is recorded, so
    a spec-invalid event raises here instead of being silently collected — the fake never accepts
    more than the real thing does.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def send(self, event: Mapping[str, Any]) -> None:
        validate_run_event(event)
        self.events.append(dict(event))


_LINEAGE_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class _NoRedirectLineageHandler(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an ``HTTPError`` instead of a followed request.

    Returning ``None`` from ``redirect_request`` makes ``urllib`` raise for the ``3xx`` rather
    than re-issuing the request against the ``Location`` host. That matters because urllib's
    *default* opener replays the original headers — including ``Authorization: Bearer <api key>``
    — to whatever host the redirect names, so a compromised, misconfigured or MITM'd collector
    could harvest the hosted-catalog credential. Same defence, same reason as the SharePoint
    connector's ``_NoRedirectHandler``.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class HttpLineageTransport:
    """POSTs events to an OpenLineage HTTP endpoint (Marquez, Atlan, Collibra, Dataplex, …).

    Stdlib ``urllib`` only — the ``openlineage-python`` client is not a ``latence-core``
    dependency (ADR-0016). ``url`` is the collector root (``http://localhost:5000``); the standard
    ``/api/v1/lineage`` path is appended unless the URL already names a path. An optional
    ``api_key`` rides as a ``Bearer`` header. A non-2xx response or a transport error raises — the
    *emitter* is what swallows it, so this class stays honest about failure and remains reusable.

    **The API key never leaves the configured collector.** Redirects are refused rather than
    followed (urllib's default opener would replay the ``Authorization`` header to the redirect
    target, including a different host), and a key may only travel over ``https`` — plain ``http``
    stays available for a keyless local Marquez and for a loopback emulator. Because the emitter
    swallows every transport exception, a leak here would be silent, so it is prevented rather
    than reported.
    """

    def __init__(self, url: str, *, api_key: str | None = None, timeout: float = 5.0) -> None:
        if not url:
            raise ConfigError("OpenLineage endpoint URL must not be empty")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise ConfigError(f"OpenLineage endpoint must be http(s), got {url!r}")
        if (
            api_key
            and parsed.scheme == "http"
            and (parsed.hostname or "") not in _LINEAGE_LOOPBACK_HOSTS
        ):
            raise ConfigError(
                "Refusing to send the OpenLineage API key over plaintext http to "
                f"{parsed.hostname!r}: a bearer token may only travel over https (loopback http "
                "is allowed for a local collector). Use an https LATENCE_OPENLINEAGE_URL."
            )
        path = parsed.path.rstrip("/")
        self.url = url.rstrip("/") + ("" if path else "/api/v1/lineage")
        self._api_key = api_key
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirectLineageHandler)

    def send(self, event: Mapping[str, Any]) -> None:
        body = json.dumps(event, sort_keys=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(  # noqa: S310 - scheme is validated to http(s) above
            self.url, data=body, headers=headers, method="POST"
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                status = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as exc:
            # Includes the 3xx the no-redirect handler surfaces: a status the collector really
            # returned, reported as a typed error rather than a raw urllib traceback (ADR-0034).
            raise ProviderLineageError(f"lineage endpoint returned HTTP {exc.code}") from None
        if status >= 300:
            raise ProviderLineageError(f"lineage endpoint returned HTTP {status}")


class ProviderLineageError(RuntimeError):
    """A lineage endpoint rejected an event — always caught by the emitter, never fatal."""


@dataclass
class OpenLineageEmitter:
    """The configured lineage seam: a transport + the namespace this deployment's jobs live in.

    Long-lived and run-agnostic (the ``LocalRunner`` holds one). :meth:`for_run` mints the
    run-scoped :class:`RunLineage` the Runner actually calls — the same shape as
    :class:`~latence_core.tracing.RunTracer`.
    """

    transport: LineageTransport
    namespace: str = DEFAULT_LINEAGE_NAMESPACE

    def for_run(self, pipeline: str, run_id: str) -> RunLineage:
        return RunLineage(emitter=self, pipeline=pipeline, run_id=run_id)


@dataclass
class RunLineage:
    """The run-scoped emitter the Runner drives: one parent job + one job per Stage.

    Five calls — :meth:`run_start`, :meth:`stage_start`, :meth:`stage_end`, :meth:`run_end` — build
    OpenLineage ``RunEvent``\\ s and hand them to the transport. **Nothing here can fail a run**:
    every method is wrapped end-to-end (building *and* sending), so a malformed field, a dead
    endpoint, or a 500 is logged at WARNING and the pipeline continues.

    Each Stage is its own OpenLineage *job* (``<pipeline>.<stage>``) carrying a ``parent`` run
    facet pointing at the pipeline's run, which is how a catalog nests the Stages under one run and
    stitches the per-Stage datasets into a single graph.
    """

    emitter: OpenLineageEmitter
    pipeline: str
    run_id: str

    @property
    def namespace(self) -> str:
        return self.emitter.namespace

    @property
    def run_uuid(self) -> str:
        """The parent (pipeline-level) run UUID — deterministic, stable across a resume."""
        return lineage_run_uuid(self.namespace, self.pipeline, self.run_id)

    def stage_run_uuid(self, stage: str) -> str:
        return lineage_run_uuid(self.namespace, self.pipeline, self.run_id, stage)

    # --- the four calls the Runner makes -------------------------------------------------
    def run_start(self) -> None:
        self._emit(
            "START",
            job=self.pipeline,
            run_uuid=self.run_uuid,
            job_type="PIPELINE",
            facet={"latence_run_id": self.run_id, "pipeline": self.pipeline},
        )

    def run_end(self, *, documents: int | None = None, error_category: str | None = None) -> None:
        facet: dict[str, Any] = {"latence_run_id": self.run_id, "pipeline": self.pipeline}
        if documents is not None:
            facet["documents"] = documents
        if error_category is not None:
            facet["error_category"] = error_category
        self._emit(
            "FAIL" if error_category is not None else "COMPLETE",
            job=self.pipeline,
            run_uuid=self.run_uuid,
            job_type="PIPELINE",
            facet=facet,
        )

    def stage_start(
        self,
        stage: str,
        *,
        capability: str,
        provider: str,
        records_in: int,
        inputs: Sequence[LineageDataset] = (),
        outputs: Sequence[LineageDataset] = (),
    ) -> None:
        self._emit(
            "START",
            job=f"{self.pipeline}.{stage}",
            run_uuid=self.stage_run_uuid(stage),
            job_type="STAGE",
            parent=True,
            inputs=inputs,
            outputs=outputs,
            facet={
                "stage": stage,
                "capability": capability,
                "provider": provider,
                "records_in": records_in,
            },
        )

    def stage_end(
        self,
        stage: str,
        *,
        capability: str,
        provider: str,
        records_in: int,
        records_out: int,
        duration_ms: float,
        inputs: Sequence[LineageDataset] = (),
        outputs: Sequence[LineageDataset] = (),
        error_category: str | None = None,
        from_checkpoint: bool = False,
        skipped: bool = False,
    ) -> None:
        facet: dict[str, Any] = {
            "stage": stage,
            "capability": capability,
            "provider": provider,
            "records_in": records_in,
            "records_out": records_out,
            "duration_ms": duration_ms,
            "from_checkpoint": from_checkpoint,
            "skipped": skipped,
        }
        if error_category is not None:
            # The stable typed *category* only (ADR-0034). The standard ``errorMessage`` facet is
            # deliberately never emitted: an exception message can quote a document.
            facet["error_category"] = error_category
        self._emit(
            "FAIL" if error_category is not None else "COMPLETE",
            job=f"{self.pipeline}.{stage}",
            run_uuid=self.stage_run_uuid(stage),
            job_type="STAGE",
            parent=True,
            inputs=inputs,
            outputs=outputs,
            facet=facet,
        )

    # --- event construction ---------------------------------------------------------------
    def _emit(
        self,
        event_type: str,
        *,
        job: str,
        run_uuid: str,
        job_type: str,
        facet: Mapping[str, Any],
        parent: bool = False,
        inputs: Sequence[LineageDataset] = (),
        outputs: Sequence[LineageDataset] = (),
    ) -> None:
        try:
            event = self._build(
                event_type,
                job=job,
                run_uuid=run_uuid,
                job_type=job_type,
                facet=facet,
                parent=parent,
                inputs=inputs,
                outputs=outputs,
            )
            self.emitter.transport.send(event)
        except Exception:  # noqa: BLE001 - lineage is observability; it never fails a run
            logger.warning(
                "OpenLineage emission failed (%s %s); continuing", event_type, job, exc_info=True
            )

    def _build(
        self,
        event_type: str,
        *,
        job: str,
        run_uuid: str,
        job_type: str,
        facet: Mapping[str, Any],
        parent: bool,
        inputs: Sequence[LineageDataset],
        outputs: Sequence[LineageDataset],
    ) -> dict[str, Any]:
        run_facets: dict[str, Any] = {
            "latence_run": {
                "_producer": LINEAGE_PRODUCER,
                "_schemaURL": _LATENCE_FACET_URL,
                **dict(facet),
            }
        }
        if parent:
            run_facets["parent"] = {
                "_producer": LINEAGE_PRODUCER,
                "_schemaURL": _PARENT_FACET_URL,
                "run": {"runId": self.run_uuid},
                "job": {"namespace": self.namespace, "name": self.pipeline},
            }
        return {
            "eventType": event_type,
            "eventTime": datetime.now(UTC).isoformat(),
            "producer": LINEAGE_PRODUCER,
            "schemaURL": RUN_EVENT_SCHEMA_URL,
            "run": {"runId": run_uuid, "facets": run_facets},
            "job": {
                "namespace": self.namespace,
                "name": job,
                "facets": {
                    "jobType": {
                        "_producer": LINEAGE_PRODUCER,
                        "_schemaURL": _JOB_TYPE_FACET_URL,
                        "processingType": "BATCH",
                        "integration": "LATENCE",
                        "jobType": job_type,
                    }
                },
            },
            "inputs": [d.to_json() for d in inputs],
            "outputs": [d.to_json() for d in outputs],
        }


def build_lineage_emitter(
    url: str | None,
    *,
    namespace: str = DEFAULT_LINEAGE_NAMESPACE,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> OpenLineageEmitter | None:
    """Build the emitter for an endpoint — or ``None`` when no endpoint is configured.

    ``None``/empty ``url`` returns ``None``, which is what "off by default" means concretely: the
    Runner holds no emitter, builds no event, and opens no socket. A malformed URL is a
    :class:`~latence_core.errors.ConfigError` (an operator who *asked* for lineage and mistyped the
    endpoint should hear about it at wiring time, not silently get no lineage).
    """
    if not url:
        return None
    return OpenLineageEmitter(
        transport=HttpLineageTransport(url, api_key=api_key, timeout=timeout),
        namespace=namespace or DEFAULT_LINEAGE_NAMESPACE,
    )
