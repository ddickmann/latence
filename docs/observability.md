# Observability: metrics, traces, and the error taxonomy

Latence ships observability in two layers:

1. **Dependency-free, always on** (ADR-0034): a typed **error taxonomy**, per-Stage **tracing
   spans**, and per-Stage **metrics** rolled into the persisted **Quality Report**. Nothing to
   install; silent unless you turn it on.
2. **An optional EXPORT seam** (ADR-0047): turn those same numbers into **Prometheus** time series
   or an **OpenTelemetry** meter, behind an optional dependency group so the core stays dep-light.

This page covers what is emitted and how to enable each layer.

## What is emitted

Everything is **counts only, never content** — a value is always a duration/count/tally and a label
is always a low-cardinality *name* (Stage / capability / provider / error-category / pipeline). A
document body, a PII surface, a secret, or an error *message* never enters a metric, a label, or a
span field. This is asserted adversarially in the test suite.

Source of truth: the `QualityReport` and its per-Stage `StageMetrics` (see
`packages/latence-core/src/latence_core/quality.py`). The export seam is a *view* of them — it never
recomputes a number.

| Metric | Type | Labels | From |
| --- | --- | --- | --- |
| `latence_run_duration_seconds` | histogram | `pipeline` | `QualityReport.total_duration_seconds` |
| `latence_run_documents_total` | counter | `pipeline` | `QualityReport.document_count` |
| `latence_stage_duration_seconds` | histogram | `stage, capability, provider` | `StageMetrics.duration_seconds` |
| `latence_stage_records_in_total` | counter | `stage, capability, provider` | `StageMetrics.records_in` |
| `latence_stage_records_out_total` | counter | `stage, capability, provider` | `StageMetrics.records_out` |
| `latence_stage_endpoint_retries_total` | counter | `stage, capability, provider` | `StageMetrics.endpoint_retries` |
| `latence_stage_errors_total` | counter | `stage, capability, provider, category` | `StageMetrics.error_count` + `error_category` |
| `latence_stage_induction_empty_documents` | gauge | `stage, capability, provider` | `StageMetrics.induction_empty_after_sanitization` |
| `latence_er_merges_applied_total` | counter | `pipeline` | `DisambiguationQuality.merges_applied` |
| `latence_er_merges_below_policy_total` | counter | `pipeline` | `DisambiguationQuality.merges_below_policy` |

The **error taxonomy** `category` label is the stable tag `error_category()` produces:
`contract` (mis-wired DAG / carrier mismatch), `config` (bad operator config), `provider` (a
Provider/endpoint failed at runtime), `storage` (a Storage/Corpus-Version inconsistency),
`screening` (a fatal-Screening quarantine), or `unknown` (a stray builtin). The
**induction-empty** gauge is the loud "the corpus induced no usable domain schema and the run
degraded to the bare config-label floor" signal (W4). The **ER merge audit** surfaces the
no-silent-over-merge posture (merges applied vs recorded-but-below-policy).

## Layer 1 — traces + the Quality Report (no install)

The Quality Report is written on **every** run (`.../_latence/runs/<run_id>/quality-report.json`
and `.md`). It carries the per-Stage `StageMetrics` table above and, on a Stage failure, a FAIL
report (the completed Stages plus the failed one, `error_count=1` + the category) before the
exception re-raises. Read it directly, or point any JSON pipeline at it.

**Tracing spans** (`latence_core.tracing`) emit a structured event per Stage `begin`/`end` (and per
checkpoint `batch`), stamped with the `run_id` as the correlation id. They are silent by default —
the `LoggingSpanSink` logs at DEBUG on the `latence.trace` logger. Turn them on:

```python
import logging
logging.getLogger("latence.trace").setLevel(logging.DEBUG)
```

Or collect them in-process:

```python
from latence_core import CollectingSpanSink, LocalRunner
sink = CollectingSpanSink()
report = LocalRunner(span_sink=sink).run(pipeline, run_id="run-0001")
for span in sink.spans:
    print(span.as_dict())   # counts only — stage, capability, provider, records, duration, category
```

## Layer 2 — Prometheus / OpenTelemetry export (optional extra)

Install the optional group (isolated from the core; a default install pulls neither):

```bash
pip install 'latence-core[observability]'   # prometheus-client + opentelemetry-api/sdk
```

### From the CLI (env-toggle, off by default)

```bash
export LATENCE_METRICS=prometheus                          # or: otel  (unset/none = no-op default)
export LATENCE_METRICS_TEXTFILE=/data/out/metrics/latence.prom   # Prometheus batch bridge (see below)
latence run stack.yaml --run-id run-0001
```

With `LATENCE_METRICS` unset (or `none`) nothing changes and no optional package is imported. If the
extra is **not installed**, the seam logs a warning and runs as a no-op — it never breaks a run.

### From code

```python
from latence_core import build_metric_sink, record_quality_report, LocalRunner

report = LocalRunner().run(pipeline, run_id="run-0001")

sink = build_metric_sink("prometheus")   # degrades to NoOpMetricSink if the extra is absent
record_quality_report(report, sink)      # walks StageMetrics + the ER audit, emits the metrics
```

For **live** per-Stage metrics (emitted as each Stage finishes, not only at end-of-run), pass a
`MetricsSpanSink` — it rides the existing tracing seam:

```python
from latence_core import MetricsSpanSink, LoggingSpanSink, build_metric_sink, LocalRunner

sink = build_metric_sink("otel")
runner = LocalRunner(span_sink=MetricsSpanSink(sink, chain=LoggingSpanSink()))  # metrics + logs
runner.run(pipeline, run_id="run-0001")
```

### The batch-job caveat (Prometheus)

A Latence pipeline run is a **batch Job**, not a long-lived server — after it exits there is no
`/metrics` endpoint to scrape. Two supported bridges:

* **Textfile collector** (simplest on-prem): set `LATENCE_METRICS_TEXTFILE` (CLI) or call
  `PrometheusMetricSink.write_textfile(path)`. Point the Prometheus **node-exporter** textfile
  collector at that directory:

  ```yaml
  # node-exporter args
  --collector.textfile.directory=/var/lib/node_exporter/textfile_collector
  ```

  In the Helm chart, `metrics.kind=prometheus` writes to `metrics.textfilePath` on the data PVC.

* **Pushgateway** (multi-node / ephemeral pods):

  ```python
  from prometheus_client import push_to_gateway
  sink = build_metric_sink("prometheus")            # a PrometheusMetricSink
  record_quality_report(report, sink)
  push_to_gateway("pushgateway:9091", job="latence", registry=sink.registry)
  ```

A **long-running** adopter (e.g. an embedding server built on the core) can instead scrape a shared
registry directly — the pull model works there.

### OpenTelemetry collector

The `OpenTelemetryMetricSink` records into an OTel `Meter`; **you** configure the SDK's
`MeterProvider`/exporter at the edge (the core carries no exporter/transport choice). A minimal OTLP
setup:

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint="http://otel-collector:4317"))
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
# now build_metric_sink("otel") records into that provider
```

A matching collector config (receive OTLP, export to Prometheus):

```yaml
receivers:
  otlp:
    protocols:
      grpc: {endpoint: 0.0.0.0:4317}
exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
```

## A sample Prometheus scrape + alert

```yaml
# prometheus.yml (textfile collector via node-exporter)
scrape_configs:
  - job_name: node
    static_configs:
      - targets: ["node-exporter:9100"]
```

```yaml
# a rule: alert when any Stage is failing
groups:
  - name: latence
    rules:
      - alert: LatenceStageErrors
        expr: increase(latence_stage_errors_total[1h]) > 0
        annotations:
          summary: "Latence stage {{ $labels.stage }} failing ({{ $labels.category }})"
      - alert: LatenceInductionDegraded
        expr: latence_stage_induction_empty_documents > 0
        annotations:
          summary: "Schema induction degraded to the config-label floor"
```
