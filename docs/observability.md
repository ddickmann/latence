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

## Layer 3 — OpenLineage run events (data lineage in your catalog)

Point latence at any **OpenLineage** endpoint and a run shows up in the catalog your platform team
already governs with — **Marquez**, **Atlan**, **Collibra**, **Dataplex**, or anything else that
speaks the standard. One emitter, every consumer; nothing to build per catalog.

Off by default, exactly like the metrics seam: with no endpoint configured no event is built and no
socket is opened, and the run's Quality Report is identical either way.

```bash
# The env-toggle (mirrors LATENCE_METRICS)
export LATENCE_OPENLINEAGE_URL=http://localhost:5000        # a local Marquez
export LATENCE_OPENLINEAGE_NAMESPACE=my-platform            # optional; default "latence"
export LATENCE_OPENLINEAGE_API_KEY=...                      # optional; sent as a Bearer token
latence run stacks/default.yaml
```

```python
# Or from code
from latence_core import LocalRunner, build_lineage_emitter

runner = LocalRunner(lineage=build_lineage_emitter("http://localhost:5000", namespace="my-platform"))
report = runner.run(pipeline, run_id="run-0001")
```

`latence-core` speaks the wire format with `json` + `urllib` — the `openlineage-python` client is
**not** a dependency (ADR-0016).

### What a run looks like in the graph

| latence concept | OpenLineage vocabulary |
| --- | --- |
| Pipeline (`Pipeline.name`) | a parent **Job** `<pipeline>`, `jobType` facet `PIPELINE`/`BATCH` |
| Stage (`Stage.name`) | a **Job** `<pipeline>.<stage>`, `jobType` facet `STAGE`/`BATCH` |
| Run (`run_id`, an operator string) | **Run** `runId` = `uuid5(namespace/pipeline/run_id[/stage])` — the spec requires a UUID; the hash is deterministic, so a **resume continues the same run** instead of forking a new one |
| the Stage DAG (`depends_on`) | each Stage Job's `inputs` = its parents' checkpoint **Datasets**; a `parent` run facet nests every Stage under the pipeline run |
| a Stage checkpoint (`_latence/runs/<run_id>/checkpoints/<stage>.jsonl`) | the Stage's output **Dataset** (`file` + path, or `s3://bucket` + key) |
| the Source's `path`/`uri` config | the entry-point input **Dataset** (no schema — the raw corpus has no carrier contract yet) |
| the Stage's carrier contract (`ChunkRecord`, `EntityMention`, …) | the output Dataset's **`schema` facet** — the contract's field *names* + coarse types, read from the model, never sampled from data |
| `StageMetrics` counts + duration | the custom `latence_run` run facet: `records_in`, `records_out`, `duration_ms`, `from_checkpoint`, `skipped` |
| the typed error taxonomy (`error_category()`) | a **FAIL** event whose `latence_run` facet carries `error_category` |
| `Provenance` (per record) | **not** emitted — see below |

### Counts only, and why `errorMessage` is absent

The same ADR-0034 discipline as the metrics/tracing seams, applied to lineage: a facet carries
counts, durations, names, dataset URIs, and schema field names — never a record body, a PII surface,
a secret, or an **exception message**. That last one is deliberate and load-bearing: the standard
`errorMessage` run facet (and its `stackTrace`) would ship `str(exc)` to the catalog, and an
exception message routinely quotes the record that broke. latence emits the stable typed
**category** instead (`provider`, `contract`, `config`, `storage`, `screening`, `unknown`).

For the same reason per-record `Provenance` is aggregated away rather than emitted: lineage is about
*datasets*, and per-document provenance stays in the Quality Report / the exported corpus, which are
governed by the redaction posture. The dataset-level identity a catalog needs — where the run read
from and wrote to — is exactly what the input/output Datasets carry.

An emission failure **never fails a run**: a dead endpoint, a 500, or a serialisation bug is logged
at WARNING on the `latence.observability` logger and the pipeline continues.

### Verify it against a real Marquez

`scripts/verify-openlineage-marquez.py` is the manual end-to-end gate (kept out of
`verify-local.sh`, which is offline and dependency-free — Marquez needs Docker):

```bash
docker network create marquez-net
docker run -d --name marquez-db --network marquez-net \
    -e POSTGRES_USER=marquez -e POSTGRES_PASSWORD=marquez -e POSTGRES_DB=marquez postgres:14
docker run -d --name marquez-api --network marquez-net -p 5050:5000 -p 5051:5001 \
    -e POSTGRES_HOST=marquez-db -e POSTGRES_PORT=5432 -e POSTGRES_USER=marquez \
    -e POSTGRES_PASSWORD=marquez -e POSTGRES_DB=marquez marquezproject/marquez:0.51.1

uv run python scripts/verify-openlineage-marquez.py http://localhost:5050
```

It runs a real four-Stage pipeline and then asserts, through Marquez's own API, that the pipeline
Job, every Stage Job, the checkpoint Datasets (with their schema fields), and a `COMPLETED` run
state are all in the lineage graph.

> Offline, `CollectingLineageTransport` validates every event against the OpenLineage 2-0-2 JSON
> Schema constraints (`validate_run_event`) so the offline fake is never *laxer* than a real
> consumer. It is in fact slightly stricter: Marquez 0.51.1's `POST /api/v1/lineage` was measured to
> accept a non-UUID `runId` that the spec forbids and a schema-validating consumer rejects.
