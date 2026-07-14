# A metrics EXPORT seam (Prometheus / OpenTelemetry) layered on ADR-0034, behind an optional extra

**Status: accepted** — oss-deploy-observability. Adds `latence_core.observability`: a thin
metrics-export seam over the **existing** Quality Report (ADR-0008) and tracing/typed-error
foundation (ADR-0034), so an on-prem / VPC operator can scrape **Prometheus** counters/histograms
or feed an **OpenTelemetry** meter — *without* pulling any exporter dependency into the near-zero-dep
core (ADR-0016). Builds on ADR-0008 (Quality Report as a first-class artifact), ADR-0034 (typed error
taxonomy + the thin, OTel-free tracing seam), ADR-0016 (thin core, heavy/optional deps isolated), and
ADR-0004/0007 (the provider-optional pattern this mirrors). It does **not** amend ADR-0034's decision
to keep OTel out of core; it *extends* the same seam-at-the-edge posture with an opt-in export layer.

## Context

ADR-0034 deliberately stopped at a **dependency-free** observability foundation: a typed error
taxonomy (`error_category()` → a stable `"contract"|"config"|"provider"|"storage"|"screening"|
"unknown"` tag), a per-Stage `Span` seam (`RunTracer` → a pluggable `SpanSink`, silent by default),
and per-Stage `StageMetrics` (durations, record counts, `error_count`/`error_category`,
`induction_empty_after_sanitization`, `endpoint_retries`) rolled into the persisted `QualityReport`.
Its rejected alternative was explicit: **no `opentelemetry-*` in core** — that would break the
CPU-first, offline, thin-core promise and force a heavyweight transitive tree on every laptop
install; an adopter who wants a real tracer "fans the spans into OTel at the edge."

What ADR-0034 left unbuilt is that *edge*. An enterprise deployment (deploy/, the Helm chart) wants
the per-Stage timings/counts/error-taxonomy as **Prometheus** time series or **OTel** metrics an
existing collector already scrapes — the numbers `StageMetrics`/`QualityReport` already hold, in the
wire format ops tooling speaks. The open question was where to put the exporter so the core stays
thin and offline by default.

## Decision

**A three-primitive `MetricSink` seam + one dep-free translation function, with the backends behind
an optional extra.**

1. **`MetricSink` Protocol — counter / histogram / gauge, nothing else.** The whole export surface
   is three methods (`incr_counter`, `observe`, `set_gauge`), each taking a metric name, a numeric
   value, and low-cardinality **name** labels. `record_quality_report(report, sink)` is the single,
   dependency-free function that walks an existing `QualityReport` and its `StageMetrics` and calls
   those primitives — per-Stage duration histogram, records-in/out + endpoint-retry counters, the
   error counter labelled with the typed `error_category` (only on a failed Stage), the
   induction-empty gauge (only on a degraded induction Stage), the run total-duration histogram +
   document counter, and the ER merge audit (`merges_applied` vs `merges_below_policy`). It **reads**
   the report the Runner already built — it never recomputes a number and never duplicates the
   `StageMetrics`/`QualityReport` schema.

2. **The backends lazily import their optional package.** `PrometheusMetricSink` imports
   `prometheus_client`; `OpenTelemetryMetricSink` imports `opentelemetry` — both **at construction,
   inside the method**, exactly the provider-optional pattern (ADR-0016). Neither is a `latence-core`
   runtime dependency: they live in a new `[project.optional-dependencies] observability` group.
   Absent the package, a backend raises `MissingObservabilityExtra`, and the factory
   `build_metric_sink(kind)` **degrades to a `NoOpMetricSink`** (a warning, never a crash) — so a
   default install carries no `opentelemetry-*`/`prometheus-client` tree and a run that never opts in
   behaves byte-identically. `strict=True` flips the degrade to a raise for a deploy that *means* to
   fail if its observability stack is missing.

3. **Two entry points, one metric vocabulary.** `record_quality_report` is the end-of-run,
   report-driven path (the richest source — it sees the merge audit and induction flags). For **live**
   per-Stage metrics, `MetricsSpanSink` is a `SpanSink` (ADR-0034's existing seam) that translates a
   `stage_end` span into the *same* `MetricSink` primitives — so `LocalRunner(span_sink=
   MetricsSpanSink(sink))` emits metrics as each Stage finishes, with no Runner change and no metric
   logic duplicated. The CLI wires the report path behind an env toggle: `LATENCE_METRICS=
   prometheus|otel` (unset/`none` = the no-op default), with `LATENCE_METRICS_TEXTFILE` for the
   Prometheus node-exporter textfile-collector bridge a **batch** Job needs (a pull model has nothing
   to scrape after the pod exits; Pushgateway is the alternative, documented).

4. **Counts only, never content — inherited, asserted.** Every emitted value is a duration/count/
   tally and every label is a Stage/capability/provider/category/pipeline *name*. A document body, a
   PII surface, a secret, or even an error *message* never enters a metric or a label. This is the
   same no-PII discipline the Quality Report and the tracing seam keep (ADR-0034), re-asserted
   adversarially in the seam's test (a PII sentinel appears in no emitted label; the error message is
   never a label value).

## Consequences

* **The core stays thin and offline by default (ADR-0016 intact).** `latence-core`'s runtime deps are
  unchanged; the export tree is opt-in via `pip install 'latence-core[observability]'`. The
  anti-false-green test exercises the **real** absence (the extra is not in the CPU-first dev env), so
  the no-op-when-absent degrade is proven, not mocked.
* **No schema duplication.** The exporter is a *view* of `StageMetrics`/`QualityReport`; adding a
  field to the report and to the exporter is the only change needed to expose a new metric — the
  numbers have one home.
* **Batch-job caveat, stated honestly.** Prometheus is pull-based; a finite pipeline Job has no live
  `/metrics` endpoint to scrape. The seam therefore ships the textfile-collector + Pushgateway
  bridges rather than pretending a Job is a server. A long-running adopter (e.g. an embedding server)
  can still scrape a shared registry directly.
* **Rejected:** (a) an OTel/Prometheus dependency in core — the same rejection as ADR-0034, for the
  same reasons; (b) a global metrics singleton — hidden state, un-testable, no clean per-run
  correlation (the `MetricSink` is passed explicitly, mirroring `SpanSink`); (c) recomputing metrics
  inside the exporter — would fork the source of truth from `StageMetrics`.
