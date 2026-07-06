# Phase 2 — Hardening loop

After **G1 is green**, loop over each stage and harden it to perfection against the **Hardening bar** (`../DEFINITION-OF-DONE.md`). Each stage gets a **dedicated issue** (H-core, H2–H13). Order is by risk/impact, not strict linear; each Hn is blocked only by G1 (and H-core for shared machinery).

## Per-issue Hardening template
Every Hn issue must deliver, for its stage:
- **Adversarial + property-based + fuzz tests**; a documented edge-case matrix, all closed.
- **Scale**: streaming/memory-bounded/backpressure validated at the target ceiling recorded in G1; no OOM.
- **Observability**: tracing spans, per-stage latency + quality metrics, a typed error taxonomy.
- **Providers**: ≥2 reference providers incl. a cloud/endpoint variant with parity tests; document licenses.
- **Benchmarks**: any quality/perf claim backed by a reproducible benchmark vs a named baseline (beat the G1 number).
- **Docs**: capability reference + tuning guide + troubleshooting.
- **Security**: stage-specific abuse cases enumerated and closed.

## The issues
- **H-core** — contracts, Capability registry, local Runner, Storage seam, Quality Report plumbing.
- **H2** parse · **H3** chunk+screening · **H4** entity-extraction · **H5** relation-extraction · **H6** redaction · **H7** profiling · **H8** disambiguation · **H9** graph-assembly+export · **H10** quality-report · **H11** deltas · **H12** performance/runners/scale · **H13** demo+docs.

Each Hn names its stage-specific hardening targets below (create the individual files as you enter Phase 2, seeded from G1-RESULTS.md so the perf/quality baselines are concrete).

### Stage-specific hardening focus (seed notes)
- **H3 screening** — expand injection/harmful/sensitivity providers; red-team corpus; false-positive/negative rates vs baseline.
- **H8 disambiguation** — FAISS-GPU blocking at scale; over-merge/under-merge precision on a labeled set; multilingual entity linking parity.
- **H11 deltas** — chaos testing (kill during commit/reconcile); Purge completeness audit; drift-threshold tuning; large-corpus reconcile time.
- **H12 scale** — 1M-doc run; Databricks + Airflow parity; endpoint failover; cost/throughput curves.
