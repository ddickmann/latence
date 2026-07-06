# Definition of Done — three tiers

The anti-"brittle toy repo" gate. Every issue declares which bar it must meet.

## Baseline bar (Phase-1 slices S1–S13)

Prove the machinery at enterprise level, without yet being perfect.

- [ ] **Typed**: strict type-checking passes (mypy/pyright strict) on the touched packages.
- [ ] **Contracts**: inter-Stage data is a versioned Pydantic model; `Provenance` and `Classification` present and validated at every boundary the slice touches; schema-version bumped on any contract change.
- [ ] **Tested**: unit tests for happy path **and** the key failure modes; ≥1 end-to-end test through the local Runner; deterministic (seeded, hash-stable).
- [ ] **CI green**: lint + type + test + build pass in CI for the slice's packages.
- [ ] **Packaged**: correct package boundary; heavy/optional deps isolated so `pip install latence-core` stays lean (ADR-0016).
- [ ] **Licensed**: any new model has weights+code license verified and recorded (ADR-0012).
- [ ] **Secure-by-default**: input validation on external data; no secrets in code/logs; Screening respected; no unsafe deserialization.
- [ ] **Observable**: structured logs + basic run/stage metrics emitted into the Quality Report.
- [ ] **Documented**: minimal usage doc for the Capability/CLI; ADR added if a hard, surprising trade-off was made.

## Gate bar (G1 — profound testing, hard stop)

All must be green before ANY Phase-2 issue starts.

- [ ] **E2E on real messy corpora** (≥3, mixed types incl. deliberately dangerous files): Source→…→Export yields KG + RAG corpus + Quality Report with every contract satisfied.
- [ ] **Security review**: documented threat model (malicious files, prompt injection, PII leakage, secret handling, SSRF via Source URIs, dependency CVE scan) with findings closed or explicitly accepted.
- [ ] **Reliability**: crash-mid-run resume proven; re-running a Delta is idempotent; add/update/delete/Purge correctness proven on a fixture corpus.
- [ ] **Performance baseline** recorded (throughput + peak memory at 10k and 100k docs) with documented limits and backpressure behavior.
- [ ] **Quality baseline**: gold-set precision/recall per Stage recorded as the number Phase-2 must beat.
- [ ] **Provider-swap** proven for every Capability that has ≥2 providers (incl. one cloud/endpoint where available).
- [ ] **Determinism**: identical input + config ⇒ identical Corpus Version fingerprint.
- [ ] **Zero known Sev-1/Sev-2 defects**; coverage threshold met on core.

## Hardening bar (Phase-2 H-core, H2–H13)

Take one stage to perfection.

- [ ] **Adversarial + property-based + fuzz** tests; documented edge-case matrix closed.
- [ ] **Scale**: streaming/memory-bounded/backpressure validated at target scale; no OOM at the documented ceiling.
- [ ] **Observability**: tracing spans, per-stage latency/quality metrics, typed error taxonomy.
- [ ] **Providers**: ≥2 reference providers incl. a cloud/endpoint variant with parity tests.
- [ ] **Benchmarks**: any quality/perf claim backed by a reproducible benchmark vs a named baseline.
- [ ] **Docs**: capability reference + tuning guide + troubleshooting.
- [ ] **Security**: stage-specific abuse cases enumerated and closed.
