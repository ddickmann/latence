# Phase-2 Hardening Roadmap

Ordered plan for the remaining executable long-tail after Phase-1 + G1 (passed) + the top G1-finding hardening (#61 redaction recall, #62 screening, #63 streaming Runner) already merged.

**Ordering principle:** land the **core Runner/architecture changes first** (Waves 1–2), *then* the systematic per-stage adversarial/fuzz test-hardening (Wave 3). Writing the big fuzz suites against a still-changing Runner would mean rewriting them after each Runner change; doing Runner-stable-first is the difference between one clean pass and repeated churn. Second providers (Wave 4) come last because a seam is only worth a second implementation once it's battle-tested. Each item goes through the verify gate and merges only on green CI; core-touching items also get the dedicated determinism/resume/delta re-verify.

## Wave 1 — Runner foundation & architecture completion (serial; core-touching)
Do these first and one at a time — they change the execution model everything else tests against.
1. **#73 — H-core observability** *(in flight)* — typed error taxonomy + per-stage latency/quality metrics + tracing spans. The foundation the fuzz waves assert against (typed errors, metrics).
2. **#69 — stream the executor `outputs` dict** — completes ADR-0033: release doc-level records before the corpus-level accumulators run. Last doc-level memory transient. Re-verify determinism/resume/delta/Purge.
3. **#68 — bound relation `_find_candidates` O(n²) on the input side** — pairs with the scale/streaming theme; small, contained (relation provider).

## Wave 2 — Small correctness/security hardening (quick; may overlap Wave 3)
4. **#66 — extended secret recall** (redaction: `aws_secret_access_key`, bare JWT `token:`) — contained recall extension; add to the #61 recall benchmark.

## Wave 3 — Systematic per-stage adversarial/fuzz (the bulk; after the Runner is stable)
5. **#74 — doc-level fuzz** (Parse/Chunk/NER/RE): property-based + fuzz + edge-case matrix; assert offset/page-provenance/Evidence invariants. **∥ parallel with #75** (different stages).
6. **#75 — corpus-level fuzz + disambiguation precision** (Profiling/Disambiguation/Graph): adversarial inputs + a labeled entity-resolution precision/recall metric; FAISS-GPU blocking stays flagged (#64).
7. **#76 — quality + delta chaos** (gold-set harness property tests; randomized interleaved add/retract/purge; reconciliation-under-drift; WAL-recovery fuzz).

## Wave 4 — Additive (last)
8. **#77 — second CPU reference providers + parity/benchmarks** for the single-provider Capabilities (Chunk, Profiling, Disambiguation, Graph, Screening).

## Wave 5 — Publish (maintainer-manual)
9. **#72 — pre-publish polish** (README meta-package wording, `latence-core` license field, THIRD-PARTY version refresh) — sweep immediately before publish.
10. **Manual, outward-facing (Dennis triggers):** `gh repo edit … --visibility public`, then PyPI publish. Not automated.

## Parked — needs a real environment (not executable in a CPU/no-weights session)
- **#64 — GPU-provider quality baseline + live-swap parity, and the 100k/1M perf-rig numbers.** Validated at reduced scale in G1; headline numbers need a real GPU/perf rig. Do not fabricate.

## Independent
- Fork `task_afb7379a` (no-op `PYTHONHASHSEED` autouse fixture) — low-risk test fix; fold its PR into the merge order whenever it lands.
