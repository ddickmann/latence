# G1 — Profound testing gate (HARD STOP)

**Parent**: S1–S13 · **Gate · Gate bar**

## What this is
A dedicated validation milestone between Phase 1 (machinery proven) and Phase 2 (hardening). **No Phase-2 issue starts until every box below is green.** The purpose is to prove the whole machinery works at enterprise-level quality, security, and reliability before investing in per-stage perfection — and to record the baselines Phase-2 must beat.

## Exit criteria (all required)
- [ ] **End-to-end on ≥3 real messy corpora** (mixed file types, scans, office docs, and deliberately dangerous files): Source→…→Export yields KG + RAG corpus + Quality Report with every contract satisfied and no crash.
- [ ] **Security review**: written threat model covering malicious files, prompt injection, PII leakage, secret handling, SSRF via Source URIs, and a dependency CVE scan; every finding closed or explicitly risk-accepted with rationale.
- [ ] **Reliability**: crash-mid-run resumes correctly; re-running a Delta is idempotent; add/update/Retract/Purge correctness proven on a fixture corpus with assertions on what survives.
- [ ] **Performance baseline**: throughput and peak memory recorded at 10k and 100k docs; documented ceiling and backpressure behavior; no OOM at the stated ceiling.
- [ ] **Quality baseline**: gold-set precision/recall/F1 per Stage recorded — the numbers Phase-2 must improve.
- [ ] **Provider-swap**: proven for every Capability with ≥2 providers, including one cloud/endpoint variant; outputs consistent modulo model.
- [ ] **Determinism**: identical input + config ⇒ identical Corpus Version fingerprint, verified across two clean runs.
- [ ] **License audit**: every shipped default Provider's weights+code verified permissive; any restricted model is opt-in with flow-down surfaced (ADR-0012).
- [ ] **Defects**: zero known Sev-1/Sev-2; core coverage threshold met.

## Output
A signed-off `gate/G1-RESULTS.md` capturing every baseline number and the security/threat-model findings. Phase 2 references these baselines.

## Blocked by
All of S1–S13.
