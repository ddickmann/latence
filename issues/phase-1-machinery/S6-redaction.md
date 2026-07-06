# S6 — Redaction (PII)

**Parent**: S4 · **Phase 1 · Baseline bar**

## What to build
A **Redaction** Capability that detects PII and produces masked/replaced variants of the text, with a CPU-viable hybrid rule+model reference Provider. Redaction is metadata-aware: the sensitivity attribute of `Classification` informs policy, and redacted spans are recorded with Provenance so the action is auditable.

## Ports (algorithmic assets)
- None to port directly; reference Provider may reuse the mt5/GLiNER-PII approach via plain `transformers` (CPU), license verified.

## Acceptance criteria
- [ ] PIIDetector/Redaction Capability protocol defined; ≥1 hybrid rule+model reference Provider.
- [ ] Produces both a masked text variant and a structured record of detected PII spans (type, offset, action) with Provenance.
- [ ] Redaction policy is configurable per Pipeline and can key off `Classification.sensitivity`.
- [ ] PII detection counts by type land in the Quality Report; no raw PII is written to logs.
- [ ] Baseline bar met.

## Blocked by
S4. (Parallelizable with S5.)
