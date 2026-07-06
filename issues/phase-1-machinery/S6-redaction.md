# S6 — Redaction (PII)

**Parent**: S4 · **Phase 1 · Baseline bar**

## What to build
A **Redaction** Capability that detects PII and produces masked/replaced variants of the text, with a CPU-viable hybrid rule+model reference Provider. Redaction is metadata-aware: the sensitivity attribute of `Classification` informs policy, and redacted spans are recorded with Provenance so the action is auditable.

## Ports (algorithmic assets)
- None to port directly; reference Provider may reuse the mt5/GLiNER-PII approach via plain `transformers` (CPU), license verified.

## Acceptance criteria
- [x] PIIDetector/Redaction Capability protocol defined; ≥1 hybrid rule+model reference Provider.
- [x] Produces both a masked text variant and a structured record of detected PII spans (type, offset, action) with Provenance.
- [x] Redaction policy is configurable per Pipeline and can key off `Classification.sensitivity`.
- [x] PII detection counts by type land in the Quality Report; no raw PII is written to logs.
- [x] Baseline bar met.

## Blocked by
S4. (Parallelizable with S5.)

## Delivered
- **Contracts** (`latence-core`, `SCHEMA_VERSION` → 6): `RedactionRecord` (one
  per document: masked/replaced text variant + ordered non-overlapping `PIISpan`
  list), `PIISpan` (type / document-offset / action / placeholder / resolved page —
  **no raw value field**), `RedactionAction` (mask/replace/hash/tag).
- **Capability**: `PIIDetector` protocol + `CapabilityKind.REDACTION`, wired into
  the `LocalRunner` (execute / checkpoint / export routing) and the Quality Report
  (`RedactionQuality`: per-type + per-action counts, documents-with-PII — counts
  only, never raw PII).
- **Providers**: in-core `redaction.hybrid_rule` (deterministic hybrid of a
  well-known-PII regex library + config gazetteer + custom patterns; the rule half)
  and the separately-packaged learned `redaction.gliner_pii` (`latence-pii-gliner`,
  GLiNER-PII, Apache-2.0 weights+code, the model half) behind the same seam.
- **Shared machinery** (`latence_core.redaction_policy`): sensitivity-keyed
  `RedactionPolicy`, `placeholder_for`, `apply_masking`, span/text reconstruction —
  framework-owned so both Providers share one implementation and cannot drift.
- **Docs**: ADR-0024; `examples/redaction-slice.yaml`; `latence-pii-gliner/README.md`.
- **Tests**: contract + in-core provider (mask/replace/hash/tag, sensitivity policy,
  gazetteer/patterns, page round-trip, determinism, no-raw-PII, config validation) +
  gliner-pii provider (faked `gliner`) + end-to-end through the Runner (resume +
  determinism + no-raw-PII-in-any-artifact). Local: ruff clean, mypy --strict clean,
  262 passed / 1 skipped.
