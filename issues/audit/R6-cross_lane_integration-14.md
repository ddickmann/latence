# R6-cross_lane_integration-14 — Chained Redactors double-count in the Quality Report, and can name a PII type the corpus never carries

**Severity:** major
**Status:** FIXED
**Site:** `packages/latence-core/src/latence_core/quality_report_builder.py:816-877`
(`QualityReportBuilder._redaction_quality`)
**Triggered by:** the chained topology committed in
`deploy/env/gpu-learned/stack.yaml:91-114` and `deploy/env/full-stack/stack.yaml:80-102`
(`redact_financial.depends_on: [redact_pii]`, commits `bce9b44` / `895b95f` — audit R5-12).

## What is wrong

`_redaction_quality` is handed the FLAT concatenation of every Redaction Stage's output
(`_records_of_kind` walks `ordered` and `extend`s), and folds it directly.

Since R5-12 chained `redact_pii -> redact_financial` in both shipped GPU deploy stacks, the SAME
chunk appears twice in that list:

1. as `redact_pii` emitted it, and
2. as `redact_financial` re-emitted it after `_compose_prior_spans` (`redaction_policy.py:454`)
   unioned the upstream spans into its own.

Every upstream span is therefore tallied twice, and the report is the operator's redaction audit
trail (ADR-0034, counts-only observability). The error scales with CHAIN LENGTH, not with what
actually reached the exported RAG corpus.

## Failure scenario (observed)

Probed on the exact shipped shape — two `redaction.hybrid_rule` Stages chained A -> B with
overlapping detector sets, the corpus Export hanging off B, one document carrying an email, a phone
number and the synthetic card `4111111111111111`:

```
REPORT: documents_redacted=1 documents_with_pii=1 pii_total=5
        per_type={'credit_card': 1, 'email': 2, 'phone': 2} per_action={'mask': 5}
EXPORT: 3 pii_spans, types ['email', 'phone', 'phone']
```

Two independent errors:

**(a) Double count.** `email: 2` — Stage A emitted the span and Stage B re-emitted the identical
span via `_compose_prior_spans`; both Stages' `ChunkRecord`s are in `records`. On the shipped
`gpu-learned` / `full-stack` stacks every person / email / phone / address span the learned
`redaction.gliner_pii` finds is double-counted this way in the operator-facing PII audit.

**(b) Phantom type.** `credit_card: 1` names a redaction NO exported span records. Stage A's
ADR-0044 financial-floor `credit_card` span for the card was superseded in Stage B by an
identical-extent `phone` span (the provider wins the tie in `_merge_floor` /
`_compose_prior_spans`), so the card is labelled `phone` in the export while the report still
certifies a `credit_card` redaction.

`packages/latence-core/tests/test_redaction_composition.py` — the R5 regression test for this exact
topology — asserts only that masks survive in the text and that
`{sp.pii_type for sp in memo.pii_spans} == {'email', 'iban'}`. It never inspects
`RedactionQuality`, so the whole suite stayed green.

## What to build

Collapse each record to its LAST version before rolling up — the downstream-most Stage that emitted
that `record_id`, which is the version that flows on to the Export. Composition is
coverage-preserving and unions the upstream spans in, so that last version already carries every
mask the chain applied.

Insertion order (hence determinism) must be preserved, and a single-Redactor Pipeline has no
duplicate `record_id`s, so its counts stay byte-identical (OFF-BY-DEFAULT invariant).

## Acceptance criteria

- [x] On the chained shipped shape, `RedactionQuality.pii_total` / `per_type` / `per_action` equal
      the spans actually present on the EXPORTED corpus records.
- [x] No `pii_type` is reported that no exported span carries (no phantom `credit_card`).
- [x] `documents_redacted` / `documents_with_pii` / `redaction_disabled_for_sensitive` unchanged.
- [x] Single-Redactor Pipelines produce byte-identical counts (off-by-default invariant).
- [x] The deprecated doc-level `RedactionRecord` path is deduped the same way.
- [x] Deterministic: rollup order is insertion-stable, no `set` iteration in the tally.
- [x] Non-vacuous regression test that fails pre-fix
      (`test_chained_redactors_do_not_double_count_in_the_quality_report`).
- [x] No PII value crosses into the report (unchanged — `PIISpan` never stores the surface).
- [x] Fast suite does not regress.
