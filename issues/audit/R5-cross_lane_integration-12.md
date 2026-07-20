# R5-cross_lane_integration-12 — `redact_pii` is wired-but-dead in both GPU deploy stacks

**Severity:** major
**Status:** FIXED
**Scope:** `git diff main...feat/release-fleet-integration` (tickets #178–#199)

## What was wrong

`deploy/env/gpu-learned/stack.yaml` and `deploy/env/full-stack/stack.yaml` both declared their two
learned Redactors as parallel SIBLINGS off `content_screen`:

```yaml
  - name: redact_pii          # redaction.gliner_pii
    depends_on: [content_screen]
  - name: redact_financial    # redaction.gliner2 (ADR-0044 financial floor)
    depends_on: [content_screen]
  - name: export_corpus       # export.jsonl_parquet
    depends_on: [redact_financial]
```

The Runner threads only a Stage's declared parents' output into an Export
(`LocalRunner._gather_inputs`), so a parallel redaction branch is discarded wholesale.
`redact_pii` — `pii_labels: [person, email, phone number, address, credit card number]` — executed
on the GPU, was certified by the #198 placement probe, and contributed its Stage metrics to the
Quality Report, while **none of its masks reached `records.parquet` / `records.jsonl`**. The
ADR-0044 universal financial floor still fired via the branch that IS exported, which is exactly
why every gate read green over a corpus that retained names, emails, phone numbers and addresses.
This is the #199 full-stack verification's own deliverable corpus.

A second, deeper half: **merely chaining the two Stages would not have fixed it.**
`finalize_redacted_chunk` masks from the chunk's clean `content` and OVERWRITES
`masked_content` / `pii_spans`, so a downstream Redactor clobbered every mask its predecessor had
applied. Redaction did not compose.

## Failure scenario (proven)

A stack with two `redaction.hybrid_rule` Stages with DISJOINT detectors — `redact_a`
(`detectors: [email]`) and `redact_b` (`detectors: [iban]`) — with `export_corpus` downstream of
`redact_b` only. Exported `corpus.jsonl` content, pre-fix:

```
'Acme Corp and Erika Mustermann, erika@acme.example, IBAN [IBAN].'
```

The IBAN from the exported branch is masked; the email `redact_a` detected is in the clear.

## What was built

1. **`_compose_prior_spans` in `packages/latence-core/src/latence_core/redaction_policy.py`** —
   `finalize_redacted_chunk` now UNIONs the chunk's incoming `pii_spans` (populated by an upstream
   Redaction Stage) into the span set before masking. Both sets are chunk-local offsets into the
   same clean `content` (every Redactor detects on `content`, never on a predecessor's
   `masked_content`), so they share one coordinate system. Overlap resolution mirrors
   `_merge_floor` exactly: longer span wins, identical span → the Stage that just ran wins the tie
   and keeps its own provenance. Applied unconditionally, so a chunk a downstream policy SKIPS
   still keeps the masks a predecessor applied.
2. **Both deploy stacks rewired** to `redact_financial.depends_on: [redact_pii]`, with the corpus
   Export hanging off the last redaction link. Comments corrected — the old "deliberate SIBLINGS"
   rationale was the defect.

## Acceptance criteria

- [x] Two chained Redaction Stages with disjoint detectors both land their masks in the exported
      corpus, and both Stages' spans survive on the record, each attributable to its own detector.
- [x] A single-Redactor Pipeline is byte-identical (no upstream spans ⇒ the compose is a no-op) —
      OFF-BY-DEFAULT holds.
- [x] No committed `deploy/env/*/stack.yaml` carries a redaction Stage that is not an ancestor of
      its `export.jsonl_parquet` corpus Export.
- [x] ADR-0044 financial floor intact; ADR-0042 redaction stays chunk-level; ADR-0031 provenance
      unchanged (spans keep chunk-local offsets resolving to original source pages).
- [x] Fast suite does not regress.

## Regression test

`packages/latence-core/tests/test_redaction_composition.py`

- `test_chained_redactors_compose_masks_into_the_exported_corpus` — the behavioural half.
- `test_single_redactor_stack_is_byte_identical` — the off-by-default guard.
- `test_every_redaction_stage_reaches_the_corpus_export[gpu-learned|full-stack]` — the structural
  guard over the committed deploy stacks, so this shape cannot be re-introduced.

All three failed pre-fix, pass post-fix.

## Known residual (NOT fixed here — out of scope)

`QualityReportBuilder._redaction_quality` flattens the records of EVERY redaction Stage, so in a
chain the composed upstream spans are counted once per downstream Stage — `pii_total` / `per_type`
over-count in a multi-Redactor stack. This is counts-only observability, pre-existing (the sibling
layout already double-counted the labels the two Redactors share), and carries no PII and no
correctness or privacy consequence. Worth a follow-up: dedupe the rollup to the terminal Redaction
Stage per chunk.
