# R1 / seams-real-not-hypothetical #5 — the learned PII Redactors must pass a call-scoped `resolver` to `plan_chunk_redaction()`

**Status: WONTFIX — DUPLICATE. Already fixed on this branch by commit `96a462a`, filed as
`issues/audit-arch/R1-depth_actually_gained-1.md`.** No code change made by this ticket.
The evidence below is a re-verification, not a new fix.

## Reported defect (severity: critical)

- **Dimension:** seams real, not hypothetical / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Reported sites:**
  - `packages/latence-pii-gliner/src/latence_pii_gliner/provider.py:299` (`GlinerPIIRedactor._prepare`)
  - `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:690` (`Gliner2PIIRedactor._prepare`)
- **Invariants at stake:** ADR-0021/0031 (provenance resolves to real original-document pages),
  ADR-0034 (no silent degradation), ADR-0044/0042 (chunk-level redaction, the universal PII floor
  and its audit artifact)

Both learned PII redactors were reported to call `plan_chunk_redaction()` without a `resolver`, so
each chunk received a throwaway `PageIndexResolver`. Under schema v18 a document's `PageMap` rides
its **first** chunk only (`latence_core/stages/chunk.py`, `page_map=doc.page_map if i == 0 else
None`), so a per-chunk resolver never observes the map for chunks 1..N; `plan.resolver.page_for(...)`
returns `None` and `build_chunk_span` stamps `page_start=page_end=None` on every `PIISpan`. Not a
wrong page — *no* page. `redaction.gliner_pii` is the Redaction Stage in `stacks/gpu-sota-pod.yaml`,
`stacks/gpu-sota-glinker.yaml` and `stacks/schema-induction.yaml`, so on the blessed stacks a
compliance reviewer could no longer tell which source page a redacted span came from for any chunk
but a document's first.

`plan_chunk_redaction`'s own docstring warned about exactly this, in prose — an invariant written
as a warning is not an invariant, which is the architectural half of the finding.

## Failure scenario

A multi-page document whose `PageMap` rides chunk 0 only:

* chunk 0: `content = text[:boundary]`, `page_map` **carried**;
* chunk 1: `content = text[boundary:]`, `char_start=boundary`, `page_map=None`.

A PII match lying wholly on a later page, inside chunk 1, is emitted with
`page_start=None, page_end=None` instead of its real source page, for both `redaction.gliner_pii`
and `redaction.gliner2` (and `redaction.presidio`, the third shipped learned redactor).

## What to build

Nothing new. The correct fix — already applied by `96a462a` — makes the seam **smaller and realer**,
not another layer:

1. `resolver` is now a **required** keyword argument of
   `plan_chunk_redaction` (`packages/latence-core/src/latence_core/redaction_policy.py:218`); it has
   no default *because* it once had one. Omitting it is a `TypeError` at the call site, not a null
   page. That is the structural half: the optional-default seam was hypothetical, the required
   argument is real.
2. Each learned Redactor builds **one** `PageIndexResolver` per `redact()` call and threads it into
   `_prepare` / `_redact_chunk`, so planning — which runs for EVERY chunk, before the skip decision —
   is the one place guaranteed to observe the chunk carrying the document's map.

This is a genuine deepening and must not be reverted.

## Acceptance criteria

- [x] `plan_chunk_redaction` has **no** default for `resolver`; a caller that omits it fails loudly
      (`TypeError`), so the reported defect cannot silently reappear.
- [x] All three shipped learned Redactors (`latence-pii-gliner`, `latence-pii-gliner2`,
      `latence-pii-presidio`) construct one resolver per `redact()` call and pass it to every
      `plan_chunk_redaction`.
- [x] A PII span on a non-first v18 chunk resolves its OWN source page (not `None`).
- [x] Per-Provider regression tests exist and FAIL on the pre-fix code.
- [x] `make test` green; no test weakened or deleted.

## Evidence of re-verification (this ticket)

The current sites are `provider.py:314` (gliner), `:705` (gliner2) and `:340` (presidio), all
`resolver=resolver` — the reported line numbers (299 / 690) predate `96a462a`, confirming the
verifier ran against a stale tree.

Re-simulating the pre-fix behaviour by substituting `resolver=PageIndexResolver()` at those three
call sites reproduces the reported failure mode verbatim, in all three Providers:

```
FAILED packages/latence-pii-gliner/tests/test_gliner_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map
FAILED packages/latence-pii-gliner2/tests/test_gliner2_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map
FAILED packages/latence-pii-presidio/tests/test_presidio_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map
```

```
>       assert (span.page_start, span.page_end) == (2, 2), (
            "a PII span on a non-first v18 chunk must resolve its OWN source page through the "
            "document-scoped resolver, not degrade to None"
        )
E       AssertionError: a PII span on a non-first v18 chunk must resolve its OWN source page through the document-scoped resolver, not degrade to None
E       assert (None, None) == (2, 2)
```

The simulation was reverted (`git checkout --`); at `HEAD` those three tests pass and the three PII
Provider suites are green.

## Why WONTFIX rather than a second fix

The defect was real and critical; it is remediated at `HEAD` by an earlier fix in this same audit
campaign. Re-implementing it would duplicate `96a462a` and churn the blessed SOTA Redactors for no
behavioural change. Filed here for audit traceability.
