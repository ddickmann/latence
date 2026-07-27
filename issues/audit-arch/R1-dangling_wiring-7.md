# R1 / dangling-wiring #7 — the three out-of-core PII Providers must pass a call-scoped `resolver` to `plan_chunk_redaction()`

**Status: WONTFIX — DUPLICATE (third report of the same finding).** Already fixed on this branch by
commit `96a462a`, filed as `issues/audit-arch/R1-depth_actually_gained-1.md`, and previously
re-verified as `issues/audit-arch/R1-seams_real_not_hypothetical-5.md`. **No code change made by
this ticket.** Everything below is an independent re-verification, not a new fix.

## Reported defect (severity: critical)

- **Dimension:** dangling wiring / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Reported sites:**
  - `packages/latence-pii-gliner/src/latence_pii_gliner/provider.py:299`
  - `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:690`
  - `packages/latence-pii-presidio/src/latence_pii_presidio/provider.py:323`
- **Reported seam:** `packages/latence-core/src/latence_core/redaction_policy.py:257`
- **Invariants at stake:** ADR-0021/0031 (chunk offsets resolve to real original-document
  provenance), ADR-0034 (no silent degradation), ADR-0044/0042 (chunk-level redaction, the
  universal PII floor and its audit artifact)

The three out-of-core learned PII Providers were reported to call `plan_chunk_redaction()` without
the `resolver=` argument, so `redaction_policy.py` minted a **fresh** `PageIndexResolver` per chunk.
Under schema v18 a document's `PageMap` rides its **first** chunk only
(`latence_core/stages/chunk.py`, `page_map=doc.page_map if i == 0 else None`), so a per-chunk
resolver never observes the map for chunks 1..N-1; `plan.resolver.pages_for_span(...)` returns
`None` and `build_chunk_span` stamps `page_start=page_end=None` on every `PIISpan` there. Not a
degraded page — *no* page. `redaction.gliner_pii` is the Redaction Stage wired by all three
gpu-sota stacks, so a compliance reviewer auditing a scanned 519-page contract would see a source
page for PII in chunk 0 and a blank page for the other ~398 chunks. The in-core twin
(`stages/redaction.py:496`) was converted and threads one shared resolver.

## Failure scenario

A multi-page document whose `PageMap` rides chunk 0 only:

* chunk 0: `content = text[:boundary]`, `page_map` **carried**;
* chunk 1: `content = text[boundary:]`, `char_start=boundary`, `page_map=None`.

A PII match lying wholly on page 2, inside chunk 1, is emitted with `page_start=None,
page_end=None` instead of `(2, 2)` — for `redaction.gliner_pii`, `redaction.gliner2` and
`redaction.presidio` alike.

## What to build

Nothing. The fix already on the branch makes the seam **smaller and realer** — the correct
architectural response, not another layer:

1. `resolver` is a **required keyword-only** parameter of `plan_chunk_redaction`
   (`packages/latence-core/src/latence_core/redaction_policy.py:218`) — verified at `HEAD`:
   `inspect.signature(...).parameters['resolver']` reports `kind=KEYWORD_ONLY`,
   `default=inspect._empty`. It has no default *because* it once had one; omitting it is now a
   `TypeError` at the call site, not a null page. The optional-default seam was hypothetical; the
   required argument is real.
2. Each of the four Redactors (in-core `stages/redaction.py` plus the three out-of-core Providers)
   builds **one** `PageIndexResolver` per `redact()` call and threads it into
   `_prepare` / `_redact_chunk`. Planning runs for EVERY chunk before the skip decision, so it is
   the one place guaranteed to observe the chunk carrying the document's map.

This is a genuine deepening and must not be reverted.

## Acceptance criteria

- [x] `plan_chunk_redaction` has **no** default for `resolver`; a caller that omits it fails loudly.
- [x] All three out-of-core learned Redactors construct one resolver per `redact()` call and pass
      it to every `plan_chunk_redaction` — `provider.py:314` (gliner), `:705` (gliner2), `:340`
      (presidio), all `resolver=resolver`, each preceded by `resolver = PageIndexResolver()` in the
      `redact()` body (`:284`, `:664`, `:317`).
- [x] The in-core twin (`stages/redaction.py:496`) does the same.
- [x] A PII span on a non-first v18 chunk resolves its OWN source page, not `None`.
- [x] Per-Provider regression tests exist and FAIL on the pre-fix code.
- [x] `make test` green; no test weakened or deleted.

## Evidence of re-verification (this ticket)

The reported line numbers (299 / 690 / 323, and seam `redaction_policy.py:257`) predate `96a462a`;
the verifier ran against a stale tree. At `HEAD` all three call sites pass `resolver=resolver`.

Independently re-simulating the pre-fix behaviour — substituting `resolver=PageIndexResolver()` at
the three Provider call sites, which still type-checks, so the required-arg guard alone does not
cover it — reproduces the reported failure mode verbatim in all three Providers:

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

The simulation was reverted (`git checkout --`); at `HEAD` those three tests pass.

## Why WONTFIX rather than a third fix

The defect was real and critical; it is remediated at `HEAD`. Re-implementing it would duplicate
`96a462a` and churn the blessed SOTA Redactors for no behavioural change. Filed for audit
traceability.

## Residual noted (not a defect, no change made)

The required-argument guard prevents **omission** but not **per-chunk construction**: a future
Provider could pass a freshly built `PageIndexResolver()` inside its chunk loop and type-check
clean — which is exactly how the simulation above was performed. What actually holds the invariant
today is the per-Provider regression test above, one per shipped Redactor. Any new out-of-core
Redactor should copy that test. Making per-chunk construction structurally impossible (e.g. binding
resolver lifetime to a call scope) was considered and rejected here as speculative: it would add a
layer rather than shrink the seam, which is the opposite of what this audit asks for.
