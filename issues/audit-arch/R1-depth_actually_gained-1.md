# R1 — the new `resolver` argument was OPTIONAL, and all three shipped PII Redactors omitted it

- **Severity:** critical
- **Dimension:** depth actually gained / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Invariants at stake:** ADR-0021/0031 (provenance resolves to real original-document
  pages), ADR-0034 (no silent errors — a degradation is counted or typed, never swallowed),
  ADR-0044/0042 (chunk-level redaction, the financial floor), ADR-0016 (thin core)
- **Status:** resolved (fixed in this commit)
- **Sibling:** `R1-depth_actually_gained-0.md` — the same C1 blast radius on the four
  *extraction* Providers. This ticket is the *redaction* half, and it fixes the seam itself.

## What is wrong

C1 moved a document's `PageMap` off every chunk onto its **first** chunk only
(`SCHEMA_VERSION = 18`) and introduced `latence_core.page_index.PageIndexResolver` — a
document-keyed memoiser that absorbs the map from whichever record carries it and resolves
every later chunk of that document from it. The redaction seam was migrated by threading a
resolver into the shared plan:

```python
def plan_chunk_redaction(
    chunk, policy, *, detector_types, sensitive_set, min_chars,
    resolver: PageIndexResolver | None = None,   # <- new on this branch, OPTIONAL
) -> ChunkRedactionPlan:
    ...
    page_resolver = resolver if resolver is not None else PageIndexResolver()
```

The `None` branch mints a **throwaway resolver per chunk**. Its own docstring says so, in
prose: *"Omitting it creates a throwaway resolver … but it forfeits page resolution on later
chunks of a v18 stream."* An invariant written as a warning is not an invariant. Every one of
the three shipped learned Redactors outside core took the default:

| Provider | Capability | Site |
| --- | --- | --- |
| `latence-pii-gliner2` `Gliner2PIIRedactor._prepare` | `redaction.gliner2` | `provider.py:690` |
| `latence-pii-gliner` `GlinerPIIRedactor._prepare` | `redaction.gliner_pii` | `provider.py:299` |
| `latence-pii-presidio` `PresidioRedactor._redact_chunk` | `redaction.presidio` | `provider.py:323` |

Under v18 the throwaway resolver has a map only for a document's first chunk, so on chunk
1..N `PageIndexResolver.page_for` returns `None` and `build_chunk_span` stamps
`page_start=None, page_end=None` on **every** `PIISpan`. Not a wrong page — *no* page. The
in-core `HybridPIIRedactor` (`stages/redaction.py:498`) passes a call-scoped resolver and is
correct, which is exactly what hid the gap: the core tests exercise the correct path, and each
Provider's own page test feeds a **single** chunk that carries the map.

`redaction.gliner2` is the wizard-blessed redactor, so the guided-setup stack — the default
path a new user lands on — loses PII source-page provenance on every chunk after the first of
every document.

## Failure scenario

A 2-page document `PageMap.from_page_texts(["Nothing here.", "Reach alice@acme.com."])` —
page 1 = `[0,15)`, page 2 = `[15,36)` — chunked the way v18 emits it:

* chunk 0: `content = text[:15]`, `page_map` **carried**;
* chunk 1: `content = text[15:]`, `char_start=15`, `page_map=None`.

The email lies wholly on page 2. `Gliner2PIIRedactor(...).redact([chunk0, chunk1])` emits its
span as:

```
>       assert (span.page_start, span.page_end) == (2, 2), (
E       AssertionError: a PII span on a non-first v18 chunk must resolve its OWN source page
E         through the document-scoped resolver, not degrade to None
E       assert (None, None) == (2, 2)
```

Identical output pre-fix from `GlinerPIIRedactor` and `PresidioRedactor`. On main (where every
chunk carried the map) the same call yields `(2, 2)`, so this is a straight regression, and it
scales: a 519-page document chunked at 640 tokens is ~399 chunks, 398 of which export every
PII span with a null source page. A compliance reviewer asking "which page did we mask?" gets
nothing back.

## What to build

Two changes, both of which make the seam **smaller and realer** — no new layer:

1. **Make `resolver` a required argument** of `plan_chunk_redaction`. The optional default is
   the defect: it is a silently-degrading fallback for the one thing this seam exists to
   guarantee, and three of three shipped callers took it. Removing the default converts a
   silent provenance loss into a `TypeError` at the call site, which is what ADR-0034 asks for
   (failures surface, never swallowed). The `PageIndexResolver` import inside the function and
   the `page_resolver = resolver if … else …` branch both disappear.
2. **Give each of the three Providers one resolver per `redact` call**, threaded through the
   private per-chunk helper it already has (`_prepare` / `_redact_chunk`). Planning already
   runs for EVERY chunk before the skip decision, so `plan_chunk_redaction`'s existing
   `resolver.observe(chunk)` is a guaranteed absorb point — no separate observe pass, and no
   ordering trap for the batching Providers (gliner/gliner2 buffer *after* preparing; presidio
   holds the resolver across `AdapterBase.batched` batches, next to the analyzer state it
   already carries the same way).

`latence_core.page_index` is pure Python over `latence_core.pagemap`, so ADR-0016 is untouched.
Behaviour is otherwise unchanged: same spans, same offsets, same masking, same floor.

## Acceptance criteria

- [x] In each of the three PII Providers, a `PIISpan` on a chunk that carries no `page_map`
      (every non-first chunk of a v18 document) resolves to its **own** source page.
- [x] `plan_chunk_redaction` cannot be called without a resolver — the omission that caused
      this is a type error at the call site, not a null page.
- [x] Unchanged for a single-chunk stream, a pre-v18 stream where every chunk carries the map,
      and a document with no page map at all (still degrades to `None`, still never raises).
- [x] A non-vacuous regression test per Provider, asserting the emitted `PIISpan` pages (not a
      config flag), **failing on the old code** and passing on the fix.
- [x] `make test` green; full `scripts/verify-local.sh` gate green (a seam signature changed).
      No test weakened or deleted.

## Resolution

Fixed. `plan_chunk_redaction(..., resolver=...)` is now required; the three Providers each hold
one `PageIndexResolver` for the whole `redact` call. The four in-repo test call sites that
relied on the default now pass an explicit resolver (assertions untouched).

Regression tests (each verified failing pre-fix with `assert (None, None) == (2, 2)`):

* `packages/latence-pii-gliner2/tests/test_gliner2_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map`
* `packages/latence-pii-gliner/tests/test_gliner_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map`
* `packages/latence-pii-presidio/tests/test_presidio_pii_provider.py::test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map`
