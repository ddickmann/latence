# R2 — C1's `PageIndexResolver` seam was never wired into `AdapterBase.resolve_pages`, the public adapter helper

- **Severity:** critical
- **Dimension:** invariants and ADRs — a provenance invariant the seam is *advertised* to keep
- **Branch:** feat/arch-deepening
- **Site:**
  - `packages/latence-core/src/latence_core/providers/adapter.py:433` (`AdapterBase.resolve_pages`)
  - `docs/writing-an-adapter.md:80` (the guide that instructs the broken call)
  - `packages/latence-core/tests/test_e2e_adapter_base.py:121` (the repo's own reference adapter)
- **Invariants at stake:** **ADR-0021/0031 PROVENANCE IS TRUE** (a sub-record's char span resolves
  to its own real source page), ADR-0034 no silent errors (the degradation was neither counted nor
  surfaced), and the ADR-0022 §"a mention on a later page of a multi-page chunk must cite *its own*
  page; inheriting the chunk's first page would violate the S4 core requirement".
- **Status:** **FIXED** on this branch.

## What was wrong

C1 moved the document page map off every chunk (`SCHEMA_VERSION` 17 → 18: the Chunk Stage stamps
`page_map` on a document's **first** chunk only) and put offset→page resolution behind one seam,
`latence_core.page_index.PageIndexResolver`. Five core call sites were migrated; R1 then migrated
four first-party ecosystem Providers that had been missed.

The one call site nobody migrated is the **public** one — the helper the framework tells *third
parties* to use, and the only one that ships with a written promise:

> These make the "offsets/page-provenance intact" invariant the **default**, not a per-adapter
> reimplementation. — `docs/writing-an-adapter.md`

`AdapterBase.resolve_pages` still took a per-chunk `page_map` and built a throwaway
`PageOffsetIndex` from it:

```python
if page_map is None:
    return fallback_page_start, fallback_page_end
index = PageOffsetIndex(page_map)
```

Under v18 `chunk.page_map` is `None` on every chunk after a document's first, so the *only* branch
a real adapter ever reaches on a multi-chunk document is the fallback — silently returning the
parent chunk's **inherited** page range. `providers/adapter.py` had a **zero-line diff** in this
change set, and `PageIndexResolver` appeared nowhere in `docs/`.

Two things made it invisible:

1. The degradation always produces a *plausible* page (the chunk's own range), so nothing looks
   broken — and, unlike every migrated call site, it was not even counted
   (`PageIndexResolver.unresolved`), so ADR-0034's "counted rather than swallowed" did not apply.
2. The repo's reference adapter (`test_e2e_adapter_base.py::_AdapterGazetteer`) calls the helper
   exactly as the guide instructs, and the E2E fixture is a **one-page, one-chunk** `.txt` corpus,
   where the inherited range and the true page are the same number.

## Failure scenario

A 6-page document chunked at `max_tokens=40` → 42 chunks, 10 of which straddle a page boundary.
Emitting mentions the way the shipped guide instructs — `rebase_offsets(...)` then
`resolve_pages(..., page_map=chunk.page_map, fallback_page_start=chunk.provenance.page_start, ...)`
— gives **10 of 102 mentions the WRONG page** on this branch (chunk 6 spans pages 1-2; a mention at
doc offset 528 is reported as pages `(1, 2)` when its true page is `(1, 1)`). The identical probe
against `main`'s `latence_core`: **0 of 102 wrong** — because on `main` the map rode every chunk.

Reduced to the regression test: a 2-page document, map on chunk 0 (which contains no match), a
mention at document offset 102 inside chunk 1 (`[90, 119)`, provenance pages 1-2). True page is 2;
the branch reported `(1, 2)`.

## What was built

The seam was made **real** and the interface **smaller** — no new layer.

- `AdapterBase.resolve_pages(chunk, doc_char_start, doc_char_end, *, resolver)` now delegates to
  `PageIndexResolver.pages_for_span`. It builds no index of its own, owns no end clamp, and keeps
  exactly the one thing a record-producing adapter needs on top of the seam: the degrade to the
  source chunk's inherited page range when no map is available for the document **at all** — a
  case the resolver has already COUNTED (`resolver.unresolved`), so the fallback is no longer
  silent.
- The interface shrank: the two `fallback_page_*` parameters are gone, derived from the source
  chunk instead — which is what every caller passed anyway (`chunk.provenance.page_start/_end`).
  Five parameters → three plus the resolver.
- `docs/writing-an-adapter.md` gains **the missing half of the contract**: one `PageIndexResolver`
  per Capability call (across batches) and `resolver.observe(chunk)` for **every** chunk before any
  filter — because the map-carrying first chunk is very often the one a `min_chars`/term filter
  drops. `PageIndexResolver` now appears in `docs/`. `docs/CONFORMANCE.md` C3 and
  `docs/providers/README.md` name the same seam and link the discipline.
- The reference adapter (`_AdapterGazetteer`) was rewritten to that pattern — it is the executable
  copy of the guide, so it must demonstrate the discipline, including observing the chunk it finds
  nothing in.
- Two stale docstrings in `latence-pii-presidio` that still claimed page resolution went through
  `AdapterBase.resolve_pages` over a per-chunk `PageMap` (R1 had migrated the code to
  `PageIndexResolver` but not the prose) now name the real seam.

Deliberately **not** done: keeping a `page_map=` compatibility overload. That is the shallow
module the audit forbids — it would re-admit the exact broken call the guide used to print, and the
whole point of C1 is that there is now one way to resolve a page.

## Acceptance criteria

- [x] `AdapterBase.resolve_pages` resolves through `PageIndexResolver`; no `PageOffsetIndex` is
      constructed anywhere in `providers/adapter.py`.
- [x] A mention on a later page of a page-straddling chunk that carries **no** `page_map` cites its
      own page, through the public helper and through the reference adapter.
- [x] The genuine no-map-at-all degradation still returns the inherited range **and** is counted.
- [x] The guide, `CONFORMANCE.md` and `providers/README.md` describe the v18 shape and the
      one-resolver-per-call + `observe()`-every-chunk discipline; `PageIndexResolver` is in `docs/`.
- [x] Non-vacuous regression test that fails on the old code (evidence below); no test weakened or
      deleted (unit coverage went 2 → 3 cases and now pins the degradation counter).
- [x] `make test` green; `scripts/verify-local.sh` green.

## Evidence it failed pre-fix

`test_reference_adapter_cites_the_mentions_own_page_on_a_later_chunk`, run against the reference
adapter as the branch shipped it (the guide's call, verbatim):

```
        # ...and page 2 is where offset 102 lives. Inheriting the chunk's (1, 2) is the defect.
>       assert (mention.provenance.page_start, mention.provenance.page_end) == (2, 2)
E       assert (1, 2) == (2, 2)
E
E         At index 0 diff: 1 != 2
packages/latence-core/tests/test_e2e_adapter_base.py:386: AssertionError
=========================== short test summary item ============================
FAILED packages/latence-core/tests/test_e2e_adapter_base.py::test_reference_adapter_cites_the_mentions_own_page_on_a_later_chunk
```

The test is non-vacuous in both directions: it also fails if the adapter observes only the chunks
it matches in (the map-carrying chunk 0 holds no term), which is the discipline half of the fix.
