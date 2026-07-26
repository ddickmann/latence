# R1 — Four ecosystem Providers resolve a sub-chunk mention's page via per-chunk `PageOffsetIndex`

- **Severity:** critical (as reported)
- **Dimension:** memory claims real / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Invariants at stake:** ADR-0021/0031 (chunk offsets resolve to real original-document
  pages), ADR-0034 (no silent errors), ADR-0016 (thin core)
- **Status:** **WONTFIX — confirmed defect, but already fixed on this branch.** Exact
  duplicate of `issues/audit-arch/R1-depth_actually_gained-0.md`, fixed in commit
  `0bd1165` earlier in this same audit campaign. No further code change is required or
  made by this ticket; what follows is the independent re-verification.

## What was reported

C1 changed the *carriage* of `PageMap` from "on every chunk" to "on a document's **first**
chunk only" (`SCHEMA_VERSION = 18`) and introduced `latence_core.page_index.PageIndexResolver`
as the seam that makes that safe. Four shipped extraction Providers were reported as still
doing the pre-v18 thing:

```python
page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
```

| Provider | Reported site |
| --- | --- |
| `latence-ner-gliner` `GlinerEntityExtractor._mentions_for` | `provider.py:318` |
| `latence-ner-endpoint` `EndpointEntityExtractor._extract_one` | `provider.py:181` |
| `latence-relation-gliner` `GlinerRelexExtractor._build_records` | `provider.py:374` |
| `latence-extract-gliner2` `Gliner2FusedExtractor._build_records` | `provider.py:840` |

Under v18 that expression is `None` for every chunk except a document's first, so every
`EntityMention` / `RelationMention` emitted from chunk 1..N-1 silently degrades to its
parent chunk's inherited page range instead of citing its own page.

## Failure scenario (reproduced)

A 2-page document, chunked so that chunk index 1 straddles the page boundary with
provenance `pages=(1, 2)` and `page_map=None` (v18 carriage). A mention whose resolved
document offset truly lies on page 2 must be reported as `pages=(2, 2)`. On the pre-fix
code it is reported as `pages=(1, 2)` — the chunk's inherited range.

## Verification performed (2026-07-26)

1. **Current branch state.** All four Providers now import
   `latence_core.page_index.PageIndexResolver`, construct **one** resolver per `extract()` /
   `relate()` call, call `resolver.observe(chunk)` at the top of the chunk loop **before**
   the `min_chars` filter, and stamp pages via `resolver.pages_for_span(...)`:
   `latence-ner-gliner/provider.py:272,295,342`;
   `latence-ner-endpoint/provider.py:165,168,211`;
   `latence-relation-gliner/provider.py:283,332,402,433`;
   `latence-extract-gliner2/provider.py:632,878,931,973`.
   The reported `PageOffsetIndex(chunk.page_map)` expression is gone from all four.

2. **Non-vacuity of the landed regression tests.** The pre-fix `latence-ner-gliner`
   provider (`git show 0bd1165^:…/provider.py`) was restored over the current file and the
   two page-resolution tests re-run:

   ```
   >       assert (mentions[0].provenance.page_start, mentions[0].provenance.page_end) == (2, 2)
   E       assert (1, 2) == (2, 2)
   E         At index 0 diff: 1 != 2
   FAILED test_gliner_provider.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map
   FAILED test_gliner_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
   ```

   `(1, 2)` is exactly the inherited chunk range the report predicts. The file was restored
   with `git checkout --` immediately afterwards; both tests pass on HEAD.

3. **Residual scan.** No shipped (non-test) code still builds a per-chunk index. The three
   remaining `chunk.page_map` reads are all legitimate:
   - `latence_core/page_index.py:109,117` — the seam itself (`rehome_page_maps`).
   - `latence_core/stages/relation.py:539` — selects the map-carrying chunk per document.
   - `latence_relation_llm/provider.py:512` (`_document_page_maps`) — this Provider is
     **document**-scoped by construction: it groups chunks per document and takes the map
     from whichever chunk of that document carries one, then builds one index per document
     (`provider.py:338`). That is hand-rolled memoisation with the same semantics as the
     resolver, correct under v18. It is a seam-consistency wart, not a provenance defect —
     recorded here, deliberately not "fixed", because changing it would be churn with no
     behavioural delta.
   - `latence_console/browse.py:227,529` — falls back to `document.page_map`, the
     authoritative persisted copy (ADR-0017), so the console is unaffected by carriage.

## Acceptance criteria (all met by `0bd1165`)

- [x] None of the four Providers derives a page index from `chunk.page_map` directly.
- [x] Exactly one `PageIndexResolver` lives per Stage invocation; it is `observe`d for
      **every** chunk handed in, before any `min_chars`/emptiness filter.
- [x] A regression test per package drives the Provider with a two-chunk v18 document
      (map on chunk 0 only) and asserts the later chunk's mention resolves to its **own**
      page, not the inherited range — and fails on the pre-fix code (evidence above).
- [x] Streaming is preserved: the resolver holds one `PageOffsetIndex` per document, no
      chunk accumulation (ADR-0033).
- [x] `make test` green.
