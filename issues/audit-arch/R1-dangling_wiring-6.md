# R1 / dangling-wiring #6 — the four out-of-core extraction Providers must resolve pages through the `PageIndexResolver` seam, not `PageOffsetIndex(chunk.page_map)`

**Status: WONTFIX — DUPLICATE. Already fixed on this branch by commit `0bd1165`, filed as
`issues/audit-arch/R1-depth_actually_gained-0.md`, and already re-verified once as
`issues/audit-arch/R1-seams_real_not_hypothetical-4.md`.** No code change made by this ticket.
The evidence below is an INDEPENDENT third re-verification, not a new fix.

## Reported defect (severity: critical)

- **Dimension:** dangling wiring / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** `feat/arch-deepening`
- **Reported sites (line numbers as cited by the verifier):**
  - `packages/latence-ner-gliner/src/latence_ner_gliner/provider.py:318` (`_mentions_for`)
  - `packages/latence-relation-gliner/src/latence_relation_gliner/provider.py:374`
  - `packages/latence-ner-endpoint/src/latence_ner_endpoint/provider.py:181` (`_extract_one`)
  - `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:840` (`_build_records`)
- **Invariants at stake:** ADR-0021/0031 (chunk offsets and pages resolve to REAL original-document
  coordinates), ADR-0034 (degradation is counted, not silent).

C1 shed the per-chunk `PageMap`: since schema v18 only a document's **first** chunk carries it
(`packages/latence-core/src/latence_core/stages/chunk.py:213`,
`page_map=doc.page_map if i == 0 else None`). The report says the four out-of-core extraction
Providers were never converted to the document-scoped `latence_core.page_index.PageIndexResolver`
seam and still built a throwaway `PageOffsetIndex(chunk.page_map)` per chunk. Because
`chunk.page_map is None` on every non-first chunk, `page_index` is `None`, the mention-provenance
helper never writes `page_start`/`page_end`, and the mention/relation inherits the **chunk's** whole
page range instead of its own page. That hits the blessed `stacks/gpu-sota.yaml`,
`stacks/gpu-sota-pod.yaml` and `stacks/gpu-sota-glinker.yaml`, which all wire
`fused_entity_relation.gliner2`.

## Failure scenario (as reported, and reproduced below)

A multi-page document is chunked; only chunk 0 carries the page map (v18). A later chunk straddles a
page boundary and carries no map. A mention whose resolved document offset falls on page 2 of that
chunk is stamped with the chunk's inherited range `(1, 2)` instead of its own page `(2, 2)` — the
exported KG cites page 1 for text that is on page 2.

| Provider | pre-fix | correct |
|---|---|---|
| `entity.gliner` | `page_start=1, page_end=2` | `2, 2` |
| `relation.gliner_relex` | `1, 2` | `2, 2` |
| `entity.endpoint` | `1, 2` | `2, 2` |
| `fused_entity_relation.gliner2` | `1, 2` | `2, 2` |

## What to build

Nothing new. The correct fix — applied by `0bd1165`, still in force at `HEAD` — is the
seam-making-real one, not another layer: delete the hand-rolled per-chunk `PageOffsetIndex`
construction and route every lookup through ONE call-scoped
`latence_core.page_index.PageIndexResolver`, which owns the per-document memoisation, the half-open
end clamp and the `None` degradation. Each Provider constructs one resolver per
`extract`/`extract_fused` call, calls `resolver.observe(chunk)` at the top of its loop **before** the
`min_chars` filter (a document's map-carrying first chunk is very often the one that filter drops),
and stamps provenance from `resolver.pages_for_span(chunk, doc_start, doc_end)`.

This makes each call site's interface SMALLER (one method, one `None` contract) instead of
re-teaching four callers what a `PageOffsetIndex` is. It is a genuine deepening and must not be
reverted.

## Acceptance criteria

- [x] No Provider under `packages/*/src/` constructs `PageOffsetIndex(chunk.page_map)`.
      The surviving `PageOffsetIndex(...)` uses are document-scoped and C1-safe:
      `latence_console/browse.py:529` (fed by the document's own `page_map`),
      `latence_relation_llm/provider.py:338` (fed by `_document_page_maps`, which scans **all**
      chunks of a document for the map — `provider.py:509`), and
      `latence_core/stages/chunk.py` (the producer side, `doc.page_map`).
- [x] A mention/relation on a non-first chunk resolves its OWN page, not the chunk's inherited range,
      in all four Providers.
- [x] The map is still absorbed when `min_chars` drops the very chunk that carries it.
- [x] Per-Provider regression tests exist and FAIL on the pre-fix code.
- [x] `make test` green; no test weakened or deleted.

## Evidence of re-verification (this ticket)

1. **The cited line numbers are pre-`0bd1165`.** Restoring only the four provider modules with
   `git checkout 0bd1165^ -- <the four provider.py files>` reproduces the verifier's four line
   numbers EXACTLY, which is how we know the audit ran against a tree that predates the fix:

   ```
   latence-ner-gliner/.../provider.py:318:        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
   latence-relation-gliner/.../provider.py:374:  page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
   latence-ner-endpoint/.../provider.py:181:     page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
   latence-extract-gliner2/.../provider.py:840:  page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
   ```

2. **Against that restored pre-fix code, 7 tests fail** — every one of the four Providers, including
   `latence-ner-endpoint`, whose test also covers the observe-before-`min_chars` half:

   ```
   FAILED packages/latence-ner-gliner/tests/test_gliner_provider.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map
   FAILED packages/latence-ner-gliner/tests/test_gliner_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
   FAILED packages/latence-relation-gliner/tests/test_gliner_relex_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map
   FAILED packages/latence-relation-gliner/tests/test_gliner_relex_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
   FAILED packages/latence-ner-endpoint/tests/test_endpoint_ner.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map
   FAILED packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map
   FAILED packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
   ```

   with the reported failure mode verbatim (the mention cites page 1 for text on page 2):

   ```
   >       assert (prov.page_start, prov.page_end) == (2, 2)
   E       assert (1, 2) == (2, 2)
   E         At index 0 diff: 1 != 2
   ```

3. **At `HEAD` the four modules are restored and all four suites pass** (120 tests, 0 failures), and
   the fast suite is green: `3827 passed, 29 skipped`.

4. The audit's supporting claim that these Providers' suites "stay green (124 passed) because every
   test hand-builds a single `ChunkRecord` with `page_map` explicitly set" is no longer true at
   `HEAD`: `0bd1165` added a shared `_v18_two_chunk_document()` fixture to each of the four test
   modules that reproduces the v18 carriage shape (map on chunk 0 only, chunk 1 straddling the page
   boundary with `page_map=None`).

## Why WONTFIX rather than a second fix

The defect was real and critical; it is remediated at `HEAD` by an earlier fix in this same audit
campaign, with per-Provider tests that demonstrably fail without it. Re-implementing it would
duplicate `0bd1165` and churn the blessed SOTA Providers for no behavioural change. Filed here for
audit traceability. The verifier's fourth site (`latence-ner-endpoint`) and its `gpu-sota` stack
observation are both correct as descriptions of the pre-fix tree.
