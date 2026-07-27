# R1 / seams-real-not-hypothetical #4 — learned entity/relation Providers must resolve pages through the PageIndexResolver seam, not `PageOffsetIndex(chunk.page_map)`

**Status: WONTFIX — DUPLICATE. Already fixed on this branch by commit `0bd1165`, filed as
`issues/audit-arch/R1-depth_actually_gained-0.md`.** No code change made by this ticket.
Evidence below is a re-verification, not a new fix.

## Reported defect (severity: critical)

C1 shed the per-chunk `PageMap` — since schema v18 only a document's **first** chunk carries it
(`latence_core/stages/chunk.py:220`, `page_map=doc.page_map if i == 0 else None`). The audit
reported that three learned Providers still built a throwaway `PageOffsetIndex(chunk.page_map)`
per chunk instead of going through the document-scoped `PageIndexResolver` seam:

- `packages/latence-ner-gliner/src/latence_ner_gliner/provider.py:318` (`_mentions_for`)
- `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:840` (`_build_records`)
- `packages/latence-relation-gliner/src/latence_relation_gliner/provider.py:374`

Because `chunk.page_map is None` on every non-first chunk, `page_index` is `None`, so the
mention-provenance helper never writes `page_start`/`page_end` and the mention inherits the
**chunk's** whole page range via `chunk_prov.model_copy`. Every `EntityMention`/`RelationMention`
emitted from a non-first, page-boundary-straddling chunk cites a wider/wrong page range in the
exported KG — an ADR-0021/0031 provenance regression, in the Providers blessed by
`stacks/gpu-sota-pod.yaml`, `stacks/gpu-sota-glinker.yaml` and `stacks/schema-induction.yaml`,
whose 768-token `chunk.markdown` budget makes straddling chunks the norm on real PDFs.

## Failure scenario (reproduced, see Evidence)

5-page document -> `MarkdownChunker(max_tokens=500)` -> 2 chunks; chunk 0 carries the page map,
chunk 1 does not (v18) and spans pages 1-2. A mention resolved at a document offset that falls on
page 2 of chunk 1 is stamped with the chunk's inherited range instead of its own page:

| Provider | pre-fix | correct |
|---|---|---|
| `entity.gliner` (`_mentions_for`) | `page_start=1, page_end=2` | `2, 2` |
| `fused_entity_relation.gliner2` (`_build_records`) | `1, 2` | `2, 2` |
| `relation.gliner_relex` | `1, 2` | `2, 2` |

## What to build

Nothing new. The correct fix — already applied — is the *seam-making-real* one, not another layer:
delete the hand-rolled per-chunk `PageOffsetIndex` construction and route every lookup through the
call-scoped `latence_core.page_index.PageIndexResolver`, which owns the per-document memoisation,
the half-open end clamp and the `None` degradation. Each Provider constructs **one** resolver per
`extract`/`extract_fused` call, calls `resolver.observe(chunk)` at the top of its loop **before**
the `min_chars` filter (a document's map-carrying first chunk is very often the one that filter
drops), and stamps provenance from `resolver.pages_for_span(chunk, doc_start, doc_end)`.

This makes the interface *smaller* at each call site (one method, one `None` contract) instead of
re-teaching five callers what a `PageOffsetIndex` is. It is a genuine deepening and must not be
reverted.

## Acceptance criteria

- [x] No Provider under `packages/*/src/` constructs `PageOffsetIndex(chunk.page_map)`.
      Remaining `PageOffsetIndex(...)` uses are document-scoped and C1-safe:
      `latence_console/browse.py` (falls back to `document.page_map`),
      `latence_relation_llm/provider.py:338` (fed by `_document_page_maps`, which scans **all**
      chunks for the document's map), `latence_core/stages/chunk.py:183` (`doc.page_map`, the
      producer side).
- [x] A mention on a non-first chunk resolves its **own** page, not the chunk's inherited range.
- [x] The map is still absorbed when the first chunk is dropped by `min_chars`.
- [x] Regression tests exist per Provider and FAIL on the pre-fix code.
- [x] `make test` green; no test weakened or deleted.

## Evidence of re-verification (this ticket)

Restoring the pre-fix providers with
`git checkout 0bd1165^ -- <the three provider.py files>` reproduces the audit's cited line
numbers **exactly** (318 / 840 / 374), confirming the verifier ran against a tree that predates
commit `0bd1165`. Against that restored code, 6 tests fail:

```
FAILED packages/latence-ner-gliner/tests/test_gliner_provider.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map
FAILED packages/latence-ner-gliner/tests/test_gliner_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
FAILED packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map
FAILED packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
FAILED packages/latence-relation-gliner/tests/test_gliner_relex_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map
FAILED packages/latence-relation-gliner/tests/test_gliner_relex_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars
```

with the reported failure mode verbatim:

```
>       assert (prov.page_start, prov.page_end) == (2, 2)
E       assert (1, 2) == (2, 2)
E         At index 0 diff: 1 != 2
```

Restoring `HEAD`, the same 8 page-resolution tests pass, and the fast suite is green:
`3827 passed, 29 skipped`.

The audit's remark that "the C1 lane's own `test_page_index_resolver.py` only exercises the core
`GazetteerEntityExtractor`" is likewise no longer true — `0bd1165` added Provider-level coverage
to all four learned extractors (the three above plus `latence-ner-endpoint`).

## Why WONTFIX rather than a second fix

The defect was real; it is remediated at `HEAD` by an earlier fix in this same audit campaign.
Re-implementing it would duplicate `0bd1165` and churn the blessed SOTA Providers for no
behavioural change. Filed here for audit traceability.
