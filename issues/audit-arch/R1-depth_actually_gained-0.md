# R1 — C1 moved the page map off every chunk but four shipped Providers still read `chunk.page_map`

- **Severity:** critical
- **Dimension:** depth actually gained / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Invariants at stake:** ADR-0021/0031 (provenance resolves to real original-document
  pages), ADR-0034 (no silent errors), ADR-0016 (thin core — the fix must not move heavy
  deps or grow core)
- **Status:** resolved (fixed in this commit)

## What is wrong

C1 changed the *carriage* of `PageMap` from "on every chunk" to "on a document's **first**
chunk only" (`SCHEMA_VERSION = 18`, `stages/chunk.py`: `page_map=doc.page_map if i == 0 else
None`) and introduced the seam that makes that safe:
`latence_core.page_index.PageIndexResolver` — a document-keyed memoiser that absorbs the map
from whichever record of a document carries it and resolves every later chunk of that
document from the memoised `PageOffsetIndex`. Its own docstring states the contract:

> Callers MUST NOT read this field directly to resolve a page — a `None` here means "ask the
> resolver", not "no page map exists".

Every in-core Stage was migrated to the seam (`stages/entity.py`, `stages/relation.py`,
`stages/redaction.py`, `stages/inline_refs.py`, `stages/header_refs.py`). **Four shipped
extraction Providers outside core were not**, and still do the pre-v18 thing:

```python
page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
```

| Provider | Site |
| --- | --- |
| `latence-ner-gliner` `GlinerEntityExtractor._mentions_for` | `provider.py:318` |
| `latence-ner-endpoint` `EndpointEntityExtractor._extract_one` | `provider.py:181` |
| `latence-relation-gliner` `GlinerRelexExtractor._build_records` | `provider.py:374` |
| `latence-extract-gliner2` `Gliner2FusedExtractor._build_records` | `provider.py:840` |

Under v18 that expression is `None` for **every chunk except a document's first**, so those
Providers take the "no page map available" degradation path on ~all chunks and every
`EntityMention` / `RelationMention` inherits the *chunk's* `[page_start, page_end]` range
instead of resolving its own offset. The mention then cites a page it does not lie on.

This is the deepening's own blast radius: the seam is right, the shed field is right, but the
migration stopped at the core package boundary. It is silent — no counter, no typed error
(the Providers do not hold a resolver, so `PageIndexResolver.unresolved` never sees these
lookups either, violating ADR-0034's "a degradation is counted, not swallowed"), and the full
suite is green because every existing page test feeds a **single** chunk that carries the map.

`latence-relation-llm` is NOT affected: it already hoists a document→map dictionary
(`_document_page_maps`) off the chunk list, which happens to be v18-correct.

## Failure scenario

A 2-page document assembled as `PageMap.from_page_texts(["Alice is here.", "Bob is over
here."])` — page 1 = `[0,16)`, page 2 = `[16,33)` — chunked into two chunks the way v18 emits
them:

* chunk 0: doc `[0,14)`, pages `1..1`, **carries** `page_map`;
* chunk 1: doc `[13,33)`, pages `1..2` (it straddles the boundary), `page_map=None`.

`GlinerEntityExtractor.extract([chunk0, chunk1])` on a mention "Bob" at doc `[16,19)` — wholly
on page 2 — emits `provenance.page_start=1, page_end=2`: the chunk's inherited range.
`PageIndexResolver` resolves the identical span as `(2, 2)`, and so does the same Provider on
main (where chunk 1 also carried the map). Observed pre-fix:

```
>       assert (by_text["Bob"].provenance.page_start, by_text["Bob"].provenance.page_end) == (2, 2)
E       assert (1, 2) == (2, 2)
```

Real-world scale: a 519-page document chunked at 640 tokens is ~399 chunks, of which 398 lose
per-mention page resolution — the knowledge graph cites a page range, not a page, for
essentially the whole corpus.

## What to build

Finish the C1 migration at the four sites. No new layer, no new abstraction — the seam already
exists and is the *smaller* interface (`resolver.pages_for_span(chunk, start, end) ->
(int, int) | None` replaces "know that `PageOffsetIndex` exists, know the `None` case, know the
`max(end - 1, start)` clamp"). Concretely, per Provider:

1. Build **one** `PageIndexResolver` per `extract` / `extract_fused` call (page maps are
   document-scoped; the resolver memoises one index per `document_record_id`, so a 400-chunk
   document builds the `bisect` array once instead of 400 times — a straight win over both
   the pre-v18 and the current code).
2. `resolver.observe(chunk)` at the **top of the chunk loop, before the `min_chars` filter** —
   a document's first chunk is the one carrying the map and is very often the one the filter
   drops. This includes the two Providers that filter before iterating
   (`latence-ner-endpoint`'s list comprehension, `latence-extract-gliner2`'s `_iter_batches`).
3. Replace the `page_index: PageOffsetIndex | None` parameter of the private
   `_mention_provenance` / `_relation_provenance` helpers with the already-resolved
   `pages: tuple[int, int] | None`, mirroring `stages/entity.py`. The helper stops knowing
   about page indices at all — the interface gets *narrower*, not wider.
4. Drop the now-unused `PageOffsetIndex` imports and correct the four module docstrings, which
   still describe the pre-v18 "carried on the chunk" mechanism.

`latence_core.page_index` is pure-Python over `latence_core.pagemap` — importing it adds no
heavy dependency, so ADR-0016 is untouched.

## Acceptance criteria

- [x] In each of the four Providers, a mention/relation on a chunk that carries **no**
      `page_map` (every non-first chunk of a v18 document) resolves to its **own** source page,
      not the chunk's inherited page range.
- [x] The map is absorbed even when the document's first chunk is dropped by `min_chars`
      (observe-before-filter), proven for the batching Providers as well as the simple loop.
- [x] Behaviour is unchanged for a pre-v18 stream where every chunk carries the map, and for a
      document with no page map at all (still degrades to the inherited range, never raises).
- [x] A non-vacuous regression test per Provider **fails on the old `PageOffsetIndex(chunk.page_map)`
      code** and **passes on the fix**, asserting the emitted Provenance (not a config flag).
- [x] `make test` stays green; no test weakened or deleted.

## Resolution

Fixed. All four Providers now hold one `PageIndexResolver` per extraction call, `observe`
every chunk before any filtering, and stamp Provenance from `resolver.pages_for_span(...)`.

Regression tests (each verified failing pre-fix with `assert (1, 2) == (2, 2)`):

* `packages/latence-ner-gliner/tests/test_gliner_provider.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map`
* `packages/latence-ner-gliner/tests/test_gliner_provider.py::test_page_map_is_absorbed_even_when_the_first_chunk_is_below_min_chars`
* `packages/latence-ner-endpoint/tests/test_endpoint_ner.py::test_mention_page_resolves_on_a_later_chunk_that_carries_no_page_map`
* `packages/latence-relation-gliner/tests/test_gliner_relex_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map`
* `packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py::test_page_resolves_on_a_later_chunk_that_carries_no_page_map`
