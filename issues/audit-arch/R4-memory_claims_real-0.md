# R4 — the R3 LRU window is smaller than what shipped Providers hold in flight, so a mention's page provenance silently degrades to a WRONG page

- **Severity:** critical
- **Dimension:** memory claims real (ADR-0033), PROVENANCE IS TRUE (ADR-0021/0031), NO SILENT
  ERRORS (ADR-0034)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/page_index.py` —
  `DEFAULT_MAX_LIVE_DOCUMENTS` (:110) and `PageIndexResolver._evict_beyond_window` (:374)
- **Triggered through:** `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py`
  — `_MAX_BATCH_SIZE = 64` (:136), the bare `PageIndexResolver()` (:632), `_iter_batches`
  observing every chunk (:952); consumed at :874/:878 and `_mention_provenance` (:1024)
- **Claim contradicted:** `DEFAULT_MAX_LIVE_DOCUMENTS`'s own docstring — *"every shipped batching
  Provider uses `batch_size` 16 … 64 is 4× that, so no shipped stack can evict a document it is
  still working on"* — restated in `docs/adr/0033-streaming-memory-bounded-runner.md:107` and
  `docs/writing-an-adapter.md:123`.
- **Invariants at stake:** ADR-0021/0031 (a chunk offset resolves to its REAL source page — the
  exact thing the C1 seam was introduced to protect), ADR-0033 (the retention bound is real),
  ADR-0034 (a degradation is counted and observable, never silent)
- **Status:** FIXED (this commit)
- **Not on main:** on `main` the page map rode EVERY chunk, so resolution never depended on
  retention. This is a regression introduced by the C1 seam + the R3 window, not a pre-existing
  defect.

## What was wrong

R3 bounded the resolver's retention with an LRU window of **64 documents**. Both halves of the
justification for that number are false:

1. **`batch_size` is not clamped to 16.** `latence_extract_gliner2` clamps to `_MAX_BATCH_SIZE =
   64`, so an operator writing `batch_size: 64` (in range, accepted, clamped to itself) gets a
   window-sized batch — a 1× margin, not 4×. `latence_pii_gliner2` / `latence_pii_gliner` /
   `latence_pii_presidio` are worse: their `batch_size` comes from `AdapterBase.batch_size()` →
   `validate_batch_size`, which enforces only `>= 1` with **no upper clamp**, and they resolve
   pages only at flush (`redaction_policy.build_chunk_span` → `plan.resolver.page_for`), so
   `batch_size: 256` puts 256 documents in flight against a 64-document window.

2. **More fundamentally, `batch_size` does not bound the number of DISTINCT DOCUMENTS in the
   window at all.** A Provider must `observe()` every chunk it is handed — including the ones its
   `min_chars` floor drops, because a document's map rides its first chunk and that chunk is very
   often exactly the one dropped (the module docstring says so). So a corpus of SHORT documents
   (one-page memos, notes, covering letters — the ordinary shape of a real corpus) pushes hundreds
   of documents through the window while ONE batch of long-document chunks accumulates. Each short
   document consumed a full window slot despite weighing ~1/500th of a long one, and evicted the
   long document the batch was still holding.

When that happens, the long document's later chunks carry no page map of their own (schema v18),
`pages_for_span` returns `None`, and `_mention_provenance` degrades to the chunk's inherited page
range. The emitted record is not missing a page — it carries a **wrong, plausible-looking one**.

**Observability did not cover it.** `evicted_reentries` / `unresolved` live on a resolver that is
local to the Provider's `extract()` / `redact()` call and is discarded when the call returns; the
only production reader of those counters is `_DriftAggregator`, which reads its OWN resolver. The
operator got one `logger.warning` line and a corpus of subtly wrong page citations in the KG.
(The evicted-id ring was itself sized to `max_live`, so a document evicted long enough ago fell
out of the ring and its re-entry was not even counted.)

## Failure scenario (reproduced, no torch)

Real `MarkdownChunker` + real `_iter_batches` + real `PageIndexResolver` + real
`_mention_provenance`. Corpus: 400 documents, every 9th long (a short cover page then 25 short body
pages → 5 chunks), the rest single-short-page stubs that survive chunking but fall under the
Provider's `min_chars = 40` floor. Identical inputs; only the knob differs:

| config | multi-page mentions | WRONG_PAGE | evicted_reentries |
| --- | --- | --- | --- |
| `batch_size=16`, `max_live=64` | 225 | 0 | 0 |
| `batch_size=64`, `max_live=64` | 225 | **3** | 3 |
| `batch_size=64`, `max_live=100000` | 225 | 0 | 0 |

Concrete emitted record: `doc0108#chunk-4` → mention stamped page range `(22, 26)`; the true page
for that offset span is `(26, 26)`. The third row is the proof that the window is the sole cause —
an unbounded resolver (the pre-R3 behaviour, and `main`'s per-chunk `PageOffsetIndex` behaviour)
gets it right on byte-identical input.

## What was built

The window is kept — R3's memory problem is real — but its **unit** is corrected, so the interface
stays one knob and no layer is added.

* `DEFAULT_MAX_LIVE_DOCUMENTS = 64` → **`DEFAULT_MAX_LIVE_PAGES = 50_000`**, and the constructor
  argument `max_live_documents` → `max_live_pages`. A page span is what the retained state is
  actually made of (~550 bytes; the measured 519-page document is ~279 KB of `PageMap`), so the
  budget is now a real memory ceiling (~27 MB, fixed in the corpus) rather than a proxy that
  priced a one-page memo and a 519-page contract identically. 50,000 spans holds ~96 documents of
  519 pages — more big-document headroom than the 64-document window it replaces — **or** 50,000
  single-page ones, which is what removes the pathology: the short documents that dominate a real
  stream can no longer evict the long one a batch is working on.
* `PageOffsetIndex.page_count` — the index now says what it weighs, which is what a caller that
  must bound page state has to be able to ask.
* `_evict_beyond_window` never evicts the last live index. An index is inserted before the window
  is trimmed, so a document whose map alone exceeds the budget would otherwise be evicted the
  instant it was absorbed and EVERY lookup against it would degrade — the same wrong-page bug,
  made unconditional.
* The evicted-id ring is sized independently (`_EVICTED_RING = 1024`) instead of following the
  window, so a re-entry is still *named* when it happens far from the eviction.
* The false claim is removed from the constant's docstring, `docs/adr/0033` and
  `docs/writing-an-adapter.md`, and replaced with what is actually true and what is not.

Deliberately NOT done: raising the number and leaving the unit alone (papers over it — the
document count can be exceeded by any corpus shape, whatever the number), and adding a
`retain(batch)` / release call for Providers to drive (a new interface on 13 call sites to
compensate for a bound stated in the wrong unit — a layer, not a fix).

## Acceptance criteria

1. **AC1 — the shipped stack does not mis-page.** With the bare `PageIndexResolver()` every
   shipped Provider constructs, `batch_size` at the Provider's own `_MAX_BATCH_SIZE` ceiling, and
   a corpus of mostly short documents, EVERY mention's page range equals what a per-document
   `PageOffsetIndex` gives (the pre-C1 / `main` answer). Zero `evicted_reentries`, zero
   `unresolved`.
2. **AC2 — the regression test is not vacuous.** It asserts a floor on the number of
   page-straddling mentions (the only shape that can detect the defect at all) and pins the
   mechanism: one batch's worth of stream must push more distinct documents through the resolver
   than the old 64-document window held, and some document's chunks must straddle a batch
   boundary.
3. **AC3 — the bound is still a bound.** A corpus of large page maps still evicts under the
   default: retention stops at `DEFAULT_MAX_LIVE_PAGES`, it does not grow with the corpus
   (ADR-0033).
4. **AC4 — the bound does not create a worse bug.** A single document whose map exceeds the whole
   budget resolves exactly, every offset, with `unresolved == 0`.
5. **AC5 — the R3 guarantees survive.** Eviction still frees the map (weak-reference measured),
   still folds the evicted index's drift into the corpus roll-up so the Quality Report's drift
   numbers are byte-identical windowed vs unbounded (ADR-0008), and a re-entry is still counted
   and warned once, counts-only (ADR-0034).
6. **AC6 — no claim is left standing that the code does not hold.** The constant's docstring,
   ADR-0033 and `writing-an-adapter.md` state the budget in pages, say why, and no longer assert
   that no shipped stack can evict a document it is still resolving.

## Tests

- `packages/latence-extract-gliner2/tests/test_gliner2_page_window.py`
  - `test_a_streamed_short_document_corpus_never_mis_pages_a_long_one` (AC1) — **fails on the
    pre-fix code**: `3 of 225 mentions were stamped with a WRONG page range … ['doc0108#chunk-4:
    stamped (22, 26), true (26, 26)', 'doc0225#chunk-3: stamped (17, 21), true (21, 21)',
    'doc0225#chunk-4: stamped (22, 26), true (26, 26)']`
  - `test_the_short_documents_are_what_fills_the_window` (AC2)
- `packages/latence-core/tests/test_page_index_resolver.py`
  - `test_the_default_budget_still_bounds_a_corpus_of_large_page_maps` (AC3)
  - `test_a_document_bigger_than_the_whole_budget_still_resolves` (AC4)
  - `test_resolver_does_not_retain_every_documents_page_map_for_the_whole_run`,
    `test_drift_from_an_evicted_document_still_reaches_the_corpus_roll_up`,
    `test_a_lookup_for_an_evicted_document_is_counted_and_announced` (AC5, ported to the new unit
    with their assertions intact)
