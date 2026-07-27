# R3 — `PageIndexResolver` retains one `PageMap` + `PageOffsetIndex` per document for the whole run

- **Severity:** major
- **Dimension:** memory claims real (ADR-0033), NO SILENT ERRORS (ADR-0034), PROVENANCE IS TRUE
  (ADR-0021/0031)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/page_index.py`
  — `PageIndexResolver.__init__` (`_page_maps` / `_indices` / the never-written `_diag`) and
  `PageIndexResolver._index_for`
- **Claim contradicted:** `docs/adr/0033-streaming-memory-bounded-runner.md:98` —
  "Both sides of those Stages are `O(checkpoint batch)`", written into the ADR's **Honest memory
  boundary** section by this change set, under that section's own restated discipline: *do not
  claim a bound the code does not hold.*
- **Invariants at stake:** ADR-0033 (streaming / memory-bounded runner), ADR-0034 (a degradation is
  counted, never silent), ADR-0021/0031 (a chunk offset resolves to its real source page),
  ADR-0008 (the Quality Report's drift section stays exact)
- **Status:** FIXED (this commit)

## What was wrong

One `PageIndexResolver` is constructed per Stage invocation (per `extract()` / `relate()` /
`redact()` call — 13 production construction sites, all the bare `PageIndexResolver()`), and a
Stage's Provider is handed the **whole corpus stream** in one call. The resolver absorbed a
document's `PageMap` from the first chunk that carried it (route 2, schema v18), built a
`PageOffsetIndex` from it, and **never released either**:

```python
self._page_maps[document_record_id] = carried     # never deleted
self._indices[document_record_id] = index          # never deleted
```

So a doc-level Stage's page-map retention was `O(documents seen so far)` — linear in the corpus —
for the entire duration of the Stage, on both the streamed-input and streamed-output side that
ADR-0033 had just declared `O(checkpoint batch)`.

The module already *described* the missing half. `__init__` carries a `_diag` accumulator whose
comment reads "Drift for documents whose index has been discarded is folded in here as it is
produced" — an eviction policy that did not exist: `_diag` was never written and `_indices` was
never evicted. `R1-memory_claims_real-13.md` recorded this as a residual and deferred it "if a
memory-bound ticket is ever raised against the resolver". This is that ticket.

## Failure scenario (measured)

Probe: 40 documents × 60 pages fed **lazily** (generators throughout, nothing materialised) through
the real `MarkdownChunker` into the real `GazetteerEntityExtractor.extract()`:

```
resolver.documents      == 40
len(resolver._indices)  == 40      # linear in corpus, not in batch
```

Cost per document, measured by `tracemalloc` on the C1 report's own scenario (a 519-page document):
**279,052 B** for the `PageMap` + `PageSpan` objects and **4,949 B** for the index ≈ **284 KB
retained per document** — ~9.5× the 29,927 B of JSON that C1 removed from the wire. A 1,000-document
519-page corpus therefore retained ~284 MB *inside a single* Entity / Redaction / Fused Stage, and
~2.8 GB at 10,000 documents. An operator who sizes RAM off ADR-0033's "Honest memory boundary"
section provisions for `O(checkpoint batch)` and OOMs.

The retention was avoidable, not inherent: chunks arrive **grouped by document** (`_emit_chunk_records`
emits per document and checkpoint order is preserved), so at any instant only the documents a Stage
holds *in flight* need an index — at most `batch_size` of them (16 for every shipped batching
Provider), never the corpus.

## What was built

`PageIndexResolver` now keeps a **bounded live window** of document indices, LRU by last use:

* `max_live_documents` (default `DEFAULT_MAX_LIVE_DOCUMENTS = 64`, 4× the largest shipped Provider
  in-flight batch of 16 chunks) bounds `_indices`. Absorbed maps are held **only** by the index
  built from them, so evicting the index frees the `PageMap` too — the 279 KB, not just the 4.9 KB.
* Eviction **folds the evicted index's drift diagnostics into `_diag`** — exactly what the existing
  comment promised — so the corpus-level `diagnostics` roll-up the Quality Report's drift section
  reads (ADR-0008) stays byte-for-byte what it was with unbounded retention.
* Seeded maps (`from_documents` / the `page_maps` argument, route 1) are **pinned**: they are the
  caller's own materialised state, they must resolve in any order, and evicting them could not free
  anything the caller was not already holding. Only the absorbed route — the one every shipped Stage
  uses, and the one the defect is about — is bounded.
* A chunk that arrives for a document whose index was already evicted, carrying no map of its own,
  is **counted and warned, never silent** (ADR-0034): `evicted_reentries` plus one counts-only
  `logger.warning` naming the window (no ids, no content). That is the signal that a Provider holds
  more documents in flight than `max_live_documents` — and the constructor argument is the fix.
* `documents` becomes a monotone count of the distinct documents whose map the resolver has learned
  (eviction does not decrease it); `live_documents` exposes the retained window.

`docs/adr/0033-streaming-memory-bounded-runner.md` records the bound rather than leaving the reader
to infer it from the `O(checkpoint batch)` sentence.

## Acceptance criteria

- [x] Streaming N documents through one resolver retains page-map state for **strictly fewer than N**
      documents — bounded by the live window, not the corpus — with the earliest document's
      `PageMap` provably unreachable (weak reference dead after collection).
- [x] Every chunk of every document still resolves to **exactly** the pages a per-document
      `PageOffsetIndex` gives (ADR-0021/0031), and `unresolved == 0` across the whole run.
- [x] Drift produced by a document that was later evicted still reaches the corpus-level
      `diagnostics` roll-up (ADR-0008) — identical numbers to an unbounded resolver.
- [x] Re-entry after eviction is counted (`evicted_reentries`) and warned counts-only, never a
      silent wrong page (ADR-0034).
- [x] Route 1 (seeded) semantics unchanged.
- [x] `make test` green.
