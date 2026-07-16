# R1-invariants-0 — Index-time SPARSE signal re-leaks the W13 context header (un-redacted KG PII)

**Severity:** critical
**Invariants violated:** REDACTION-BY-DEFAULT (ADR-0053); the W13 "embedding input ONLY" invariant (ADR-0039).
**Location:** `packages/latence-core/src/latence_core/stages/export.py` — `JsonlParquetExport._attach_signals` (the sparse branch).

## What to build

Feed the `SparseEmbedder` (and every other INSPECTABLE index-time signal generator) the **redacted
corpus content** (`_corpus_content` — `masked_content` when present, no W13 header), exactly like the
BM25 postings path already does (`_bm25_postings_batch` → `_corpus_content`, export.py). Reserve the
W13 context header (`_corpus_text` = redacted content **+** rendered header) for the **dense
`Embedder` only**, where the header inflates the vector without materializing any inspectable PII
surface (the ADR-0039 crux: "the header goes into the VECTOR").

## Failure scenario (the confirmed defect)

Default RAG stack wiring redaction + `context_enrichment` (W13) + `sparse_embedder`. Under the
DEFAULT `extract_on='unmasked'` posture the KG keeps the REAL entities (pipeline.py), so a chunk
whose text is redacted (`content` "Patient [REDACTED] was admitted", `masked_content` exported)
still carries a `context_header` with the raw PII surface (e.g. "Bruce Wayne", "Dr Thompkins").

`_attach_signals` computed `texts = [_corpus_text(row) for row in rows]`, which PREPENDS that header,
then encoded it with the `SparseEmbedder` (both reference `sparse.hashing` and the real
`SpladeCompositionEmbedder`). The emitted `sparse_indices`/`sparse_values` columns — which a store
ingests into its **inspectable** sparse inverted index — therefore contained term ids for
`bruce`, `wayne`, `thompkins`: tokens absent from the stored/redacted corpus text. The SPLADE
Provider's own redaction gate is defeated too, because the surface IS present in the
header-prefixed input text it receives.

BM25 correctly avoided this (`_corpus_content`); the sparse (and multi-vector sidecar) path did not.

## Fix

In `_attach_signals`, split the input text per signal class:

* dense `Embedder` → `_corpus_text(row)` (redacted content + W13 header) — unchanged;
* `SparseEmbedder` → `_corpus_content(row)` (redacted content, NO header);
* `MultiVectorEmbedder` → `_corpus_content(row)` (redacted content, NO header) — its raw per-token
  vectors ride an inspectable sidecar under `keep_multivectors`, so it must not carry the header
  either.

## Acceptance criteria

1. A wired `SparseEmbedder` over a redaction+W13 row emits the sparse vector of the **redacted
   content only** — byte-identical to encoding `masked_content` with no header — so no term id for a
   header-only PII token appears in `sparse_indices`.
2. The dense `Embedder` still embeds the header-prefixed text (W13 unchanged): existing
   `test_context_enrichment_export` assertions stay green.
3. A NON-VACUOUS regression test fails on the old code (header PII term id present in the sparse
   vector) and passes on the fix.
4. No regression: the feature-package suite + ruff + mypy --strict stay green; byte-identical export
   when no generator is wired (ADR-0017/0049) is preserved.
