# R1 — the S10 drift audit read `record.page_map`, collapsing to 1 chunk per document (v18)

- **Severity:** critical
- **Dimension:** memory claims real / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/quality_report_builder.py:710`
  (`_DriftAggregator.audit`); producer at
  `packages/latence-core/src/latence_core/stages/chunk.py:220`
  (`page_map=doc.page_map if i == 0 else None`)
- **Invariants at stake:** ADR-0021/0031 (provenance resolves to real original-document
  offsets — this is the audit that *proves* it), ADR-0034 (a degradation is counted, never a
  silent clean), ADR-0008 (the Quality Report substantiates the "AI-ready" claim)
- **Status:** **DUPLICATE — already fixed on this branch** by commit `96a462a`
  (ticket `R1-depth_actually_gained-2.md`). One residual coverage gap named by this audit was
  still open and **is closed by this commit**.

## What was wrong

C1 (schema v18) moved a document's `PageMap` off every chunk onto its **first** chunk only.
`_DriftAggregator.audit` still gated on the record:

```python
if not isinstance(record, ChunkRecord) or record.page_map is None:
    return
```

Under v18 that gate is true for every chunk after a document's first, so the corpus-level
Provenance-integrity drift audit silently narrowed from *every* chunk to *one chunk per
document* (~1/N coverage) while still rendering a confident `provenance drift: clean`. Drift
is most likely at **large** offsets — i.e. in exactly the later chunks the gate dropped. The
one Quality-Report section whose entire job is to notice this class of regression was itself
regressed by it.

## Failure scenario (reproduced)

One document whose Parser emitted a page map covering only the first half of the assembled
markdown, chunked by the real `MarkdownChunker` into 20 chunks, fed to
`LocalRunner._drift_diagnostics`:

| | `resolutions` | `recovered` | `max_drift_chars` | `records_audited` | `clean` |
|---|---|---|---|---|---|
| pre-fix (v18 producer) | 2 | 0 | 0 | **1** | **True** |
| main (v17, map on all chunks) | 40 | 20 | 1290 | 20 | False |

A real offset→page drift affecting 19 of 20 chunks was invisible.

## The fix (commit `96a462a`)

The gate was replaced by the seam the C1 lane had already built for exactly this: the
document-keyed, memoised `PageIndexResolver`. Every chunk is `observe`d before it is judged
interesting (a document's map-bearing first chunk is often the one with no char span), spans
resolve via `pages_for_span`, and the section is rolled up from
`PageIndexResolver.diagnostics` — the corpus-level roll-up that existed with no production
consumer. `PageIndexResolver.unresolved` now warns, counts-only, when chunks went unaudited,
so a section that merely shrank can no longer read as a confident `clean` (ADR-0034).

Post-fix the v18 stream reproduces main byte-for-byte:
`resolutions=40 recovered=20 max_drift_chars=1290 records_audited=20 clean=False`.

## Residual closed by THIS commit

The audit correctly noted that **no test caught the original defect**, and the fix's own three
regression tests only partly close that. All of them *hand-build* the v18 stream
(`page_map=page_map if i == 0 else None`), which encodes the producer's half of the contract
as an **assumption** rather than testing it: if `stages/chunk.py` and `_DriftAggregator` ever
disagree again, those tests still pass. The C5 byte-identity fixture cannot cover it either —
its corpus is one chunk per document (`records_audited: 3`), where "audit the first chunk" and
"audit every chunk" are indistinguishable — and `test_e2e_quality.py` only asserted
`records_audited > 0`.

Added `test_drift_audit_covers_every_chunk_the_real_chunker_emitted`: runs the **real**
Source→Parse→Chunk pipeline through the **real** `LocalRunner` on a document the real
`MarkdownChunker` splits into 20 chunks, and asserts the report audited what the chunker
actually emitted.

## Acceptance criteria

- [x] The drift audit resolves through `PageIndexResolver`, not `record.page_map`.
- [x] A v18 multi-chunk document reports drift identically to main's v17 carriage.
- [x] Chunks whose document never carried a map are counted `unresolved` and warned about,
      never folded in as clean resolutions (ADR-0034).
- [x] Producer and consumer are tested **together**, end-to-end, at a chunk count > 1:
      `records_audited == chunk.chunks_total`, `resolutions == 2 * chunks_total`.
- [x] Test fails on the pre-fix aggregator (`assert 1 == 20`) and passes after.
- [x] `make test` green; no test weakened or deleted; the C5 fixture untouched (ADR-0008).
