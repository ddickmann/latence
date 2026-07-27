# R1 — `_DriftAggregator.audit()` gated on `record.page_map`, collapsing the S10 drift audit to 1 chunk/document

- **Severity:** major
- **Dimension:** dangling wiring / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/quality_report_builder.py:710`
  (`_DriftAggregator.audit`)
- **Invariants at stake:** ADR-0021/0031 (chunk offsets resolve to real original-document
  offsets — this aggregator is the audit that *proves* it), ADR-0008 (the Quality Report is
  byte-identical across the C5 rewrite), ADR-0034 (a degradation is counted, never a silent
  clean)
- **Status:** **DUPLICATE — already resolved on this branch**
- **Resolved by:** commit `96a462a` ("fix(audit-R1): The Quality Report's S10 provenance
  drift audit gates on `record.page_map is None` …"), ticketed as
  `R1-depth_actually_gained-2.md`. Independently re-verified here (see *Verification* below);
  **no further code change was required.**
- **Siblings (same C1 blast radius, all resolved on this branch):**
  `R1-depth_actually_gained-0.md` (extraction Providers), `-1.md` (redaction seam +
  the three PII Redactors), `-2.md` (this site), `R1-dangling_wiring-6/-7.md`
  (the out-of-core extraction / PII Providers), `R1-seams_real_not_hypothetical-4/-5.md`.

## What was wrong

C1 (schema v18) moved a document's `PageMap` off *every* chunk onto its **first** chunk only
(`stages/chunk.py:220` — `page_map=doc.page_map if i == 0 else None`) and introduced
`latence_core.page_index.PageIndexResolver`, a document-keyed memoiser that absorbs the map
from whichever record carries it and resolves every later chunk of that document from it.

`_DriftAggregator` — the corpus-level Provenance-integrity roll-up whose entire job is
catching offset→page drift, i.e. exactly the invariant C1 modified — was not converted. Its
first line was:

```python
if not isinstance(record, ChunkRecord) or record.page_map is None:
    return
```

Under v18 that gate is true for every chunk after a document's first, so the audit silently
narrowed from N chunks per document to 1 — while still emitting a confident `clean`.

## Failure scenario

For a 519-page document (~399 chunks), `main` audited 399 chunks and reported
`drift.resolutions ≈ 798` (two endpoint lookups per chunk). On this branch, before the fix,
`page_map` is present on chunk 0 only, so `_audited == 1` and `drift.resolutions == 2` —
99.75% of the audit skipped.

Two consequences, both invariant regressions:

1. **The detector stops detecting.** Drift is most likely at *large* offsets — a page map
   whose spans drift mid-document (the realistic OCR case) lives in exactly the later chunks
   the gate dropped. Such a corpus reports `clean is True`.
2. **The report changes value for identical input** versus `main`, against ADR-0008.

Why the existing suite could not catch it: the C5 byte-identity fixture's three documents are
single-chunk, so the collapsed and correct audits agree on it.

## What was built (in `96a462a`)

`_DriftAggregator` holds one `PageIndexResolver` for the run. `audit()` `observe`s **every**
`ChunkRecord` first (the map-bearing first chunk may itself be the one with no char span),
then resolves the chunk's `[char_start, char_end)` through
`resolver.pages_for_span(record, …)`; a `None` result means the chunk's document genuinely
has no map (an unparsed parent) and is left unaudited *and counted* as
`resolver.unresolved`. `finish()` reads the resolver's corpus-level `diagnostics` roll-up
instead of hand-summing per-index tallies, and (per `0f76a6a`) warns counts-only when
`unresolved` is non-zero, so a shrunken section can never read as a confident clean.

This makes the seam **real** rather than adding a layer: the per-chunk throwaway
`PageOffsetIndex` is deleted, not wrapped, and the resolver's `diagnostics`/`unresolved`
accessors — which existed with no production consumer — gain their intended one.

Note: `quality_report_builder.py:587` (`ParseQuality.page_map_present`) is *not* affected —
it reads `DocumentRecord.page_map`, whose carriage v18 did not narrow.

## Acceptance criteria

- [x] `_DriftAggregator.audit()` contains no `record.page_map` gate and builds no per-chunk
      `PageOffsetIndex`; all resolution goes through the run's single `PageIndexResolver`.
- [x] For a v18 stream of N chunks from one document (map on chunk 0 only),
      `records_audited == N` and `resolutions == 2N`.
- [x] Drift injected in the **last** chunk of such a stream is reported
      (`recovered >= 1`, `max_drift_chars` exact, `clean is False`).
- [x] Two interleaved documents each resolve through **their own** map (document-keyed;
      a shared index would itself be a provenance lie).
- [x] Chunks whose document never carried a map are not audited and not counted as clean
      resolutions; the gap is surfaced counts-only per ADR-0034.
- [x] Regression tests fail against the pre-fix gate and pass after.

## Verification performed for this ticket

The pre-fix gate was temporarily reinstated at the assigned site and the branch's regression
tests re-run; all four failed with exactly the predicted collapse, then passed on restore:

```
assert diag.records_audited == 7
E  assert 1 == 7
E   +  where 1 = DriftDiagnostics(resolutions=2, recovered=0, max_drift_chars=0,
                                  records_audited=1).records_audited

FAILED test_quality_report.py::test_drift_audit_covers_every_chunk_of_a_v18_document_not_just_the_first
FAILED test_quality_report.py::test_drift_audit_resolves_each_document_through_its_own_page_map
FAILED test_quality_report.py::test_drift_audit_skips_chunks_whose_document_never_carried_a_page_map
FAILED test_page_index_resolver.py::test_drift_audit_surfaces_the_unresolved_count_instead_of_shrinking_silently
```

Note the pre-fix `recovered=0` / `max_drift_chars=0` on a corpus whose last chunk overruns
the map by 40 characters — the "confident clean on a drifting corpus" this ticket is about.

A sweep for residual instances of the same class found none in the drift lane:
`QualityReportBuilder._drift_diagnostics` → `_DriftAggregator` is the single producer and
`LocalRunner._drift_diagnostics` (`runner.py:2367`) forwards to it verbatim.

## Regression tests

- `packages/latence-core/tests/test_quality_report.py::test_drift_audit_covers_every_chunk_of_a_v18_document_not_just_the_first`
- `packages/latence-core/tests/test_quality_report.py::test_drift_audit_resolves_each_document_through_its_own_page_map`
- `packages/latence-core/tests/test_quality_report.py::test_drift_audit_skips_chunks_whose_document_never_carried_a_page_map`
- `packages/latence-core/tests/test_page_index_resolver.py::test_drift_audit_surfaces_the_unresolved_count_instead_of_shrinking_silently`
