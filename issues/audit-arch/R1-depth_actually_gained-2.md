# R1 — the S10 drift audit still read `record.page_map`, so it audits 1 chunk per document

- **Severity:** major
- **Dimension:** depth actually gained / PROVENANCE IS TRUE (ADR-0021, ADR-0031)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/quality_report_builder.py:710`
  (`_DriftAggregator.audit`)
- **Invariants at stake:** ADR-0021/0031 (provenance resolves to real original-document
  offsets — the audit that *proves* it), ADR-0008 (the Quality Report is the
  substantiation of the "AI-ready" claim, and byte-identical across the C5 rewrite),
  ADR-0034 (a degradation is counted, never a silent clean)
- **Status:** resolved (fixed in this commit)
- **Siblings:** `R1-depth_actually_gained-0.md` (extraction Providers) and
  `-1.md` (redaction Providers) — the same C1 blast radius on the *record-producing*
  side. This ticket is the *auditing* side: the one section of the Quality Report whose
  job is to notice exactly this class of regression was itself regressed by it.

## What is wrong

C1 moved a document's `PageMap` off every chunk onto its **first** chunk only
(`SCHEMA_VERSION = 18`) and introduced `latence_core.page_index.PageIndexResolver` as the
one seam through which a page is resolved. `ChunkRecord.page_map`'s own field docstring
now says it in the contract:

> Callers MUST NOT read this field directly to resolve a page — a `None` here means
> "ask the resolver", not "no page map exists".

The S10 drift audit does exactly the forbidden thing:

```python
def audit(self, record: Record | StageOutput) -> None:
    if not isinstance(record, ChunkRecord) or record.page_map is None:
        return                                   # <- "no map on this record" == "skip"
    ...
    index = PageOffsetIndex(record.page_map)     # <- a throwaway index, per chunk
```

Under v18 that gate is true for every chunk of a document except its first, so the
corpus-integrity audit silently collapsed from N chunks to **1 chunk per document** —
~1/N coverage on a report that still says `clean: true` with a confident
`records_audited`. Drift is by nature most likely at LARGE offsets (accumulated
assembly/strip skew), i.e. in precisely the later chunks the gate now drops. The audit
keeps only the chunk least likely to be wrong.

Two further points make this an architecture defect rather than a stray `if`:

1. `PageIndexResolver` — introduced by this same lane, document-keyed, memoised, with a
   corpus-level `diagnostics` roll-up whose docstring names this exact consumer ("The
   corpus-level roll-up the per-chunk `PageOffsetIndex` construction made impossible:
   previously each throwaway index carried its own diagnostics and no caller summed
   them") — was **not used here**. The section the seam was built for was the section
   left behind.
2. The C5 byte-identity fixture cannot catch it.
   `packages/latence-core/tests/data/quality_report_c5_fixture.json` records
   `drift = {resolutions: 6, recovered: 0, max_drift_chars: 0, records_audited: 3}` for
   **3 single-chunk documents** — 1-per-document and all-per-document are the same number
   when N = 1, so the byte-identity gate stays green either way.

## Failure scenario

One document, 7 pages × 100 chars, chunked into 7 chunks, emitted the way v18 emits it
(map on chunk 0 only). The last chunk's `char_end` overruns the map's `total_chars` by 40
— real drift, at a large offset.

Folding that stream through `LocalRunner._drift_diagnostics` on the pre-fix code:

```
DriftDiagnostics(resolutions=2, recovered=0, max_drift_chars=0, records_audited=1)
```

`records_audited=1` of 7, and `clean is True` for a corpus that is not clean. The same
seven chunks with the map restored on every record (i.e. main's v17 carriage) give
`records_audited=7, resolutions=14, recovered=1` — so the number the report publishes
depends on the *carriage* of the page map rather than on the corpus, which is the
definition of a broken audit.

Multi-document is worse in the same way: two 2-chunk documents interleaved report
`records_audited=2` of 4.

## What to build

Make `_DriftAggregator` resolve through the seam instead of reading the record:

* hold one `PageIndexResolver` for the whole report build (the same call-scoped shape all
  twelve other resolver call sites on this branch use — no new memory axis);
* `observe()` **every** ChunkRecord before deciding it is interesting (the resolver's
  documented contract: a document's map-bearing first chunk is often the one with no
  span);
* audit a chunk when `pages_for_span(...)` resolves, i.e. when its *document* has a map —
  not when its *record* carries one;
* take `resolutions` / `recovered` / `max_drift_chars` from the resolver's corpus-level
  `diagnostics` roll-up rather than hand-summing per-chunk indices.

No contract change, no new report field, no schema bump: `DriftDiagnostics` keeps its
four fields, so ADR-0008's byte-identity fixture is untouched.

## Acceptance criteria

1. A v18 stream (page map on the first chunk only) of N chunks audits **N** records, not
   1; `resolutions == 2N`.
2. Drift that occurs in a chunk other than the first is caught: `recovered >= 1`,
   `clean is False`, `max_drift_chars` reports the real gap.
3. Documents are resolved against **their own** map — an interleaved two-document stream
   never resolves document B's offsets through document A's index.
4. A chunk whose document never carried a map is **not** audited and not counted as a
   clean resolution (the `None` section contract for a Parse-free / unparsed corpus
   survives).
5. `tests/data/quality_report_c5_fixture.json` is unchanged and the byte-identity gate
   stays green (3 single-chunk documents: `records_audited=3`, `resolutions=6` under both
   the old and the new rule).
6. `make test` green; `ruff` + `mypy --strict` clean on the touched files.

## Regression tests

`packages/latence-core/tests/test_quality_report.py`

* `test_drift_audit_covers_every_chunk_of_a_v18_document_not_just_the_first`
  — AC 1 + 2. Pre-fix: `assert 1 == 7`.
* `test_drift_audit_resolves_each_document_through_its_own_page_map` — AC 3.
  Pre-fix: `assert 2 == 4`.
* `test_drift_audit_skips_chunks_whose_document_never_carried_a_page_map` — AC 4.
  Pre-fix: `assert 1 == 2`.

## Follow-up (not in this fix)

`PageIndexResolver.unresolved` — the counted degradation ADR-0034 asks for — has nowhere
to land in the report: `DriftDiagnostics` is `extra="forbid"` and a fifth field would bump
`QUALITY_SCHEMA_VERSION` and rewrite the C5 byte-identity fixture. Surfacing
"chunks whose document had no page map" belongs with the next schema revision, not with a
provenance fix.
