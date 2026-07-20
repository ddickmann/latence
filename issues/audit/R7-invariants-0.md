# R7-invariants-0 — Duplicate documents halve the Quality Report's PII tally

**Severity:** major
**Where:** `packages/latence-core/src/latence_core/quality_report_builder.py` (`_redaction_quality`)
**Invariant regressed:** ADR-0034 (counts-only observability — the redaction audit trail must
describe the exported corpus) and OFF-BY-DEFAULT/BYTE-IDENTICAL (the plain single-Redactor default
path must be unchanged versus `main`).

## What is wrong

The audit-R6 fix to `_redaction_quality` de-duplicated the redaction rollup by `record_id`:

```python
latest: dict[tuple[str, str], StageOutput] = {}
for record in records:
    if isinstance(record, ChunkRecord | RedactionRecord):
        latest[(type(record).__name__, record.record_id)] = record
```

It did so on the stated premise — in its own docstring — that "A single-Redactor Pipeline has no
duplicate `record_id`s, so its counts are byte-identical to before."

That premise is false. `record_id`s are **content-addressed**:

- `packages/latence-core/src/latence_core/stages/source.py:37-39` — `_document_id` is
  `"sha256:" + sha256(data).hexdigest()`.
- `packages/latence-core/src/latence_core/stages/chunk.py:196` — `record_id=f"{doc.record_id}#chunk-{i}"`.

So two byte-identical documents in a corpus produce two separately-exported chunk records that
share one `record_id`. The dedup collapses them into one, and every PII span in the second copy
disappears from the report.

## Failure scenario

A corpus of two byte-identical files (`memo_a.txt`, `memo_b.txt`), each containing an email and an
IBAN, run through `source.local_folder -> parser.plaintext -> chunk.markdown ->
redaction.hybrid_rule -> export.jsonl_parquet` — **one** Redaction Stage, the default path.

- `records.jsonl` holds **2** rows carrying **4** PII spans in total.
- This branch reports `pii_total=2`, `per_type={email: 1, iban: 1}`.
- `main` (no `latest` dict; iterates `records` directly) reports `pii_total=4`,
  `per_type={email: 2, iban: 2}`.

The operator's redaction audit trail therefore under-states the PII actually present in the
exported corpus, silently, on the default path.

The existing regression test `test_single_redactor_quality_counts_are_unchanged`
(`packages/latence-core/tests/test_redaction_composition.py:585`) is a hollow guard: its fixture
corpus contains no duplicate documents, so it passes with the dedup reverted and does not guard the
property its name claims.

## What to build

Fix the chain double-count (the real R6 defect) at the **Stage** level rather than by
de-duplicating records:

- A Redaction Stage that another Redaction Stage depends on, directly or transitively, is
  **superseded** — the downstream Stage re-emits the same chunks with the upstream spans composed
  in (`_compose_prior_spans`), and only that version reaches the Export. Do not gather superseded
  Stages' output at all.
- Tally the terminal Redaction Stages' records with **full multiplicity**. No `record_id` dedup.

## Acceptance criteria

1. A single-Redactor pipeline over a corpus with two byte-identical documents reports a
   `pii_total` / `per_type` / `per_action` that matches the exported `records.jsonl`, not half of
   it.
2. The R6 chained-Redactor property still holds: a `redact_a -> redact_b` chain tallies each span
   exactly once and names no `pii_type` absent from the exported corpus.
3. Document-level counts (`documents_redacted`, `documents_with_pii`,
   `redaction_disabled_for_sensitive`) remain distinct-document rollups.
4. Determinism preserved; no content or PII values in the report (counts and type/action names
   only).
5. A non-vacuous regression test that fails on the pre-fix code and passes on the fix.
6. `make test` stays green at or above 3320 passed / 29 skipped.

## Resolution

Fixed. `QualityReportBuilder._terminal_redaction_stages` drops superseded Redaction Stages before
the records are gathered; `_redaction_quality` sums the gathered records directly again.
Regression test: `test_duplicate_documents_keep_their_full_pii_tally`
(`packages/latence-core/tests/test_redaction_composition.py`), which asserts the report against the
exported corpus rather than a literal.
