# R3 — the Qdrant sink's reconcile guard counts Export ROWS but compares against distinct POINTS, so a corpus with two byte-identical documents can never load

- **Severity:** major
- **Dimension:** cross-lane integration / ADR-0017 (files are the source of truth), ADR-0033 (memory-bounded), ADR-0034 (no silent errors)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #183 (qdrant sink), #182 (SharePoint/OneDrive Source), ADR-0018/0029 (retraction/purge)
- **Status:** fixed

## What is wrong

`latence-core` is deliberately **content-addressed**: a document's id is the sha256 of its bytes
(`packages/latence-core/src/latence_core/stages/source.py:37` `_document_id`, carried onto the
`DocumentRecord` verbatim at
`packages/latence-parser-plaintext/src/latence_parser_plaintext/provider.py:100`
`record_id=raw.provenance.document_id`), and a chunk id is derived from it
(`packages/latence-core/src/latence_core/stages/chunk.py:196`
`record_id=f"{doc.record_id}#chunk-{i}"`). Two **byte-identical** documents in one corpus therefore
produce Export rows carrying the **same `record_id`** — by design, since the same content *is* the
same record. The rows still differ (Provenance carries each file's own `file_name` / `source_uri`).

The sink derives one deterministic UUIDv5 point per `record_id`
(`packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py:130` `point_id`), so those two rows
collapse onto **one point** — which is correct, and is the whole idempotency story.

But the reconcile convergence guard compares the two quantities as if they were the same number:

- `sink.py:322` — `self._sweep(load_id, upserted + unchanged, ...)` passes the count of Export
  **rows** processed;
- `sink.py:435-460` — `_sweep` counts the **points** carrying this load's marker and raises
  `SinkLoadError` on `marked != touched`.

With a duplicate, `touched == 2` while the collection can only ever hold `1` marked point, so the
guard trips on every run. The failure is **deterministic and permanent** — a second and third run
fail identically — and the error message **misdiagnoses the cause** as a concurrent load, offering a
remedy ("re-run once no other load is in flight") that can never help.

The `LoadReport` on the non-reconciling path was wrong for the same reason: `reconcile: false`
reported "upserted 2" over a collection holding one point.

## Failure scenario

Proven end to end on the branch. Two byte-identical `.txt` files (`a.txt`, `b_copy.txt`) in a source
folder → `latence run` → `records.parquet` holds two rows whose `record_id` is the same string
(`sha256:...#chunk-0`). Driving `QdrantSink({'collection': 'corpus'}).load(export)` against the
package's own shipped `FakeQdrantClient`:

```
SinkLoadError: refusing to sweep collection 'corpus': this load processed 2 record(s) but
1 point(s) carry its marker '5f367d9c...'. Another load is running against this collection
concurrently (or the Export repeats a record_id). Nothing was deleted — the collection still
holds the data this load read. Re-run the load once no other load is in flight.
```

**Why the green suite could not see it.** Every sink test builds its Export through
`write_export(out_dir, records: dict[str, str])`
(`packages/latence-sink-qdrant/tests/test_qdrant_sink.py:78-95`) — a dict keyed by `record_id`, so a
duplicate id is structurally impossible in every fixture. The fake was never driven with the input
the real pipeline produces.

**Cross-lane impact.** Duplicated files are the normal state of an enterprise document store, and
#182's SharePoint/OneDrive Source (which ingests a whole library) makes them near-certain. The only
workaround was `reconcile: false`, which the module's own docstring (`sink.py:29-45`) and
`verify_absent` identify as the setting that leaves retracted/purged records live and searchable —
so the documented GDPR erasure runbook silently stops working for any corpus with a duplicate
document.

## What to build

Make the guard compare like with like: count the **distinct points** this load touched, not the
Export rows it read, and keep the count exact **without** materialising a corpus-sized id set
(ADR-0033 — the sweep and the reader are both O(batch)).

1. **Within a batch** — collapse rows onto their derived point id, keeping the *last* occurrence.
   That is exactly what an upsert of a repeated id does on a real Qdrant (last writer wins), so the
   stored state is byte-identical to today's; only the counting changes.
2. **Across batches** — the same `record_id` reappearing in a later batch is detected from the
   store, not from memory: `_partition` already retrieves the stored payload for the batch's ids, so
   it also reads `_latence_load_id` and treats a point already carrying **this run's** marker as a
   repeat. Exact, O(batch), and no extra round-trip.
3. **Report the truth** — `upserted` / `unchanged` count distinct points; a new
   `LoadReport.duplicate_rows` records the Export rows that collapsed onto a point already handled;
   `records` stays "rows read from the Export" and `points` is the new distinct count. The summary
   names duplicates only when there are any (so an ordinary run's output is unchanged).
4. **Drop the misdiagnosis** — the guard's message no longer blames "the Export repeats a
   record_id", because a repeated `record_id` is now handled, not a fault.

## Acceptance criteria

- [x] An Export containing two rows with the same `record_id` loads successfully with the default
      `reconcile: true`, converging on one point that carries the last row's content.
- [x] `LoadReport` for that load reports `upserted=1`, `duplicate_rows=1`, `records=2` (rows read),
      `points=1` — on both the reconciling and the `reconcile: false` path.
- [x] A re-run of the same duplicate-bearing Export is idempotent: `unchanged=1`, `upserted=0`,
      `deleted=0`, and still no error.
- [x] Retraction still converges in the presence of duplicates: dropping the duplicated document
      from the Export and re-loading deletes its point, and `verify_absent` reports it absent.
- [x] The concurrency guard still fires: a genuine marker mismatch (a co-running load restamping
      this load's points) still raises `SinkLoadError` and deletes nothing.
- [x] Duplicates spanning two batches (`batch_size=1`) are counted once, proving the cross-batch
      path, not just the within-batch dedupe.
- [x] Memory stays O(batch): no set of ids sized by the corpus is accumulated in `load` (ADR-0033).
- [x] Off-by-default byte-identity: for an Export with no duplicate `record_id`, every stored point,
      payload and report field is unchanged from before the fix.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.

## Regression test

`packages/latence-sink-qdrant/tests/test_qdrant_sink.py`:

- `test_a_corpus_with_two_byte_identical_documents_loads_and_converges`
- `test_duplicate_record_ids_are_counted_once_across_batches`
- `test_a_duplicated_document_is_still_verifiably_absent_after_retraction`

The first fails on the old code with the `SinkLoadError` quoted above.
