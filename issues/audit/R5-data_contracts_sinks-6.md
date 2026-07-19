# R5-data_contracts_sinks-6 — A failed Iceberg commit leaves a full corpus-sized orphan Parquet copy inside the Export deliverable, reachable by no cleanup

- **Severity:** major
- **Dimension:** data contracts & sinks (ADR-0017 derived-not-authoritative; ADR-0034 no silent
  errors; ADR-0010 temp-then-rename)
- **Branch:** feat/release-fleet-integration
- **File:** `packages/latence-core/src/latence_core/iceberg.py`
  (`IcebergCorpusTable.register` / `_publish_data_file`), `docs/iceberg-export.md`
- **Status:** fixed

## What is wrong

`register` publishes the Iceberg data file **before** it opens the transaction — it has to, since
`add_files` can only reference a file that already exists — and nothing removes that file when the
transaction fails:

```python
data_uri = self._publish_data_file(storage, parquet_uri, out_dir, basename)  # finalize_atomic
...
with table.transaction() as tx:      # raises → except → ProviderError → return
    tx.add_files([...])
```

The published file is a byte-for-byte copy of the whole corpus, chunk `content` included. Once the
commit has failed it is referenced by no manifest, and it is unreachable by every cleanup an
operator would reach for:

* `expire_snapshots` deletes only files recorded in an **expired snapshot's manifests** — a
  never-committed file is in no manifest;
* `remove_orphan_files` scans the **table location** (the catalog warehouse), while this file lives
  in the Export's `out_dir`.

`docs/iceberg-export.md` compounds this by discussing only the *superseded* (time-travel-reachable)
files and naming `expire_snapshots` as the remedy — which does not apply here.

This also hollows out the ADR-0010 discipline the method's own docstring invokes: temp-then-rename
prevents a *torn* file, but the failure mode here is a fully-written orphan.

## Failure scenario

Against the repo's real REST-catalog harness (`packages/latence-core/tests/_iceberg_rest`) with a
fault injector answering 503 on the **commit** endpoint only (`POST /v1/namespaces/<ns>/tables/
<table>`), one export run raises `ProviderError` and leaves `<out_dir>/iceberg-data/records-<sha256>
.parquet` on disk — a full corpus copy that never entered the table and that nothing will ever
collect. Every subsequent failed run against a differing corpus adds another. No existing test sees
this: the one fault test (`test_a_catalog_fault_on_table_load_surfaces_as_a_typed_provider_error`)
faults on table **load**, which happens before the copy is made.

(The stream's earlier fix to content-addressing bounds the *identical-bytes retry* case onto one
path, so this is orphan-per-distinct-corpus, not orphan-per-retry — but the orphan itself is real,
unreachable, and corpus-sized.)

## Acceptance criteria

1. A registration whose commit definitively did not land removes the data file **this run
   published**; `<out_dir>/iceberg-data/` is left with no file the catalog does not reference.
2. A file an **earlier** run already committed (reached by content addressing) is never deleted —
   cleanup is scoped to what this run actually created.
3. An **unknown** commit outcome (`CommitStateUnknownException`; pyiceberg's mapping of 500/502/504
   on the commit endpoint) **keeps** the file — the commit may have landed and deleting a
   referenced file would corrupt the table — and the typed error says so explicitly.
4. The `ProviderError` names the data file and states what happened to it; a failure to delete is
   appended to that same error, never swallowed and never allowed to mask the original failure
   (ADR-0034).
5. `docs/iceberg-export.md` documents this behaviour, including the deliberate unknown-state
   exception, and stops implying `expire_snapshots` covers it.
6. Regression tests drive the **real** REST-catalog harness with a commit-time fault, and fail on
   the pre-fix code.
7. No regression: `make test` stays green at or above 2985 passed / 28 skipped.

## Tests

`packages/latence-core/tests/test_export_iceberg.py`:

* `test_a_failed_commit_leaves_no_orphan_data_file_in_the_deliverable` (fails pre-fix)
* `test_an_unknown_commit_outcome_keeps_the_data_file_on_purpose` (fails pre-fix)
* `test_a_failed_commit_never_deletes_a_data_file_an_earlier_run_committed` (guards AC 2)
