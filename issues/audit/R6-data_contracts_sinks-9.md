# R6-data_contracts_sinks-9 — The Iceberg data file is published inside the run-artifact tree that Purge deletes

- **Severity:** major
- **Dimension:** data_contracts_sinks
- **Branch:** feat/release-fleet-integration
- **File:** `packages/latence-core/src/latence_core/iceberg.py`
  (`IcebergCorpusTable._publish_data_file`, `data_dir = f"{out_dir}/iceberg-data"`)
- **Interacts with:** `packages/latence-core/src/latence_core/delta_runner.py`
  (`DeltaRunner._purge_run_artifacts` → `storage.remove_tree(run_entry)`) and
  `packages/latence-core/src/latence_core/runner.py` (`LocalRunner._export_dir` / `run_dir`)
- **Status:** fixed

## What is wrong

`_publish_data_file` published the catalog's data file at
`<out_dir>/iceberg-data/<basename>-<sha256>.parquet`, and `out_dir` is always
`<storage_uri>/_latence/runs/<run_id>/export` (`LocalRunner._export_dir`, runner.py:432-443,
467-468). That is the single most ephemeral directory in the system:
`DeltaRunner._purge_run_artifacts` — the GDPR-erasure step of **every** purge, for **any**
document — iterates `storage.ls(<storage_uri>/_latence/runs)` and `remove_tree`s every entry.

The committed Iceberg snapshot records that file's **absolute path verbatim** in its Avro manifest.
So a purge of a completely unrelated document permanently broke the registered table.

The module docstring justified the copy entirely in terms of *byte immutability* ("the catalog gets
content that can never move under it") and never reasoned about the **lifetime of the path** — then
placed the file in the one directory the framework itself destroys on a routine operation. Ordinary
run-retention cleanup breaks it the same way.

## Failure scenario

1. Export with the optional `iceberg:` block into a normal run:
   `<storage_uri>/_latence/runs/run-1/export`. The snapshot commits, referencing
   `.../runs/run-1/export/iceberg-data/records-<sha>.parquet`.
2. Any later Purge (any document, unrelated) calls `_purge_run_artifacts`, which `remove_tree`s
   `.../runs/run-1` wholesale.
3. `table.scan()` now raises
   `FileNotFoundError: Failed to open local file '.../export/iceberg-data/records-<sha>.parquet'`.
   The governed table is unreadable.
4. It is also **unrecoverable**: the post-purge re-export lands under a NEW `run_id`, so in
   `append` mode (the default) the old dangling path stays referenced by the table forever. Nothing
   in the purge path is aware the catalog exists.

Observed pre-fix on the real purge method with the real run layout — see the regression test below.

## What to build

Publish the immutable, content-addressed data file into the **target table's own location**
(`<table.location()>/data/`) instead of the Export's `out_dir`. That is the only path with the
lifetime a snapshot reference needs: it belongs to the catalog, whose operator already owns
retention for it (`expire_snapshots` / `remove_orphan_files` scan exactly there), nothing in this
framework deletes it, and it is where every other Iceberg writer puts its data files.

The warehouse may be a different filesystem than the Export's, so the data file gets its own
`Storage` bound to that location's scheme, and the copy streams source→destination (still O(chunk),
ADR-0033). Failures resolving or writing that location surface as typed errors naming the table
(ADR-0034), never a raw traceback.

Consequence worth stating: the export deliverable is now byte-for-byte unchanged whether or not the
`iceberg:` block is configured — the optional output adds nothing to `export/` at all.

## Acceptance criteria

1. After the **real** `DeltaRunner._purge_run_artifacts` wipes the run tree an Export wrote into,
   the registered table still scans back its rows.
2. The published data-file URI is under `<table.location()>/data/`, and no Iceberg artifact is
   written anywhere under `_latence/runs/`.
3. Content addressing, streamed hashing (O(chunk)), the skip-if-identical rule and the
   temp-then-rename publication are all preserved.
4. `append` / `overwrite` delta semantics are unchanged; failed-commit cleanup still deletes only a
   file THIS run published and still keeps the file on an unknown commit outcome.
5. Determinism is asserted where the artifact now lands: two independent runs over identical
   records content-address onto ONE path; the export tree compares equal under the real
   `stacks.validate._export_artifacts` and contains no Iceberg entry.
6. OFF-BY-DEFAULT stays byte-identical and `pyiceberg` is still never imported when the block is
   absent.
7. A regression test fails on the old code and passes on the fix; `make test` stays green at or
   above 3320 passed / 29 skipped.

## Regression test

`packages/latence-core/tests/test_export_iceberg.py::test_a_purge_of_the_run_artifacts_leaves_the_registered_table_readable`
— exports into the real run layout, drives the real `DeltaRunner._purge_run_artifacts`, asserts the
run tree is really gone and the table still scans. Pre-fix it failed with the `FileNotFoundError`
above.
