# R7-invariants-1 — `latence purge` never reaches the derived Iceberg table

**Severity:** major
**Scope:** `packages/latence-core/src/latence_core/iceberg.py`,
`packages/latence-core/src/latence_core/delta_runner.py`, `docs/iceberg-export.md`
**Stream:** `feat/release-fleet-integration` (#184 — the optional Iceberg Export output)

## What is wrong

The optional `iceberg:` Export block publishes a **full copy of the corpus** — `content` column
included — into the target table's own location in the catalog warehouse
(`<table location>/data/records-<sha256>.parquet`) and commits a snapshot referencing it.

`DeltaRunner.purge` — the GDPR right-to-be-forgotten path — knows nothing about it. There is no
`iceberg` reference anywhere in `delta_runner.py`. `purge` erases the source file, expunges the
derived records from every Corpus Version, and `remove_tree`s the whole `_latence/runs` tree, then
re-runs the pipeline over the survivors. In the **default** `mode: append`, the re-export appends a
SECOND data file; because `tx.delete(AlwaysTrue())` is issued only when `mode == "overwrite"`
(`iceberg.py:273`), the FIRST data file — the one holding the purged document's text — is still
referenced by the table's **current** snapshot. An ordinary `table.scan()`, not time travel, returns
the purged rows. The bytes also remain on disk in the warehouse forever.

Meanwhile the shipped docs promise the opposite:

* `docs/TUTORIAL-fresh-pod-walkthrough.md:1077` — purged records are "physically removed
  everywhere and cannot be recovered";
* `docs/THREAT-MODEL.md:160` — Purge is "physical removal of the source, all derived records";
* `docs/iceberg-export.md` carries no erasure warning at all.

The Qdrant sink shipped in this same stream solved exactly this case (reconciling sweep +
`verify_absent`). The Iceberg derived sink has no equivalent and no disclosure.

## Failure scenario

1. Operator wires the documented `iceberg:` block (mode defaults to `append`) on the corpus Export
   and runs the pipeline. Data file A — the whole corpus — is published into the warehouse and
   committed.
2. A right-to-be-forgotten request arrives; the operator runs
   `latence purge <stack> --doc sha256:… --yes`. Source file gone, derived records expunged,
   `_latence/runs` wiped.
3. The purge run re-exports the survivors, appending data file B.
4. `table.scan()` still returns the purged document's rows (from A), and A's bytes still sit in the
   warehouse. Exit 0, no warning anywhere.

## Acceptance criteria

1. `IcebergCorpusTable` gains an erasure verb that (a) commits a delete-all so the table's CURRENT
   snapshot references none of the pre-purge data files, and (b) **physically removes** the
   content-addressed corpus copies this framework published under the table's `data/` directory.
2. `DeltaRunner._delete` invokes it on `DeletionMode.PURGE`, for every Export Stage that configures
   an `iceberg` block, **before** the survivor re-run — so the re-run's own registration republishes
   exactly the surviving corpus, in `append` mode as well as `overwrite`.
3. Fail-closed and typed (ADR-0034): a purge that cannot reach the configured catalog raises a
   `ProviderError`/`ConfigError` and does **not** report a successful erasure. A table that was
   never created is a no-op, not an error.
4. Off-by-default is untouched: with no `iceberg` block, `purge` behaves byte-identically to before
   and never imports `pyiceberg` (ADR-0016).
5. `docs/iceberg-export.md` documents the erasure semantics, including the one thing erasure
   necessarily costs: pre-purge snapshots are no longer readable via time travel.
6. Regression test drives the REAL local Iceberg REST catalog end-to-end: register a corpus in
   `append` mode, purge a document, assert `table.scan()` no longer returns it and no warehouse data
   file still contains its text. Must FAIL on the pre-fix code.
