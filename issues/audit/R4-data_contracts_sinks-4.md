# R4-data_contracts_sinks-4 — Iceberg schema drift is only caught in the widening direction

**Severity:** major
**Where:** `packages/latence-core/src/latence_core/iceberg.py` (`IcebergCorpusTable._ensure_table`,
the `tx.add_files` call in `register`); claims at `iceberg.py:277-279` and
`docs/iceberg-export.md:55`; untested in `packages/latence-core/tests/test_export_iceberg.py`
**Tickets/ADRs:** #184 (Iceberg Export output), ADR-0017 (files are the source of truth),
ADR-0034 (no silent errors)

## The defect

The module claimed that "a schema drift between an existing table and this run therefore surfaces
from Iceberg's own `add_files` validation rather than from a guess made here", and the docs claimed
"there is no second, drifting schema definition". `add_files` only validates the **widening** half.
A run that **drops** an optional signal column — the embedder / sparse / FDE / W13 context columns
unwired by a config edit, a second differently-configured Export writing the same table, or a delta
run wired without the signal — is silently ACCEPTED, and Iceberg back-fills the missing column with
NULLs.

In `mode: overwrite` this destroys the whole governed table's signal: every row's `embedding`
becomes NULL while `table.schema()` still advertises the column and the table properties still
carry `latence.contract.schema_version`. Exit code 0, no warning, no typed error.

## Failure scenario (reproduced)

Against the repo's real local Iceberg REST catalog
(`packages/latence-core/tests/_iceberg_rest.local_rest_catalog` — real `pyiceberg` `RestCatalog`
over HTTP onto a real `SqlCatalog`):

1. Run 1: `JsonlParquetExport({'iceberg': {..., 'mode': 'overwrite'}, 'embedder':
   {'provider': 'embedding.hashing', 'config': {'dimension': 8}}})` exports `c1`, `c2`. Table is
   created with an `embedding` column; `table.scan()` yields both rows with real 8-dim vectors.
2. Run 2: same corpus, same `iceberg` block, **embedder block removed**. `records.parquet` no
   longer has an `embedding` column.
3. `add_files` accepts it. The export exits 0, `register` returns normally, and `table.scan()` now
   returns `[('c1', None), ('c2', None)]` — every row's embedding NULL — while `table.schema()`
   still lists `embedding`.

A downstream consumer doing vector search over the governed table gets nulls for the entire corpus,
with no error anywhere. The mirror case (run 1 without embedder, run 2 with) correctly raised
`ProviderError('PyArrow table contains more columns: embedding, … use union_by_name')`, which is
why the gap read as covered. No test in `test_export_iceberg.py` exercised drift in either
direction.

This violates #184's AC "The table schema mirrors the documented consumer contract" / "Delta runs
produce consistent table state" and ADR-0034 (no silent errors).

## What was built

`IcebergCorpusTable._assert_no_drift(table, arrow_schema)`: whenever an EXISTING table is loaded
(both the normal load path and the load-after-a-lost-create-race path), the table's column names are
compared to the run's Arrow schema column names. Any difference — missing columns, unexpected
columns, or a differing order — raises a typed `ProviderError` naming the identifier, the catalog
URI and the specific columns, **before** the transaction opens, so the existing table is left
untouched. Type-level drift stays with Iceberg's own validation, which does catch it. The module
docstring and `docs/iceberg-export.md` were corrected to state the real behaviour.

## Acceptance criteria

- [x] A second run into an existing table that DROPS a signal column fails with a `ProviderError`
      naming the column and the table, instead of NULL-back-filling.
- [x] The existing table is unchanged after the refusal (previous rows and their vectors intact),
      and the Parquet deliverable is still written (ADR-0017: the file is the source of truth).
- [x] The mirror direction (a run that ADDS a column) also surfaces as a typed `ProviderError`.
- [x] A run whose columns match the table exactly still registers (no false refusal): the existing
      append / overwrite / lost-create-race / schema-mirror tests stay green.
- [x] Off-by-default remains byte-identical; no new `latence-core` dependency (ADR-0016).
- [x] Regression tests in `packages/latence-core/tests/test_export_iceberg.py`:
      `test_a_run_that_drops_a_signal_column_is_refused_not_null_backfilled` (fails pre-fix with
      "DID NOT RAISE ProviderError") and
      `test_a_run_that_adds_a_signal_column_is_refused_with_a_typed_error`.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.
