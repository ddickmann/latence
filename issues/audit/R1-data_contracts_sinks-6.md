# R1-data_contracts_sinks-6 — Iceberg catalog faults escape as raw pyiceberg exceptions

**Severity:** major
**Area:** `packages/latence-core/src/latence_core/iceberg.py` (#184)
**Invariant:** ADR-0034 (no silent errors, no raw native traceback — failures surface as TYPED errors)

## What is wrong

`IcebergCorpusTable.register` normalises only the *commit* transaction. The two lines that reach
the catalog first —

```python
catalog = self._catalog()
table = self._ensure_table(catalog, arrow_schema)
```

— sit **outside** the `try/except` that produces the `ProviderError`. And `_ensure_table` itself
catches only `NoSuchTableError` (around `load_table`) and `NamespaceAlreadyExistsError` (around
`create_namespace`). Every other catalog fault raised by `load_table` / `create_table` therefore
propagates as a raw `pyiceberg.exceptions.*` object straight out of
`JsonlParquetExport.export`.

`docs/iceberg-export.md` claims: "the failure surfaces as a `ProviderError` naming the catalog
URI, never as a silent skip". That is only true for a *construction-time* connection failure or a
*commit* failure — not for any catalog fault in between.

## Failure scenario

Against a real Iceberg REST catalog:

1. **Transient server fault.** A second export run loads the existing table; the catalog answers
   `503` on `GET /v1/namespaces/{ns}/tables/{tbl}` (a routine transient fault).
   `pyiceberg.exceptions.ServiceUnavailableError` escapes `JsonlParquetExport.export` unwrapped —
   not a `ProviderError`, and with no mention of the catalog URI or the table identifier.
2. **Lost create race.** Several exports register concurrently into a not-yet-existing table. Each
   sees `NoSuchTableError` from `load_table` and calls `create_table`; the losers get a raw
   `pyiceberg.exceptions.TableAlreadyExistsError`, even though the correct response is simply to
   re-load the table another writer just created and continue.

Both reproduced end-to-end against the repo's own real REST catalog
(`packages/latence-core/tests/_iceberg_rest.local_rest_catalog`).

## What to build

1. Normalise the whole registration path: `register` wraps catalog acquisition, table
   ensure, data-file publication and the commit, re-raising already-typed `ProviderError` /
   `ConfigError` unchanged and wrapping anything else into a `ProviderError` that names the data
   file, the table identifier and the catalog URI.
2. In `_ensure_table`, wrap `load_table` and `create_table` in their own typed-error normalisation,
   and treat `TableAlreadyExistsError` from `create_table` as a **lost create race**: re-load the
   table and proceed.

## Acceptance criteria

- A catalog fault on table load surfaces from `JsonlParquetExport.export` as a
  `latence_core.errors.ProviderError` whose message names the catalog URI and the table
  identifier, with the raw pyiceberg exception preserved as `__cause__`.
- A `TableAlreadyExistsError` on `create_table` does **not** fail the export: the table is
  re-loaded and the run registers successfully.
- No behaviour change on the happy path; the Export stays byte-identical when `iceberg` is absent.
- Regression tests fail on the pre-fix code and pass after, against the real local REST catalog.
- `make test` stays green at or above 2985 passed / 28 skipped.
