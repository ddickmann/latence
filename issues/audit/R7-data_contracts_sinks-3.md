# R7-data_contracts_sinks-3 — A failed Iceberg data-file *publication* strands a full-corpus temp Parquet, unnamed

**Severity:** major
**Area:** `packages/latence-core/src/latence_core/iceberg.py` (#184, optional Iceberg output)
**Invariants touched:** ADR-0034 (no silent errors — typed errors must name what to sweep), ADR-0044
(corpus `content`/PII must not be stranded in a customer warehouse), docs truth.

## What is wrong

`IcebergCorpusTable._publish_data_file` streams the whole corpus into
`dst_storage.temp_uri(...)` and only then `finalize_atomic`s it to its content-addressed name.
Any failure in that window — a transport drop, ENOSPC, a permission-denied rename — propagates
raw, and the temp file is never deleted.

`register` gates its cleanup on `published_here`, which is still `False` when `_publish_data_file`
raises, so `_discard_unreferenced` never runs; `data_uri` is still the placeholder
`"<not yet published>"`, so the typed error tells the operator nothing about the file that now
exists in their warehouse.

Because `Storage.temp_uri` embeds a fresh random `_token()` per call, **every retry strands another
distinct full-corpus copy** — the unbounded-growth mode the R5 fix (6205245) claimed to close. R5
only covered a failure AFTER the file was finalized.

This directly contradicts:

* `docs/iceberg-export.md` §Failure behaviour — "A failed registration leaves no orphan copy… The
  error message names the file and says whether it was removed."
* `_discard_unreferenced`'s docstring — "the warehouse is left exactly as the failed run found it."

## Failure scenario (reproduced)

Patch `Storage.finalize_atomic` to raise `OSError` only for the `<table>/data/*.parquet`
publication, then run `JsonlParquetExport` with an `iceberg` block over chunks whose content
contains the marker `SECRET`, against a real local Iceberg REST catalog. Result:

```
ProviderError: Iceberg registration of '<not yet published>' into lake.latence.corpus via
'http://127.0.0.1:…' failed: simulated transport drop
```

and the warehouse holds
`…/lake.latence/corpus/data/records.parquet.<random>.tmp` containing `SECRET`. Nothing deletes it,
no snapshot references it, the error names no path, and N retries leave N distinct copies.

## What to build

Make `_publish_data_file` responsible for its own temp copy: on any failure between `temp_uri` and
`finalize_atomic`, best-effort `remove` the temp path and raise a `ProviderError` naming the table,
the catalog, the underlying failure, the temp path and its disposal. No `CommitStateUnknown`-style
exception applies here — the catalog has never been told about a temp name, so deleting it can
never corrupt a table. A sweep that itself fails must ride along on the same typed error (never
mask the original), so the operator learns the exact path still holding corpus text.

## Acceptance criteria

1. A failure during the data-file publication leaves **no** file of any name under the table's
   `data/` directory — temp copies included.
2. Repeated retries against a persistently failing store leave zero accumulated copies.
3. The raised error is a `ProviderError` that never says `<not yet published>`, names the temp
   path, and states that it was removed.
4. When the sweep itself fails, the error names the surviving path, says it could not be removed,
   and still reports the ORIGINAL failure.
5. The Parquet/JSONL deliverable is untouched by any of this (ADR-0017).
6. `docs/iceberg-export.md` §Failure behaviour describes the publication window, not just the
   commit window.
7. No regression: full suite stays at or above 3320 passed / 29 skipped.

## Resolution

Fixed. `_publish_data_file` became an instance method wrapping the copy+finalize in a handler that
delegates to the new `IcebergCorpusTable._publication_failed` (the publication-side twin of
`_discard_unreferenced`). `register`'s `except (ConfigError, ProviderError): raise` comment and
`docs/iceberg-export.md` were corrected to match.

Regression tests (`packages/latence-core/tests/test_export_iceberg.py`):

* `test_a_failed_data_file_publication_leaves_no_orphan_temp_copy` (ACs 1, 2, 3, 5)
* `test_a_failed_publication_whose_sweep_also_fails_names_the_path_to_delete` (AC 4)
