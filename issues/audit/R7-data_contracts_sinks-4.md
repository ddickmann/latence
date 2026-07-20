# R7-data_contracts_sinks-4 — Iceberg cleanup deletes a data file a CONCURRENT run's snapshot references (check-then-act race)

**Severity:** major
**Area:** `packages/latence-core/src/latence_core/iceberg.py` — `IcebergCorpusTable._publish_data_file` / `_discard_unreferenced`
**Stream:** feat/release-fleet-integration (#184, the optional Iceberg Export output)

## What is wrong

`_publish_data_file` decides "did THIS run create the published data file?" with a check-then-act
pair:

```python
data_uri = f"{data_dir}/{basename}-{digest.hexdigest()}.parquet"
if dst_storage.exists(data_uri):
    dst_storage.remove(tmp_uri)
    return data_uri, False        # someone else's file — do not touch it
dst_storage.finalize_atomic(tmp_uri, data_uri)
...
return data_uri, True             # "I created it" → eligible for deletion
```

`register()` uses that boolean as the ONLY gate on `_discard_unreferenced`, whose docstring states
the guarantee outright:

> **A file this run did not create.** … that file belongs to that run's snapshot and is never
> touched (the caller gates on `published_here`).

The gate does not hold under concurrency. Two Exports of **identical** corpus content into the same
table content-address onto the *same* path. If both evaluate `exists(data_uri)` before either
`finalize_atomic` lands, both observe `False`, both rename onto that path, and **both** return
`published_here=True`. If either run's commit then fails, its `_discard_unreferenced` deletes the
shared file — including when the other run has ALREADY committed a snapshot that references it. The
committed table is then permanently broken: the snapshot records the absolute path in its Avro
manifest forever, so `table.scan()` raises `FileNotFoundError` and nothing can repair it. This is
exactly the dangling-data-file failure mode R6 (e35f097) fixed by relocating the file, re-opened
through a different door.

Concurrent Exports into one table are an advertised, supported shape — `_ensure_table` carries
dedicated lost-create-race handling for precisely this.

## Failure scenario (reproduced)

Real `pyiceberg` client against the real local Iceberg REST catalog:

1. Run A exports a corpus in `append` mode and **commits**; its snapshot references
   `…/data/records-<sha256>.parquet`.
2. Run B (identical corpus, started concurrently) evaluates `exists(data_uri)` in the window before
   A's rename lands, so it answers `False`. B re-finalizes the same content-addressed path and sets
   `published_here=True`.
3. B's `add_files` commit fails (catalog 503 / rejected commit).
4. B's `_discard_unreferenced` removes the file.

Observed: `A's committed data file still exists: False`, then
`A's table BROKEN: FileNotFoundError [Errno 2] Failed to open local file '…/data/records-<sha>.parquet'`.
A reported exit 0.

## What to build

Stop trusting a boolean derived from a stale existence check as proof of exclusive ownership, and
make the destructive step answer to the **catalog**, which is the serialization point for what is
referenced:

* Before removing the file, ask the catalog (fresh metadata) whether **any** snapshot of the target
  table references that path — not just the current snapshot, since `overwrite` mode leaves earlier
  snapshots reachable through time travel.
* Referenced ⇒ KEEP the file and say so in the typed error (the same discipline already applied to
  `CommitStateUnknownException`).
* The check itself could not be performed (catalog unreachable, metadata unreadable) ⇒ KEEP the
  file and say so. An unprovable deletion of a full-corpus file is not a cleanup.
* Unreferenced ⇒ delete exactly as today, so R6's no-orphan guarantee is unchanged.

`published_here` stays as the cheap first gate; it is now a necessary, not a sufficient, condition.

Residual (documented, accepted): if the *other* run publishes but has not yet committed when this
run's check runs, the file may still be deleted underneath it. That path fails **loudly** — the
other run's `add_files` reads the Parquet footer and raises, surfacing as a typed `ProviderError`
— rather than silently corrupting a committed table. Closing it entirely would require a
cross-writer lock the Storage seam does not offer.

## Acceptance criteria

1. A run whose commit fails **never** deletes a data file that a committed snapshot of the target
   table references, even when that run's own `exists()` check said the file was absent (i.e. even
   when `published_here` is `True`).
2. After such a failure the earlier run's table still scans back its rows.
3. When the reference check cannot be performed, the file is kept and the typed `ProviderError`
   names the path and the reason (ADR-0034 — no silent errors).
4. R6 is not regressed: a genuinely unreferenced data file from a failed commit is still removed,
   and the typed error still reports the removal.
5. `CommitStateUnknownException` keeps its existing keep-on-purpose behaviour.
6. Regression test fails on the pre-fix code and passes on the fix; `make test` stays at or above
   3320 passed / 29 skipped.
