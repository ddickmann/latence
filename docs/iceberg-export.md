# Iceberg-table Export (optional)

The Export's deliverable is Parquet + JSONL on Storage, and that stays true (ADR-0017: the
AI-ready deliverable is *files*, never a database). This page covers the **optional** extra
output added in #184: the same corpus, additionally registered as an **Apache Iceberg table**
in your own Iceberg **REST catalog**, so it is directly governable in Unity Catalog, Polaris,
Glue-via-REST, Lakekeeper or any other REST-compatible catalog — with no glue code.

It is **off by default**. With no `iceberg` block configured, `pyiceberg` is never imported,
no extra file is written, and `records.jsonl` / `records.parquet` are byte-for-byte what they
were before this feature existed.

## Install

```bash
pip install 'latence-core[iceberg]'
```

That pulls `pyiceberg` (Apache-2.0), the Iceberg REST catalog client. It is an *extra*, not a
dependency — `latence-core` itself stays thin (ADR-0016).

## Enable it

Add one block to the corpus Export's config:

```yaml
stages:
  - kind: export
    provider: export.jsonl_parquet
    config:
      basename: records
      iceberg:
        uri: http://localhost:8181      # REQUIRED — your Iceberg REST catalog base URL
        namespace: lake.latence         # dotted string or a list of levels; default "latence"
        table: corpus                   # default: the Export's `basename`
        mode: append                    # append (default) | overwrite
        warehouse: null                 # optional, forwarded to the catalog
        catalog_name: latence           # optional client-side catalog name
        properties: {}                  # optional REST client properties (token, credential, …)
        table_properties: {}            # optional Iceberg table properties set at creation
```

Run the pipeline. The namespace and table are created if they do not exist, and the run's
records appear as a new snapshot.

Any malformed value here raises a `ConfigError` at **wiring** time — before the pipeline reads
a single record — so a typo in the catalog URL never surfaces after a long run.

## What gets registered

* **The schema is the Parquet schema.** The Iceberg table mirrors the documented consumer
  contract column-for-column — `record_id`, `schema_version`, `provenance`, `classification`,
  `content`, `media_type` — plus exactly whichever optional signal columns this Export was
  configured to write (`embedding`, `sparse_indices`/`sparse_values`, `fde_embedding`, the W13
  context columns). There is no second, drifting schema definition.
* **Schema drift against an existing table is refused, in both directions.** If the table
  already exists and this run's columns differ from it — a signal column newly wired *or*
  newly unwired, or a second, differently-configured Export writing the same table — the run
  fails with a `ProviderError` naming the differing columns and the table is left untouched.
  This is not symmetry for its own sake: Iceberg's `add_files` rejects an *extra* column but
  silently back-fills a *missing* one with NULLs, so without this check an unwired embedder
  would quietly NULL out the governed table's vectors (in `mode: overwrite`, all of them)
  while the schema still advertised the column (ADR-0034 — no silent errors).
* **The version stamp matches the Parquet export.** Both the table properties and every
  snapshot summary carry `latence.contract.schema_version` (the contract version this build
  emits — the same constant the Parquet `schema_version` column carries per row) and
  `latence.record.schema_versions` (what the rows of *this run* actually carried, so the table
  never claims a version its own rows do not have), plus `latence.export.source` and
  `latence.export.mode`.
* **The data file is an immutable, content-addressed copy, not `records.parquet` itself.** Each
  run publishes `<table location>/data/<basename>-<sha256>.parquet` — a streamed byte copy of
  the deliverable, named after the digest of its own bytes — and registers *that* with Iceberg's
  `add_files`. This is deliberate: `records.parquet` is rewritten by every run, and an Iceberg
  data file must never change under a snapshot that references it. A content-addressed path
  *cannot*, by construction, ever hold different bytes.

  Naming it by content rather than by a random per-run token also makes the artifact
  **reproducible**: identical corpus bytes yield an identical path, so re-running an identical
  export reuses the one existing data file instead of accumulating a full-corpus copy per run
  (the delta semantics below are unchanged — `append` still adds a snapshot).
* **The copy lands in the table's own location, never in the run tree.** Immutable bytes are only
  half of what a snapshot reference needs; the other half is a **durable path**. The Export writes
  into `<storage_uri>/_latence/runs/<run_id>/export`, and that whole tree is deleted wholesale by
  every Purge (a run tree holds a purged document's raw text in re-derivable artifacts) and by
  ordinary run-retention cleanup. A data file published there was destroyed by an erasure that had
  nothing to do with it, leaving the committed snapshot pointing at a path that no longer exists —
  an unreadable table, and in `append` mode an unrecoverable one, since the re-export lands under a
  new run id while the dangling path stays referenced. The table's location has the right lifetime:
  it is yours, your catalog's retention tools (`expire_snapshots`, `remove_orphan_files`) already
  scan exactly there, and nothing in this framework ever deletes it. A welcome consequence:
  enabling the `iceberg:` block leaves your export directory **byte-for-byte unchanged**.

  The copy is streamed, so the Export's memory stays O(batch) (ADR-0033), and it is published
  temp-then-rename like every other artifact (ADR-0010). Its URI is returned in the Export's
  output URI list.
* **Nothing is re-serialised.** Iceberg reads the very bytes the Parquet writer produced.

## Delta runs — the honest semantics

`mode` decides what a *second* run means. Neither option is magic:

| `mode` | What a run does | Use it when |
|---|---|---|
| `append` (default) | Adds this run's data file as a new snapshot. Rows accumulate. | The Export emitted only the newly affected records (a delta run, ADR-0018/0029). |
| `overwrite` | Deletes all existing rows and adds this run's data file **in one transaction**, i.e. one atomic catalog commit — readers see the old snapshot or the new one, never a gap or an empty table. | The Export emitted the full corpus (a full re-export). |

Two consequences worth stating plainly:

* Running a **full** export twice in `append` mode **duplicates every row**. That is what
  "append" means; pick `overwrite` for full re-exports.
* In `overwrite` mode the superseded data files are left on disk. They are unreferenced by the
  current snapshot but still reachable through Iceberg time travel. Expiring snapshots and
  cleaning orphan files is your catalog's retention policy (`expire_snapshots`), not something
  the Export does behind your back.

## Erasure — what `latence purge` does to this table

`latence purge` is the hard, GDPR right-to-be-forgotten deletion, and it **does** reach this table.
It has to: the data file published here is a full copy of the corpus, chunk `content` and all, and
it lives in your catalog's warehouse — outside the pipeline's `storage_uri`, so none of the purge's
other erasure steps (the source file, the derived records in every committed Corpus Version, the
whole `_latence/runs` tree) touch it.

What a purge does, before it re-runs the pipeline over the survivors:

1. **Commits a delete-all** on the table, so its current snapshot references none of the pre-purge
   data files, and
2. **physically removes** every corpus copy this framework published into the table's `data/`
   directory (the content-addressed `<basename>-<sha256>.parquet` files — nothing else in that
   directory is touched, so a table you share with another engine keeps that engine's files).

The erasure is **total, not row-scoped**. The exported schema is the flat consumer contract, whose
document id lives *inside* the `provenance` JSON string, so there is no column an Iceberg predicate
could filter one document out by. That is safe here only because this table is strictly *derived*
(ADR-0017): the purge's own survivor re-run republishes the surviving corpus onto the emptied table
in the same command — in `append` mode as much as in `overwrite` — so the table converges on the
survivors without you changing `mode`.

Two things to know:

* **Pre-purge snapshots become unreadable.** Their manifests still name data files that no longer
  exist, so time travel to a snapshot older than the purge fails. That is what erasure *is*; a purge
  that left the old snapshots readable would not be one. Run `expire_snapshots` whenever you like to
  drop the stale metadata.
* **A purge fails if it cannot reach the catalog.** If the endpoint is down, or `pyiceberg` is not
  installed while an `iceberg:` block is configured, the purge raises a typed error instead of
  exiting 0 over a warehouse it never reached. A table that was never created is a no-op, not an
  error.

Retraction (the soft, default deletion) does **not** do any of this: it is an audit-preserving
tombstone, and the table converges on the next run under your configured `mode`.

## Failure behaviour

The registration runs **last**, only after the Parquet deliverable has been atomically
published. A crash or an unreachable catalog therefore leaves the normal deliverable complete
and simply no table; the failure surfaces as a `ProviderError` naming the catalog URI, never as
a silent skip. That covers every step of the registration — building the catalog client, creating
the namespace, loading or creating the table, and the commit itself — so a raw `pyiceberg`
exception never reaches your pipeline. The one catalog "failure" that is *not* an error is a lost
create race: if a concurrent Export creates the table first, this run simply loads that table and
registers into it.

**A failed registration leaves no orphan copy.** The data file has to exist before `add_files` can
reference it, so it is published *before* the commit — and if the commit then fails, the Export
deletes it again. That matters because a never-committed file is unreachable by every cleanup you
would reach for: `expire_snapshots` only deletes files recorded in an expired snapshot's manifests,
`remove_orphan_files` could eventually reach it in the table location, but "eventually, if you run
it" is not a disposal guarantee for a full corpus-sized Parquet (chunk `content` and all), and each
failed run against a flaky catalog would drop another one. The error message names the file and
says whether it was removed.

The one deliberate exception is an **unknown commit outcome** (a 500/502/504 on the commit endpoint,
which pyiceberg surfaces as `CommitStateUnknownException`): the catalog may have applied the commit
before the response was lost, and deleting a file a live snapshot references would corrupt the
table. That file is kept, and the `ProviderError` says so — check the table's latest snapshot, then
let your catalog's orphan-file policy decide. Likewise, a file an *earlier* run already committed is
never touched: only a copy this run created is eligible for cleanup.

## Verified against a real catalog

The offline test suite (`packages/latence-core/tests/test_export_iceberg.py`) drives the real
`pyiceberg` REST client over real HTTP against a local REST catalog backed by the OSS
`SqlCatalog` — no container, no network, no cloud.

Beyond that, the feature was run against the **Apache `iceberg-rest-fixture`** reference
catalog (Apache Software Foundation, Apache-2.0) on 2026-07-18:

```bash
docker run -d --name iceberg-rest -p 8181:8181 \
  -v /tmp/iceberg-wh:/tmp/iceberg-wh \
  -e CATALOG_WAREHOUSE=file:///tmp/iceberg-wh \
  -e CATALOG_IO__IMPL=org.apache.iceberg.hadoop.HadoopFileIO \
  apache/iceberg-rest-fixture:latest
```

with `iceberg.uri: http://localhost:8181`. Observed:

```text
export outputs:
   /private/tmp/iceberg-wh/export/records.jsonl
   /private/tmp/iceberg-wh/export/records.parquet
   file:/tmp/iceberg-wh/lake/corpus/data/records-<sha256>.parquet
metadata_location: file:/tmp/iceberg-wh/lake/corpus/metadata/00001-....metadata.json
schema: ['record_id', 'schema_version', 'provenance', 'classification', 'content', 'media_type']
properties: {'latence.contract.schema_version': '16', 'latence.record.schema_versions': '16',
             'latence.export.source': '.../records.parquet', 'latence.export.mode': 'append'}
rows: [('c1', 'alpha text'), ('c2', 'beta text')]
snapshots: 1
after append rows: ['c1', 'c2', 'c3']
after append snapshots: 2
after overwrite rows: ['c9']
```

Note the deployment constraint this makes visible: with `add_files`, the data-file path is
recorded verbatim in the Iceberg manifest, so **the catalog and its readers must be able to
resolve the path the Export wrote**. On a shared object store (`s3://…`) that is automatic; with
a containerised catalog over local files, mount the export directory into the container at the
same path, as above.
