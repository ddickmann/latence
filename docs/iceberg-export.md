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
* **The version stamp matches the Parquet export.** Both the table properties and every
  snapshot summary carry `latence.contract.schema_version` (the contract version this build
  emits — the same constant the Parquet `schema_version` column carries per row) and
  `latence.record.schema_versions` (what the rows of *this run* actually carried, so the table
  never claims a version its own rows do not have), plus `latence.export.source` and
  `latence.export.mode`.
* **The data file is an immutable copy, not `records.parquet` itself.** Each run publishes
  `<out_dir>/iceberg-data/<basename>-<token>.parquet` — a streamed byte copy of the
  deliverable — and registers *that* with Iceberg's `add_files`. This is deliberate:
  `records.parquet` is rewritten by every run, and an Iceberg data file must never change
  under a snapshot that references it. The copy is streamed, so the Export's memory stays
  O(batch) (ADR-0033), and it is published temp-then-rename like every other artifact
  (ADR-0010). Its URI is returned in the Export's output URI list.
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

## Failure behaviour

The registration runs **last**, only after the Parquet deliverable has been atomically
published. A crash or an unreachable catalog therefore leaves the normal deliverable complete
and simply no table; the failure surfaces as a `ProviderError` naming the catalog URI, never as
a silent skip. That covers every step of the registration — building the catalog client, creating
the namespace, loading or creating the table, and the commit itself — so a raw `pyiceberg`
exception never reaches your pipeline. The one catalog "failure" that is *not* an error is a lost
create race: if a concurrent Export creates the table first, this run simply loads that table and
registers into it.

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
   /private/tmp/iceberg-wh/export/iceberg-data/records-1026d87e84fd4315917cbcf80b65fcf8.parquet
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
