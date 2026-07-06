# latence-core

The thin, near-zero-dependency spine of the [Latence framework](../../README.md): data
contracts, Capability protocols, the declarative Pipeline, the local Runner, and the
fsspec Storage seam. Providers (parsers, NER, etc.) ship as separate packages
(ADR-0016) and are discovered via the `latence.providers` entry-point group.

## Install

```bash
pip install latence-core
```

Dependencies are deliberately minimal: `pydantic`, `fsspec`, `pyarrow`, `pyyaml`, `typer`.

## Run a pipeline

Declare a Pipeline in YAML (`pipeline.yaml`):

```yaml
name: walking-skeleton
storage_uri: file:///tmp/latence-demo
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: file:///path/to/docs
  - name: parse
    capability: parse
    provider: parser.plaintext        # from latence-parser-plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
```

Then:

```bash
latence run pipeline.yaml
```

This reads a folder of `.txt`/`.md` documents (Source, stamping Provenance and
Classification), passes them through Parse, and Exports `records.jsonl` +
`records.parquet` — emitting a **Quality Report** to
`<storage_uri>/_latence/runs/<run_id>/quality-report.json`.

## Guarantees

- **Contracts**: every record carries `Provenance` and `Classification`, validated
  at every Stage boundary (ADR-0006). Missing either raises `ValidationError`.
- **Storage**: `file://`, `s3://`, `gs://`, `memory://` — no code change (ADR-0009).
- **Resume**: the local Runner checkpoints each Stage atomically (WAL discipline).
  Killing a run mid-way and re-running resumes without redoing completed Stages;
  re-running a finished pipeline is a no-op (ADR-0010).
