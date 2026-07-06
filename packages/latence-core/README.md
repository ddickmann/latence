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

## Chunk + Screening (S3)

Three more built-in Capabilities ship in core (pure-Python, no extra deps),
inserted into the Pipeline as `intake_screening` → `parse` → `chunk` →
`content_screening` → `export` (see `examples/chunk-screening-slice.yaml`):

- **Chunk** (`chunk.markdown`) splits parsed markdown into retrieval-sized
  `ChunkRecord`s while preserving offsets, page alignment, Provenance and
  Classification losslessly. It ports the orchestrator asset (18-pattern markup
  stripping + token budgeting); every chunk's Provenance `char_start/char_end`
  indexes the **original** markdown and its `page_start/page_end` resolve through
  `PageOffsetIndex`, so a chunk always maps back to its source page. Config:
  `max_tokens` (512), `overlap_tokens` (64), `min_tokens` (16).

- **Intake Screening** (`screening.intake_signature`, before Parse) Quarantines
  oversized files, file-type spoofs (magic-byte mismatch), and zip bombs (judged
  from the archive's central directory — no member is decompressed). A Quarantined
  document never reaches Parse. Config: `max_bytes`, `max_zip_ratio`,
  `max_zip_uncompressed`, `check_spoof`.

- **Content Screening** (`screening.content_keyword`, after Chunk) flags
  prompt-injection / harmful chunks with a `RiskMarker` that **survives into the
  exported corpus** (a RAG consumer filters on `risk_markers`), or escalates a hit
  to Quarantine when `quarantine_threshold` < 1.0.

Quarantined records are removed from the corpus but **retained + inspectable**
under `<run>/quarantine/<stage>.jsonl`; per-record decisions land under
`<run>/screening/<stage>.jsonl`; and quarantine dispositions/reasons plus chunk
and screening metrics roll up into the Quality Report's `chunk` and `screening`
sections (ADR-0021).
