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

## Entity Extraction (S4)

**Entity Extraction** finds typed entity mentions in chunked text (zero-shot NER),
inserted after Chunk as an `entity_extraction` Stage (see
`examples/entity-slice.yaml`). Each mention is a first-class `EntityMention` record
carrying its label, a confidence, and the char span it occupies **in the parent
document's assembled markdown** — so a mention resolves back through the parent's
page map to its original source page (the round-trip: mention → offset → page). The
label set is **zero-shot**: supplied per Pipeline in `config`, nothing hardcoded in
core.

- **`entity.gazetteer`** ships in core (pure-Python, deterministic, no model
  download): a zero-shot gazetteer/regex extractor. Config: `labels` (a mapping of
  entity type → `{terms: [...], patterns: [...], confidence: float}`),
  `default_confidence` (0.9), `case_sensitive` (false), `min_chars` (1).

  ```yaml
    - name: entities
      capability: entity_extraction
      provider: entity.gazetteer
      depends_on: [chunk]
      config:
        labels:
          PERSON: { terms: [Alice, Bob] }
          ORG:    { terms: [Acme Corp], confidence: 0.95 }
          EMAIL:  { patterns: ['[\w.+-]+@[\w.-]+\.\w+'] }
  ```

- **`entity.gliner`** (in the separate `latence-ner-gliner` package) is a learned
  zero-shot Provider over a GLiNER checkpoint via `transformers`/ONNX (CPU,
  Apache-2.0 weights+code). It is a one-line swap — `provider: entity.gliner`,
  `config.labels: [person, organization, ...]` — no pipeline change (ADR-0016).

Per-type mention counts and the confidence distribution land in the Quality
Report's `entities` section (ADR-0022).

## Profiling (S7)

**Profiling** is the corpus-level Stage that computes statistical and quality
features over the corpus, inserted after Parse (and optionally after Entity
Extraction) as a `profiling` Stage (see `examples/profiling-slice.yaml`). It emits
first-class `FeatureRecord`s at two scopes:

- one **DOCUMENT**-scope record per parsed document — `char_count`, `word_count`,
  `sentence_count`, `type_token_ratio`, `density`, a normalised `readability`, the
  fitted `zipf_alpha`, a zlib `compression_ratio` (a redundancy proxy), and the
  average word/sentence lengths;
- exactly one **CORPUS**-scope record — corpus word/vocabulary counts and Zipf α,
  **source coverage** by language / category / sensitivity, cross-document
  **entity frequency** and **entity-type frequency**, per-entity **type consensus**
  (the fraction of an entity's mentions that agree on its label), and the top
  cross-document **entity co-occurrence** pairs.

The feature set is **fixed and closed** (the private stack's 9 enrichers consolidated,
the dynamic registry recast away, ADR-0025), so a seeded run is byte-identical. The
pass is **streaming** — documents are folded in one at a time, with no
whole-corpus-in-RAM assumption.

- **`profiling.statistical`** ships in core (pure-Python, deterministic, no model
  download — the Zipf fit and readability are closed-form). Config:
  `max_co_occurrences` (50), `corpus_sensitivity` (`internal`), `corpus_category`
  (`corpus_profile`).

  ```yaml
    - name: profile
      capability: profiling
      provider: profiling.statistical
      depends_on: [parse, entities]   # entities is optional (Parse-only = text-only profile)
  ```

The feature summary — per-document averages plus the corpus aggregates — lands in the
Quality Report's `profiling` section (ADR-0025).
