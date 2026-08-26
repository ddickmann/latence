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

## Disambiguation (S8)

**Disambiguation** is the corpus-level Stage — the algorithmic crown — that links
entity mentions to canonical entities and merges duplicates across documents, with
auditable, confidence-weighted merge decisions (no silent over-merge) and Evidence
back to source mentions. It is inserted after Entity Extraction (required) and
optionally after Relation Extraction as a `disambiguation` Stage (see
`examples/disambiguation-slice.yaml`). It emits first-class `DisambiguationRecord`s at
two scopes:

- one **ENTITY**-scope record per resolved `CanonicalEntity` — the cross-document
  merged cluster, its `member_mention_ids`, an audited `merge_decisions` log (each with
  its resolver strategy, confidence, `applied` flag, and `Evidence`), an optional
  external-KB `kb_id` (with a graceful fallback for the unlinked), and **page-accurate
  Provenance** inherited from the representative mention;
- one **RELATION**-scope record per `NormalizedRelation` — a relation whose endpoints
  are remapped to canonical entities and whose label is normalised (fuzzy/alias,
  type-filtered, inverse-detected + direction-swapped).

The ported algorithmic assets (all taken as-is) are re-homed in the
pure `latence_core.disambiguation` engine:

- the **EntityResolver** 5-strategy cascade — exact → alias → acronym → substring →
  embedding — with a **Union-Find** merge map (lexicographic-min canonical) and a
  **confidence-weighted, audited merge log**: a proposed merge below the confidence
  policy is recorded with `applied=False` but **never folded** (the "no silent
  over-merge" guarantee);
- the **GLinker** linking cascade — L2 candidate-gen → L3 biencoder → L4 rerank → L0
  threshold — with a **KB-text fallback** (an unmatched mention stays an unlinked
  canonical entity, not a forced wrong link);
- the **RelationNormalizer** — alias/fuzzy label resolution, type filtering, and
  **inverse detection + direction swap**.

The CPU reference path is required and works with **no embedder and no KB** (the lexical
token-similarity stand-in and the empty-KB fallback); a FAISS/GPU blocking Provider is a
pluggable option behind the same `Disambiguator` seam, **not** required (ADR-0026).

- **`disambiguation.cascade`** ships in core (pure-Python, deterministic, no model
  download). Config: `aliases`, `min_confidence` (0.82), `alias_confidence` (0.95),
  `acronym_confidence` (0.9), `embedding_threshold` (0.75), `min_substring_ratio` (0.5),
  `enable_embedding` (true); `kb` + `kb_threshold` (0.5) for linking; `label_aliases`,
  `inverses`, `allowed_labels`, `label_threshold` (0.6) for relation normalisation.

  ```yaml
    - name: disambiguate
      capability: disambiguation
      provider: disambiguation.cascade
      depends_on: [entities, relations]   # relations is optional (entity-only otherwise)
  ```

The merge/link statistics — mentions in vs canonical entities out, the
created/merged/singleton breakdown, applied vs below-policy merges by strategy,
KB-linked vs unlinked, relations normalised — land in the Quality Report's
`disambiguation` section (ADR-0026).

## Graph Assembly + Export (S9)

**Graph Assembly** is the corpus-level Stage that builds the canonical knowledge
graph — nodes, edges, and per-edge **Evidence** — from the disambiguated canonical
entities and normalized relations. It is wired after Disambiguation as a
`graph_assembly` Stage (see `examples/graph-slice.yaml`) and emits first-class
`GraphRecord`s at two scopes: one `NODE`-scope record per canonical entity (a graph
node with a **deterministic, content-addressed** id, its member mentions + source
documents, an optional external-KB link, page-accurate Provenance, and Evidence back
to the source mentions) and one `EDGE`-scope record per normalized relation (a graph
edge with a deterministic id, its head/tail node ids, the normalized label, and
**per-edge Evidence** — every edge carries Evidence). An edge is emitted only when both
endpoints resolved to a node (no dangling edges).

The ported **Graph builder** (`dataset-intelligence/graph/builder.py`, PORT-AS-IS) is
re-homed in the pure `latence_core.graph` engine: content-addressed sha256 ids
(`make_graph_id`), endpoint validation, and Evidence links, with the `max_evidence_per_node`
bound for hub entities.

**Export** materializes the two AI-ready deliverables as **portable files on Storage,
never a live database** (ADR-0017):

- **`export.knowledge_graph`** ships in core: the KG as **Parquet + TTL + GraphML**
  (nodes + edges + per-edge Evidence), loadable into Neo4j, a triple store, or a
  graph-RAG engine. Hand-emitted with no `rdflib`/`networkx` dependency; byte-stable.
  Config: `basename` (`graph`).
- **`export.jsonl_parquet`** (the S1 exporter) writes the **RAG-ready corpus** — the
  cleaned/chunked/Screened/PII-handled text with full Provenance/Classification — as
  JSONL + Parquet.

Embeddings are **optional** (ADR-0017): the corpus exports its text whether or not an
Embedder is wired. **`embedding.hashing`** ships in core as a deterministic, dependency-free
reference `Embedder`; the learned default (IBM Granite Embedding r2, Apache-2.0) is a pluggable
Provider package, not a core dependency.

```yaml
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
```

The KG statistics — node/edge counts, nodes by type, edges by label, KB-linked node
count, and **evidence coverage** (the fraction of edges carrying Evidence) — land in the
Quality Report's `graph` section (ADR-0027). This slice completes the first end-to-end
messy-docs → KG demo.

## S10 — Quality Report + gold-set harness

The **Quality Report** (CONTEXT) is promoted to the full first-class artifact (ADR-0008):
every run writes both a machine-readable `quality-report.json` and a human-readable
`quality-report.md` (the rendered summary) under `<storage_uri>/_latence/runs/<run_id>/`.
It carries per-Stage metrics, the per-Stage quality sections S2–S9 added, **contract
completeness** (Provenance/Classification present **and offsets aligned**), and a
corpus-level **drift** roll-up (`PageOffsetIndex` recoveries across every derived offset;
`clean` = every offset resolves exactly to its source page).

The **gold-set harness** makes the "AI-ready" claim auditable on the adopter's *own* corpus.
Drop in an annotation file — for each source document, the entities and relations that
*should* be extracted, by label + char span — and point the Pipeline at it:

```yaml
name: quality-slice
storage_uri: file:///tmp/latence-quality-demo
gold_set_uri: file:///path/to/gold.json     # optional; omit and the section is skipped
stages:
  - { name: source, capability: source, provider: source.local_folder, config: {path: 'file:///docs'} }
  - { name: parse, capability: parse, provider: parser.plaintext, depends_on: [source] }
  - { name: chunk, capability: chunk, provider: chunk.markdown, depends_on: [parse] }
  - { name: entities, capability: entity_extraction, provider: entity.gazetteer, depends_on: [chunk], config: {labels: {ORG: {terms: [Acme]}, PERSON: {terms: [Alice, Bob]}}} }
  - { name: relations, capability: relation_extraction, provider: relation.pattern, depends_on: [chunk, entities], config: {relations: {works_for: {head: PERSON, tail: ORG, triggers: [works at, joined]}}} }
  - { name: export, capability: export, provider: export.jsonl_parquet, depends_on: [relations] }
```

```json
{
  "documents": [
    {"file_name": "memo.txt",
     "entities": [{"label": "PERSON", "char_start": 0, "char_end": 5},
                  {"label": "ORG", "char_start": 15, "char_end": 19}],
     "relations": [{"label": "works_for",
                    "head": {"label": "PERSON", "char_start": 0, "char_end": 5},
                    "tail": {"label": "ORG", "char_start": 15, "char_end": 19}}]}
  ],
  "match_min_overlap": 0.0
}
```

The harness matches each run's extracted mentions/relations against the annotations
(label + span-overlap, greedy **one-to-one**, a relation credited only when *both*
endpoints align) and writes **precision / recall / F1 per Stage — overall and per label —**
into the report's `goldset` section, with `documents_matched / documents_annotated` support.
`match_min_overlap` (0.0 = any overlap, 1.0 = exact span) is the single tuning knob. It is a
Runner-side evaluation, not a DAG Stage (ADR-0028) — deterministic and seeded, so a run is
byte-identical; a pipeline with no `gold_set_uri` pays nothing. See
`examples/quality-slice.yaml`.

## Incremental deltas (S11)

The corpus is not rebuilt on every change. A **Delta** is a content-hash diff of the Source
manifest against the last committed **Corpus Version**; a delta run transactionally produces an
immutable **Version N+1** (CONTEXT `Corpus Version` / `Delta` / `Affected set` / `Reconciliation` /
`Retraction` / `Purge`; ADR-0018 / ADR-0029). You do not change the Pipeline — you change how you
run it: drive the same pipeline with the **`DeltaRunner`** (which wraps the `LocalRunner`) instead
of `LocalRunner().run(...)`.

```python
from latence_core import DeltaRunner, Pipeline, Storage

pipeline = Pipeline.load("examples/delta-slice.yaml", Storage.for_uri(store))
dr = DeltaRunner()                                  # drift_threshold=0.5 by default
dr.apply_delta(pipeline, run_id="v0")               # first run -> Corpus Version 0
# ...add / change / remove files under the source folder...
dr.apply_delta(pipeline, run_id="v1")               # only the affected set recomputes
dr.retract(pipeline, "v2", document_ids=[doc_id])   # soft delete: tombstone + re-resolve (audit-retained)
dr.purge(pipeline,   "v3", document_ids=[doc_id])   # hard GDPR erase: source + derived + Evidence gone
```

On each `apply_delta` the DeltaRunner:

- **detects the delta** — a content-hash diff of the Source manifest classifies every document
  added / updated / deleted / unchanged (ported `delta_detector`);
- **recomputes only the affected set** — doc-level Stages re-run on changed docs; corpus-level
  Stages (Disambiguation, Graph Assembly) recompute only the blocking neighborhoods the delta
  touched, catching **merge-on-add** (a bridging doc unites two clusters) and **split-on-delete**
  (its removal splits them). Untouched records are carried forward verbatim;
- **triggers a full Reconciliation** when a cheap drift metric (fraction of the corpus changed)
  crosses `drift_threshold` — the bounded-error backstop;
- **commits an immutable Corpus Version** via a write-ahead log + atomic manifest swap (ported
  `write_ahead_log`): re-running the same delta is **idempotent**, a failed run leaves Version N
  intact, and any prior Version is inspectable for audit and rollback (`CorpusStore.read_version` /
  `read_records`);
- emits the **delta churn** (docs +/~/-, entities created/merged/split, edges +/-, records
  retracted/purged, drift, reconciliation) into the run's Quality Report `delta` section.

Deletion is **dual-mode**: **Retraction** (the default) soft-tombstones the deleted documents'
derived records — excluded from every Export / KG / RAG corpus but retained for audit and
rollback — and re-resolves affected clusters; **Purge** physically erases the source, all derived
records, and their Evidence (GDPR / right-to-be-forgotten), then reconciles. Both are surgical via
Evidence provenance (ported `graph/patcher`). Corpus Versions live under
`<storage_uri>/_latence/corpus/`. The in-core reference `DeltaProcessor` is `delta.affected_set`
(pure-Python, deterministic, CPU-viable); a distributed blocking Provider is a pluggable upgrade
behind the same seam. See `examples/delta-slice.yaml`.
