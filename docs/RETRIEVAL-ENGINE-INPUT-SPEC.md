# Retrieval-Engine Input Specification

**What the `latence-framework` pipeline delivers, as a build contract for a separate retrieval-engine team.**

Branch this spec was written against: **`w13-context-enrichment`** (the branch that carries `ChunkRecord.context_header`; `wave-integration` and `main` do not yet). Worktree: `/Users/dennisdickmann/Developer/latence-framework-w13`.

Schema-version constants, read from source at time of writing:

| Constant | Value | Defined in |
|---|---|---|
| `SCHEMA_VERSION` (all record contracts) | **15** | `packages/latence-core/src/latence_core/contracts.py:150` |
| `QUALITY_SCHEMA_VERSION` (Quality Report) | **19** | `packages/latence-core/src/latence_core/quality.py:97` |
| `PIPELINE_SCHEMA_VERSION` | **2** | `packages/latence-core/src/latence_core/pipeline.py:25` |

Every record carries its `schema_version` field (serialized with the record). Contracts are **additive and versioned**: new fields default to `None`/empty so an older consumer keeps validating (e.g. v11 `offset_map`, v14 `induced_labels`, v15 `context_header` are all optional additive fields — see the changelog banner at `contracts.py:17-149`).

---

## 1. Overview

Per corpus run, the pipeline emits three deliverables as **open columnar/graph/JSON files on any fsspec backend** — never a live database (ADR-0017):

1. **A chunk-level RAG corpus** — redacted text + `Provenance` + `Classification` + per-chunk risk markers / induced labels / a KG `context_header` + (optional) a single dense `embedding` vector. Files: `records.jsonl` + `records.parquet`.
2. **An evidence-linked knowledge graph** — nodes + edges, every edge citing the source mentions/documents that justify it. Files: `graph-nodes.parquet`, `graph-edges.parquet`, `graph.ttl`, `graph.graphml`.
3. **A Quality Report** — per-Stage metrics, corpus KG statistics, contract-completeness, optional gold-set precision/recall. Files: `quality-report.json` + `quality-report.md`.

**Exact output tree** (`Storage` layout; `run_dir` = `runner.py:385-391`, `_export_dir` = `runner.py:419-420`):

```
<storage_uri>/_latence/runs/<run_id>/
├── export/
│   ├── records.jsonl          # RAG corpus (streaming, full record dump)
│   ├── records.parquet        # RAG corpus (flat columnar)
│   ├── graph-nodes.parquet    # KG nodes
│   ├── graph-edges.parquet    # KG edges
│   ├── graph.ttl              # KG as RDF Turtle
│   └── graph.graphml          # KG as GraphML (yEd/Gephi/networkx)
├── quality-report.json        # machine-readable report  (runner.py:1075)
├── quality-report.md          # human-readable report     (cli.py:112)
├── checkpoints/<stage>.jsonl  # run state (not a deliverable)
├── screening/<stage>.jsonl    # screening findings sidecar (not a deliverable)
├── quarantine/<stage>.jsonl   # withheld records (not a deliverable)
├── manifest.json / wal.log    # run state (not a deliverable)
```

> **File-name note:** the corpus basename defaults to `records` (`JsonlParquetExport.__init__`, config key `basename`) and the graph basename to `graph` (`KnowledgeGraphExport.__init__`, config key `basename`). This document uses the defaults `records.*` / `graph-*`. Both exporters write into the **same** `export/` directory, so they coexist without collision. (This spec's task brief calls them `corpus.jsonl`/`corpus.parquet`; on disk they are `records.jsonl`/`records.parquet` unless the operator sets `basename: corpus`.)

The corpus Export materializes **whatever inter-Stage `Record` stream is fed to the Export sink** (`runner.py:1427-1434`) — on a Parse→Chunk→…→Export pipeline that is the `ChunkRecord` stream (optionally the `RedactionRecord` masked variant). `EntityMention`/`RelationMention`/`DisambiguationRecord` are themselves exportable `Record`s but are **not** part of the default chunk corpus file; the entity/relation layer is delivered to the engine through the **knowledge-graph files** and the per-chunk **`context_header` join keys** (see §3, §5). See §8 for the boundary.

---

## 2. Every output artifact

All nested contract fields (`Provenance`, `Classification`, `ContextHeader`, `Evidence`, …) are serialized as **sorted-key JSON strings** inside Parquet columns to keep the Parquet schema flat (ADR-0006). JSONL rows are the **full Pydantic `model_dump(mode="json")`** — every field, nested objects intact.

### 2.1 `records.jsonl` — RAG corpus (JSONL)

Writer: `stages/export.py` `JsonlParquetExport.export` / `_batched`. One JSON object per line, `json.dumps(row, sort_keys=True, ensure_ascii=False)`, exactly one trailing `\n` per line. The object is the record's full dump, so for a `ChunkRecord` it contains every field below (nested objects as JSON objects, not strings):

| Field | Type | Optional? | Meaning (source: `contracts.py` `ChunkRecord` / `Record`) |
|---|---|---|---|
| `schema_version` | int | no | Contract version (15). |
| `record_id` | string | no | The chunk's stable id (content-addressed). **Primary key.** |
| `provenance` | object | no | `Provenance` — source lineage + offsets (see 2.9). |
| `classification` | object | no | `Classification` — language/category/sensitivity (see 2.10). |
| `document_record_id` | string | no | Parent `DocumentRecord.record_id`. **Join key.** |
| `index` | int | no | 0-based chunk ordinal within the document. |
| `content` | string | no (default `""`) | Markup-stripped, PII-redacted retrieval text. **Byte-unchanged by W13.** |
| `media_type` | string | no (default `text/markdown`) | How to interpret `content`. |
| `token_count` | int | no | Budgeted token count for the chunk text. |
| `page_map` | object\|null | yes | Parent doc page-boundary map (`PageMap`), so a sub-chunk offset resolves to its own page. |
| `offset_map` | object\|null | yes | `OffsetMap`: run-length map stripped-`content`→original-markdown offsets (v11). |
| `risk_markers` | array<RiskMarker> | no (default `[]`) | Content-Screening flags that survive into the corpus (filterable — exclude flagged chunks). |
| `induced_labels` | object\|null | yes | `InducedLabels` the `schema_induction` Stage attached (v14). |
| `context_header` | object\|null | yes | `ContextHeader` — the KG projected onto the chunk (v15, W13). See 2.11. `null` when the `context_enrichment` Stage was absent. |
| `embedding` | array<float> | yes | Present only when an Embedder is wired (see 2.3). |

**Example line** (abridged, chunk with a context header, no embedding):

```json
{"classification":{"category":"contract","language":"en","schema_version":15,"sensitivity":"confidential"},"content":"Acme Corporation entered into a master services agreement with Beta LLC on 3 March 2024.","context_header":{"entities":["Acme Corporation","Beta LLC"],"entity_types":["Organization","Organization"],"kg_node_ids":["b1c2...","d3e4..."],"neighbor_node_ids":["f5a6..."],"neighbor_triples":["Acme Corporation —acquired→ Gamma Inc"],"situating_sentence":null,"model_id":null,"schema_version":15},"document_record_id":"sha256:9a…","index":4,"media_type":"text/markdown","offset_map":{"length":88,"segments":[{"doc_start":512,"local_start":0}],"schema_version":15},"page_map":null,"provenance":{"char_end":600,"char_start":512,"document_id":"sha256:9a…","file_name":"msa.pdf","file_size":48210,"file_type":"pdf","page_end":2,"page_start":2,"schema_version":15,"source_system":null,"source_uri":"file:///corpus/msa.pdf"},"record_id":"sha256:1f…","risk_markers":[],"induced_labels":null,"schema_version":15,"token_count":21}
```

### 2.2 `records.parquet` — RAG corpus (Parquet / Arrow)

Writer: `JsonlParquetExport._parquet_schema` / `_record_batch` (`export.py:309-364`). **Flat** schema, streamed as row-groups of 1024 rows. Nested fields become JSON strings.

| Column | Arrow type | Always present? | Meaning |
|---|---|---|---|
| `record_id` | `string` | yes | Chunk id. |
| `schema_version` | `int64` | yes | Contract version. |
| `provenance` | `string` (JSON) | yes | `Provenance`, sorted-key JSON. |
| `classification` | `string` (JSON) | yes | `Classification`, sorted-key JSON. |
| `content` | `string` | yes | Retrieval text (redacted, markup-stripped). |
| `media_type` | `string` | yes | e.g. `text/markdown`. |
| `embedding` | `list<double>` | **only if** an Embedder is wired | Dense vector, fixed width (see 2.3). |
| `context_header` | `string` (JSON) | **only if** `context_columns: true` | The chunk's `ContextHeader` as sorted-key JSON, `""` when the row has no header. |
| `neighbor_entity_ids` | `list<string>` | **only if** `context_columns: true` | The header's `neighbor_node_ids` (query-time KG-expansion join keys), `[]` when absent. |

**Genuinely-optional guarantee:** without `embedder` and without `context_columns`, the Parquet schema is the first six columns and is **byte-identical** to a pre-W13 run (`export.py:310-333`). An all-empty corpus still writes the schema-only Parquet so a reader always gets the column set (`export.py:279-282`).

> The flat Parquet projection deliberately drops `document_record_id`, `token_count`, `risk_markers`, `offset_map`, `page_map`, `induced_labels` — those live fully in `records.jsonl`. A retrieval engine that needs them either reads the JSONL or projects `provenance` (which carries `document_id`). **Flag:** if the engine needs `document_record_id`/`risk_markers` as first-class Parquet columns, that is a small additive change to `_parquet_schema` — the seam exists but is not wired today.

### 2.3 The `embedding` column (opt-in dense vector)

`JsonlParquetExport` resolves an Embedder Provider only when the Export config names one under `embedder: {"provider": <entry-point>, "config": {...}}` (`export.py:186-228`). Then every row gets an `embedding` of the Embedder's fixed `dimension`.

- **What is embedded** (`_corpus_text`, `export.py:77-100`): the row's corpus text = `masked_content` for a `RedactionRecord` (the PII-handled variant) else `content`; **W13 prepends the rendered `context_header`** (`header_text + "\n" + base`) to the **embedding input only** — the stored `content` never changes.
- **Reference Embedders / dims** (entry points, `latence.providers`):
  - `embedding.hashing` — in-core, deterministic, dependency-free; default `dimension` = **64** (`embedding.py:26`). The offline reference.
  - `embedding.endpoint` — served OpenAI-compatible model; dim by config (e.g. **768**).
  - `embedding.st` — sentence-transformers; dim read from the model. Default IBM Granite Embedding r2 (ADR-0045): GPU tier 311m = **768** (declared default), CPU tier 97m = **384** (`latence-embedder-st/provider.py`).

Absent an Embedder, the corpus still ships full text + Provenance/Classification, so any vector DB can embed later (ADR-0017).

### 2.4 `graph-nodes.parquet` — KG nodes

Writer: `stages/graph_export.py` `_write_nodes_parquet` (`graph_export.py:140-177`). One row per `GraphNode`.

| Column | Arrow type | Meaning (`contracts.py` `GraphNode`) |
|---|---|---|
| `node_id` | `string` | Deterministic content-addressed id = sha256(corpus_id + canonical-entity id). **Node primary key.** |
| `entity_id` | `string` | Source `CanonicalEntity.record_id`. |
| `type` | `string` | Entity type (the record's `label`; **column named `type`** — W12 — to match GraphML `type` and TTL `lg:entityType`). |
| `canonical_name` | `string` | Canonical surface string. |
| `kb_id` | `string` | External KB id, `""` when unlinked. |
| `confidence` | `float64` | Node (cluster) confidence 0..1. |
| `member_mention_ids` | `string` (JSON array) | `record_id`s of every `EntityMention` behind this node. **Join key → mentions.** |
| `source_document_ids` | `string` (JSON array) | Distinct source documents this node appears in. |
| `properties` | `string` (JSON) | Open attribute bag (mention_count, document_count, …). |
| `evidence` | `string` (JSON) | `Evidence` (mention_ids, document_ids, snippet, confidence). |

### 2.5 `graph-edges.parquet` — KG edges

Writer: `_write_edges_parquet` (`graph_export.py:179-208`). One row per `GraphEdge`.

| Column | Arrow type | Meaning (`contracts.py` `GraphEdge`) |
|---|---|---|
| `edge_id` | `string` | Deterministic content-addressed id = sha256(corpus_id + source-relation id). |
| `relation_id` | `string` | Source `NormalizedRelation.record_id`. |
| `source_node_id` | `string` | Head `GraphNode.node_id`. **Join key → nodes.** |
| `target_node_id` | `string` | Tail `GraphNode.node_id`. **Join key → nodes.** |
| `label` | `string` | Normalized relation label (edge label). |
| `confidence` | `float64` | Edge confidence 0..1. |
| `properties` | `string` (JSON) | Open bag (`inverted`, `source_document_ids`; inferred/scorer for W2 predicted edges). |
| `evidence` | `string` (JSON) | Per-edge `Evidence` — the mentions/documents justifying the edge. **Every edge carries Evidence** (S9 AC). |

Head and tail node are always distinct (no self-loop; contract-enforced).

### 2.6 `graph.ttl` — KG as RDF Turtle

Writer: `_render_ttl` (`graph_export.py:212-244`). Hand-emitted, no `rdflib` dep. Prefixes:

```
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix lg:   <urn:latence:graph:> .
```

- **Node** → subject `<urn:latence:graph:node:NODE_ID>`:
  `a lg:Entity ; rdfs:label "…" ; lg:entityType "…" ; [lg:kbId "…"] ; lg:confidence "0.xxxxxx"^^xsd:decimal .`
- **Edge** → a direct triple `<node:SRC> <rel:LABEL> <node:TGT> .` **plus** a reified statement `<edge:EDGE_ID> a rdf:Statement ; rdf:subject … ; rdf:predicate … ; rdf:object … ; lg:confidence "…"^^xsd:decimal ; lg:evidenceCount N .` where `N = len(edge.evidence.mention_ids)`.

IRIs percent-escape RFC-3987-forbidden chars; literals scrub XML/Turtle-illegal control chars. Base IRI is a non-dereferenceable `urn:` (self-contained, no network).

### 2.7 `graph.graphml` — KG as GraphML

Writer: `_render_graphml` (`graph_export.py:279-315`). XML, `edgedefault="directed"`. Declared `<key>`s:

| key id | for | attr.name | attr.type | Source |
|---|---|---|---|---|
| `label` | node | `label` | string | `canonical_name` |
| `type` | node | `type` | string | node `label` (entity type) |
| `kb_id` | node | `kb_id` | string | `kb_id` (omitted when empty) |
| `confidence` | node | `confidence` | double | node `confidence` |
| `rel` | edge | `label` | string | edge `label` |
| `econfidence` | edge | `confidence` | double | edge `confidence` |
| `evidence` | edge | `evidence_count` | int | `len(edge.evidence.mention_ids)` |

`<node id=node_id>` and `<edge id=edge_id source=source_node_id target=target_node_id>` reuse the same ids as the Parquet/TTL, so the three graph formats are the same graph under the same keys.

### 2.8 `quality-report.json` (+ `.md`)

Writers: `runner.py:1075` (`report.to_json()` → `quality-report.json`); `cli.py:112` (`report.render()` → `quality-report.md`). Contract: `quality.py` `QualityReport` (`schema_version` = 19). Top-level fields:

| Field | Type | Present when | Meaning |
|---|---|---|---|
| `schema_version` | int | always | 19. |
| `pipeline_name` / `run_id` | string | always | Run identity. |
| `document_count` | int | always | Distinct source documents. |
| `total_duration_seconds` | float | always | Sum of Stage durations. |
| `stages` | array<StageMetrics> | always | Per-Stage in/out counts, duration, from_checkpoint, error_category. |
| `completeness` | CompletenessCheck | always | records_checked / with_provenance / with_classification / offset_bearing_records / with_aligned_offsets. |
| `contracts_complete` | bool | always | Completeness pass/fail. |
| `parse` | ParseQuality\|null | Parse Stage present | Page-map drift, parse errors, bytes_after_parse. |
| `chunk` | ChunkQuality\|null | Chunk Stage | chunks_total, tokens_total, offset_preserving, chunks_flagged. |
| `screening` | ScreeningQuality\|null | Screening Stage | intake/content screened, quarantined, flagged, findings. |
| `entities` | EntityQuality\|null | Entity/fused Stage | per_type counts, confidence dist, mentions_offset_resolving. |
| `relations` | RelationQuality\|null | Relation/fused Stage | per_type, confidence, fan-out caps. |
| `redaction` | RedactionQuality\|null | Redaction Stage | per_type / per_action PII counts, documents_with_pii, `redaction_disabled_for_sensitive`. |
| `profiling` | ProfilingQuality\|null | Profiling Stage | avg readability/density/…, corpus aggregates, coverage maps, salted co-occurrence fingerprints. |
| `disambiguation` | DisambiguationQuality\|null | Disambiguation Stage | mentions_in vs entities_out, merges_by_strategy, linked/unlinked. |
| `graph` | GraphQuality\|null | Graph Assembly Stage | nodes/edges totals, by type/label, **evidence_coverage**, inferred-edge counts. |
| `context_enrichment` | ContextEnrichmentQuality\|null | `context_enrichment` Stage (W13) | chunks_enriched, avg_neighbors_per_chunk, hub_entities_capped. |
| `drift` | DriftDiagnostics\|null | page-map chunks present | Provenance-integrity drift roll-up. |
| `goldset` | GoldSetEvaluation\|null | `gold_set_uri` supplied | Per-Stage precision/recall/F1. |
| `delta` | DeltaChurn\|null | incremental delta run | docs ±, entities created/merged/split, edges ±, drift. |
| `conformance` | ConformanceReport\|null | conformance sweep attached | Per-Provider C1–C6 verdict. |

The `.md` is the same numbers rendered deterministically (`QualityReport.render`, `quality.py:705`). **Safe-to-share caveat:** co-occurrence entity surfaces are replaced by a **salted sha256 fingerprint** in the report (not irreversible; raw pairs remain only in the exported `CorpusFeatures` data artifact) — see `quality_report_builder.py:101-138`.

### 2.9 `Provenance` (embedded in every record)

`contracts.py:153-172`, `frozen`, `extra="forbid"`.

| Field | Type | Optional? | Meaning |
|---|---|---|---|
| `schema_version` | int | no | 15. |
| `source_uri` | string (min 1) | no | Storage URI the source was read from. |
| `file_name` | string (min 1) | no | Source file name. |
| `file_type` | string (min 1) | no | Lowercased extension, no dot. |
| `file_size` | int ≥0 | no | Source size in bytes. |
| `document_id` | string (min 1) | no | **Content-addressed source doc id.** Join key across the whole run. |
| `source_system` | string\|null | yes | Originating system. |
| `page_start` / `page_end` | int ≥0 \| null | yes | Source page(s) the record spans. |
| `char_start` / `char_end` | int ≥0 \| null | yes | Half-open char span in the parent document's assembled markdown. |

### 2.10 `Classification` (embedded in every record)

`contracts.py:175-186`, `frozen`.

| Field | Type | Optional? | Meaning |
|---|---|---|---|
| `schema_version` | int | no | 15. |
| `language` | string (min 1) | no | BCP-47-ish tag; `und` if unknown. |
| `category` | string\|null | yes | e.g. `contract`. |
| `sensitivity` | string | no (default `unknown`) | `public`/`internal`/`confidential`/… — the ACL/filter key. |

### 2.11 `ContextHeader` (W13; on `ChunkRecord.context_header`)

`contracts.py:615-695`, `frozen`. Built deterministically (sorted, capped, content-addressed ids) from the run's `EntityMention` + `GraphRecord` streams.

| Field | Type | Optional? | Meaning |
|---|---|---|---|
| `schema_version` | int | no | 15. |
| `entities` | list<string> | no (default `[]`) | The chunk's canonical entity names (sorted, deduped). |
| `entity_types` | list<string> | no (default `[]`) | Entity type per name (same length + order as `entities`; validator-enforced). |
| `neighbor_triples` | list<string> | no (default `[]`) | Top-k KG-neighbor relations rendered `A —label→ B`, confidence-sorted and **capped** (`max_neighbors_per_entity`, `max_triples`). |
| `kg_node_ids` | list<string> | no (default `[]`) | `GraphNode.node_id`s of the chunk's **own** entities. **Join key → nodes.** |
| `neighbor_node_ids` | list<string> | no (default `[]`) | `GraphNode.node_id`s the triples reach (query-time KG-expansion keys; sorted, deduped). Exported as the Parquet column `neighbor_entity_ids`. |
| `situating_sentence` | string\|null | yes | OPTIONAL LLM Contextual-Retrieval sentence; `null` in the deterministic reference build. |
| `model_id` | string\|null | yes | LLM that wrote the situating sentence; `null` otherwise. |

### 2.12 Other exportable record contracts (reference)

These are inter-`Record` carriers (each carries `Provenance` + `Classification` + `record_id`) that thread the DAG and *can* be exported, but are delivered to the engine primarily via the KG files:

- **`EntityMention`** (`contracts.py:782`): `document_record_id`, `chunk_record_id`, `index`, `label`, `text`, `confidence`; Provenance carries the mention's char + page span in the parent doc.
- **`RelationMention`** (`contracts.py:838`): `document_record_id`, `label`, `head_mention_id`/`tail_mention_id`, `head_text`/`tail_text`, `confidence`.
- **`CanonicalEntity`** (`contracts.py:1329`): `canonical_text`, `entity_type`, `member_mention_ids`, `source_document_ids`, `kb_id`, `merge_decisions`, `evidence`.
- **`NormalizedRelation`** (`contracts.py:1387`): `source_relation_id`, `label`, `head_entity_id`/`tail_entity_id`, `inverted`, `evidence`.
- **`RedactionRecord`** (`contracts.py:984`): `masked_content` (the PII-handled corpus variant), `pii_spans` (`PIISpan`: type/detector/offset/action/placeholder — **never the raw value**), `policy`, `redaction_disabled_for_sensitive`.
- **`Evidence`** (`contracts.py:1237`): `mention_ids`, `document_ids`, `snippet`, `confidence` — justification for an inference (distinct from `Provenance`, which is source lineage).

---

## 3. The join model

Content-addressed, stable ids stitch the corpus to the KG to the source. Keys:

| Relationship | From | To | Key |
|---|---|---|---|
| chunk → parent document | `ChunkRecord.document_record_id` | `DocumentRecord.record_id` | record id |
| chunk / any record → source doc | `record.provenance.document_id` | source manifest | content-addressed doc id |
| mention → chunk | `EntityMention.chunk_record_id` | `ChunkRecord.record_id` | record id |
| mention → document | `EntityMention.document_record_id` | `DocumentRecord.record_id` | record id |
| canonical entity → mentions | `CanonicalEntity.member_mention_ids` | `EntityMention.record_id` | record id list |
| node → canonical entity | `GraphNode.entity_id` | `CanonicalEntity.record_id` | record id |
| node → mentions | `GraphNode.member_mention_ids` | `EntityMention.record_id` | record id list |
| edge → endpoints | `GraphEdge.source_node_id` / `target_node_id` | `GraphNode.node_id` | node id |
| edge → source relation | `GraphEdge.relation_id` | `NormalizedRelation.record_id` | record id |
| edge → justification | `GraphEdge.evidence.mention_ids` / `document_ids` | `EntityMention.record_id` / source docs | id lists |
| **chunk → its KG nodes** | `context_header.kg_node_ids` (col `context_header`) | `GraphNode.node_id` | node id |
| **chunk → KG neighbors** | `context_header.neighbor_node_ids` (col `neighbor_entity_ids`) | `GraphNode.node_id` | node id |
| any record → source offsets | `provenance.char_start/char_end`, `page_start/page_end` | source document | char/page offsets |

**The W13 shortcut:** `context_header.kg_node_ids` is the direct chunk→graph bridge, so the engine does **not** need `EntityMention` exported to connect a chunk to the KG.

**Reconstruction path — chunk → its entities → its KG neighbors → back to source offsets:**

```
records.parquet row (chunk)
  └ context_header.kg_node_ids ─────────────► graph-nodes.parquet (node_id)      # the chunk's entities
        └ graph-edges.parquet where source_node_id ∈ kg_node_ids                  # expand 1 hop
              └ target_node_id ──────────────► graph-nodes.parquet (neighbors)    # = neighbor_node_ids
                    └ node.member_mention_ids / edge.evidence.mention_ids         # justification
                          └ (EntityMention.provenance) char_start/char_end + page # exact source span
  └ own provenance.document_id + char_start/char_end + page_start/page_end ──────► source_uri  # chunk's own offsets
```

Every hop uses a content-addressed id, so two runs over the same corpus reproduce the same joins (determinism, §7).

---

## 4. Mapping to the retrieval engine's five layers

| Layer | Pipeline PROVIDES | Engine BUILDS |
|---|---|---|
| Dense / multi-vector (Qdrant + MUVERA) | Clean, bounded chunk `content` (+ optional single dense `embedding`) | ColBERT token vectors + the MUVERA FDE; the vector index |
| SPLADE (learned sparse) | Chunk `content` (+ optional `context_header` prefix) | SPLADE encoding + sparse index |
| BM25 (lexical) | Chunk `content` + filterable metadata | The lexical index |
| KG | Nodes + edges + Evidence (**pipeline owns the KG**) | Graph expansion / entity-linked retrieval / rerank |
| Advanced metadata | `provenance`, `classification`, `induced_labels`, `context_header`, redaction status | Filters, ACLs, rerank features |

**Dense / multi-vector (Qdrant + MUVERA).** The pipeline delivers clean, redacted, markup-stripped chunk `content` bounded to the extractor model window (the GLiNER-safe ≤768-token bound, `DEFAULT_GLINER_MAX_LEN = 768`, `providers/perf.py:418`), optionally with the W13 `context_header` folded into the embedding input, and optionally a **single** dense `embedding` (configurable model/dim — hashing-64 / e5-384 / endpoint-768). **MUVERA** (Multi-Vector Retrieval via Fixed Dimensional Encodings, NeurIPS 2024) asymmetrically maps a ColBERT-style *set* of token vectors into **one** Fixed-Dimensional Encoding whose inner product approximates the multi-vector (Chamfer) similarity — so multi-vector retrieval reduces to single-vector MIPS on an off-the-shelf ANN index. **Design decision:** the pipeline ships **text (+ optional single dense vector)**; the **engine owns** ColBERT token-vector production and the MUVERA FDE. **Seam:** the exporter's Embedder is opt-in and single-vector today (`export.py:186-228`, `_parquet_schema` writes one `list<double>` column); a multi-vector export embedder would be an additive Embedder Provider + a `list<list<double>>` column later — the seam is named, not wired. (MUVERA def.: [arXiv 2405.19504](https://arxiv.org/abs/2405.19504); [Google Research](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/).)

**SPLADE (learned sparse).** Pipeline delivers chunk `content`; the engine SPLADE-encodes it. The `context_header` rendering (`_context_header_text`, `export.py:49-74`) can be prepended for context-aware sparse encoding just as it is for dense — the header text is available from the `context_header` column.

**BM25 (lexical).** Chunk `content` for the lexical index; the metadata fields (§4 metadata row) are the filter predicates. Engine builds the index.

**KG.** `graph-nodes.parquet` / `graph-edges.parquet` (+ `graph.ttl` / `graph.graphml`) plus per-edge `Evidence` support graph expansion, entity-linked retrieval (HippoRAG-style), and evidence-grounded rerank. The pipeline **owns** the KG; the engine consumes it. The per-chunk `context_header` node-id columns let the engine expand/rerank without re-linking.

**Advanced metadata management** — enumerated by role:

| Field | Role |
|---|---|
| `classification.language`, `classification.category`, `classification.sensitivity` | **FILTER / ACL** |
| `risk_markers` (chunk) | **FILTER** (exclude flagged) |
| `provenance.source_system`, `provenance.file_type` | **FILTER** |
| `induced_labels.entity_types/relation_types/pii_types` | **FILTER / RERANK** |
| `context_header.entities`, `context_header.neighbor_triples`, `situating_sentence` | **RERANK / context-aware encode** |
| `context_header.kg_node_ids`, `context_header.neighbor_node_ids` (col `neighbor_entity_ids`) | **JOIN KEYS** (KG expansion) |
| `record_id`, `document_record_id`, `provenance.document_id` | **JOIN KEYS** |
| `provenance.char_start/char_end`, `page_start/page_end`, `source_uri`, `file_name` | **PROVENANCE** (cite/highlight source) |
| redaction status (`RedactionRecord.pii_spans`, `redaction_disabled_for_sensitive`) | **FILTER / audit** |

---

## 5. The W13 context-enrichment contribution

- The `context_enrichment` Stage (optional) projects the already-computed KG back onto each chunk as a compact `ContextHeader` (`contracts.py:137-149`, ADR-0039).
- The Export prepends the rendered header (`Entities: name (type), … | Relations: A —label→ B; …`) to the **embedding input ONLY** — `header_text + "\n" + base` (`export.py:85-100`). The stored `content` / `masked_content` is **byte-unchanged**, so retrieval sees cross-chunk connections in the vector while the LLM window stays the clean chunk.
- Two opt-in Parquet columns (`context_columns: true`) expose the projection without bloating text: `context_header` (JSON) and `neighbor_entity_ids` (= `neighbor_node_ids`, node-id list). These let the engine do KG expansion + rerank straight off the corpus row (`_context_columns_for_row`, `export.py:103-119`).
- **Anti-bloat discipline:** compact rendered triples (not neighbor prose), a top-k neighbor + confidence cap (hub-entity guard), and **join keys not raw KG text** — query-time KG expansion is the retriever's job (ADR-0017); W13 emits the keys.

---

## 6. Backend substitution

Everything is open columnar / RDF / JSON written through the single fsspec `Storage` seam (`storage.py`, ADR-0009). Default allowed schemes (`DEFAULT_ALLOWED_SCHEMES`, `storage.py:28-30`): **`file`, `memory`, `s3`, `gs`, `gcs`, `abfs`, `az`**. Network-fetch schemes (`http`/`ftp`/`sftp`) are deliberately excluded (anti-SSRF); credentials resolve through fsspec's own chains (the framework stores no secrets). The same URI + artifacts map onto any target:

| Target | Corpus text + dense | Metadata | KG |
|---|---|---|---|
| **Qdrant** | payload text + named/ dense vector; optional sparse (SPLADE the engine computes) | `provenance`/`classification`/`risk_markers` → payload filters | separate collection or external graph store; `neighbor_entity_ids` as payload for expansion |
| **Databricks** | `records.parquet` → Delta table + Unity Catalog; `embedding` col → Vector Search | Delta columns; sensitivity → UC governance/ACL | `graph-nodes`/`graph-edges` Parquet → Delta graph tables |
| **Azure** | `records.parquet` → AI Search index (vector + text fields) | AI Search filterable fields | `graph.ttl`/edges → Cosmos DB Gremlin |

Nothing in the output is engine-specific: Parquet/JSONL for the corpus, Parquet + TTL + GraphML for the graph, JSON for the report.

---

## 7. Guarantees the retrieval engine can rely on

1. **Exact offset round-trip to source.** Every record's `Provenance` carries `char_start/char_end` (+ `page_start/page_end`) in the parent document's assembled-markdown coordinates; `page_map` + `offset_map` on the chunk let a sub-chunk offset resolve to its **true** original offset and page (`OffsetMap`, `PageOffsetIndex`), not a lower bound.
2. **Redaction applied before export.** The corpus text is the PII-handled variant — the Embedder embeds `masked_content` when present, and `PIISpan` never stores the raw value (`contracts.py:929-945`). `redaction_disabled_for_sensitive` is a loud flag that an empty `pii_spans` on a sensitive doc means "control OFF", not "clean" (H-C1 §4).
3. **Deterministic / reproducible.** Content-addressed ids (chunk ids, `node_id` = sha256(corpus+entity), `edge_id` = sha256(corpus+relation)); sorted-key JSON; capped/sorted collections; streamed Parquet byte-identical to the buffered path. A seeded run over the same corpus + config is byte-identical.
4. **Evidence-linked edges.** Every `GraphEdge` carries `Evidence` naming the source mentions/documents; the Quality Report's `graph.evidence_coverage` reports the fraction of edges so justified (1.0 = all).
5. **Additive / versioned schemas.** `schema_version` on every record; new fields default to `None`/empty so older readers keep validating.
6. **Atomic publish.** Both corpus files and the report are written to temp paths and atomically renamed (`Storage.atomic_writers` for the streamed corpus file *set*, `Storage.write_atomic` for the buffered report), so a reader never sees a torn file — and a failed run sweeps its own temp paths rather than leaving them on the store. On an object store "atomically renamed" means the destination object is written in one atomic, read-after-write-consistent PUT; the underlying copy-then-delete is two operations, so a crash mid-publish can still leave a temp object behind (see `Storage.atomic_writer`).

---

## 8. Known boundaries — what the pipeline does NOT provide

- **No multi-vector / SPLADE vectors by default.** The pipeline ships text and (optionally) **one** dense vector; ColBERT token vectors, the MUVERA FDE, and SPLADE sparse encodings are the engine's job. A multi-vector export embedder is a named-but-unwired seam (§4).
- **No live query API.** Files only (ADR-0017); there is no server, no live graph DB, no query endpoint. The engine loads the files.
- **Situating sentence is optional/future.** `context_header.situating_sentence` / `model_id` are `null` in the deterministic reference build; the LLM Contextual-Retrieval mode is a documented follow-on (ADR-0039).
- **KG quality depends on extraction config.** Nodes/edges are only as rich as the entity/relation extractors and label config; the KG (and thus `context_header`) is empty on a Parse→Chunk→Export-only pipeline with no extraction / disambiguation / graph-assembly Stages, and `context_header` is `null` without a `context_enrichment` Stage.
- **Default corpus Parquet is chunk-level and lean.** `EntityMention`/`RelationMention`/`DisambiguationRecord` streams are exportable `Record`s but are **not** written into `records.parquet` by default; the entity/relation layer reaches the engine through the KG files + the `context_header` join columns. Promoting mentions to their own export file, or adding `document_record_id`/`risk_markers` as first-class Parquet columns, is an additive change to the exporter (seam exists).

---

### Fields that could NOT be resolved from the code (flagged, not guessed)

1. **`corpus.jsonl` / `corpus.parquet` naming.** The task brief names the corpus files `corpus.*`; the **on-disk default is `records.jsonl` / `records.parquet`** (`JsonlParquetExport` `basename` default = `"records"`). They become `corpus.*` only if the operator sets `basename: corpus`. Documented as the default with the override noted — no field guessed.
2. **"DenseOn 768" default.** No single "DenseOn" preset sets a 768-dim default in core. The reference `embedding.hashing` default dim is **64**; the sentence-transformers default is now IBM Granite Embedding r2 — **768** (GPU tier 311m, the declared default) or **384** (CPU tier 97m), per ADR-0045. Stated explicitly rather than asserting a single default.
3. **Whether `EntityMention`/`RelationMention` get their own export files in a shipped stack.** This is **pipeline-configuration-dependent** (the Export sink materializes whatever stream is wired to it); no fixed mentions/relations file name is guaranteed by core, so none is asserted. Flagged in §1 and §8.
