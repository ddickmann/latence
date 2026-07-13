# W13 — context enrichment: project the KG back onto chunks for retrieval coherence (no context bloat)

**The connections a chunk needs for retrieval relevance are ALREADY computed — as the KG — but the RAG
export embeds each chunk in isolation and discards them.** W13 adds an OPTIONAL, deterministic
`context_enrichment` Stage after Graph Assembly that stamps each chunk with a compact `context_header`
(its canonical entities + top-k KG-neighbor triples), which the Export prepends to the EMBEDDING INPUT
ONLY. Coherence goes into the vector + a metadata column; the LLM window stays the clean chunk.
Sourced from the SOTA read (Anthropic Contextual Retrieval + KG-augmented retrieval / HippoRAG): the
winning move is "context into the vector + compact triples as metadata," and this framework is one
field away from it because it already emits an evidence-linked KG.

## Contract (additive, `induced_labels`/`risk_markers` precedent)
- New small versioned model `ContextHeader`: `entities: list[str]` (canonical names), `entity_types:
  list[str]`, `neighbor_triples: list[str]` (rendered `A —label→ B`, capped), `kg_node_ids: list[str]`
  (join keys), `situating_sentence: str | None` (optional LLM mode, default None), `model_id: str | None`.
- `ChunkRecord.context_header: ContextHeader | None = None` — additive, defaulted None; SCHEMA_VERSION
  bump; a prior-version record JSON still validates. No stage ⇒ field stays None ⇒ byte-identical run.

## The Stage + reference provider (deterministic, CPU, zero-dep — ships in core, like graph.canonical)
- New OPTIONAL `context_enrichment` Capability + Stage, placed AFTER `graph_assembly`. Shaped like
  `GraphCompleter` (ADR-0037): `depends_on` the chunk stream AND graph assembly; consumes
  `ChunkRecord`s + `EntityMention`s + `GraphRecord`s; yields the SAME chunks with `context_header`
  populated via `chunk.model_copy(update=...)` (exactly as the inducer does). Genuinely optional +
  contract-preserving (offsets/provenance/content untouched).
- Reference provider `context.kg_header` — pure-Python deterministic build per chunk:
  1. **Its canonical entities**: `EntityMention.chunk_record_id == chunk.record_id` → mention record_ids
     → the `CanonicalEntity`/`GraphNode` whose `member_mention_ids` contain them (→ canonical_name,
     entity_type). Confidence-GATE (drop mentions/entities below a config `min_entity_confidence`).
  2. **Its KG-neighbor triples**: for each such node, its `GraphEdge`s (`source_node_id`/`target_node_id
     == node.node_id`), rendered `entity —label→ neighbor_canonical_name`, confidence-sorted, capped to
     `max_neighbors_per_entity` (top-k) and an overall `max_triples` — the BLOAT + hub-entity guard
     (reuse the `max_evidence_per_node` precedent). A corpus-wide hub entity contributes only its TOP
     relations, never all of them.
  3. Render `entities`/`entity_types`/`neighbor_triples`/`kg_node_ids` (sorted, capped, content-addressed
     ids) → byte-identical under seed.
- Config: `max_neighbors_per_entity` (default ~4), `max_triples` (default ~6), `min_entity_confidence`,
  `min_edge_confidence`. Determinism: deterministic=True (clears the Baseline/G1 bar).
- (Optional future mode, DOCUMENT don't-build-now unless trivial) `situating_sentence` via a pluggable
  LLM provider reusing `label_inducer.llm`'s endpoint seam (Contextual-Retrieval style); deterministic=False,
  same honest caveat as the inducer. Keep the slice deterministic; leave the LLM sentence as a documented
  follow-on so W13 stays byte-identical.

## Export wiring (the one place the "flat schema drops rich fields" gap is repaid)
- `JsonlParquetExport._corpus_text`: when the row carries a `context_header`, feed
  `header_text + "\n" + content` to the EMBEDDING path only (`_attach_embeddings`). The stored
  `content`/`masked_content` is UNCHANGED — the exported chunk text stays clean, downstream LLM prompts
  are byte-for-byte the same size as today.
- `_parquet_schema` + `_record_batch`: add `context_header` (JSON string) and `neighbor_entity_ids`
  columns so a downstream retriever/vector-DB can filter/hybrid-BM25/rerank on them (query-time KG
  expansion is the RETRIEVER's job — ADR-0017 files-not-a-DB; W13 only emits the join keys).

## Bloat discipline (the crux of the ask — enforce all four)
1. Compact triples, not neighbor prose. 2. Top-k neighbor cap + confidence sort (hub-entity guard,
mandatory). 3. Keys/summary not raw text — KG stays in its own files for the retriever. 4. Embed-but-
don't-store-in-content — the header inflates the VECTOR, not the exported chunk, so the LLM window
never grows unless the adopter chooses to inject the header.

## Tests + verify
- Deterministic build from a faithful fake KG: correct entities+triples per chunk; top-k + max_triples
  caps bite (a hub entity in many chunks contributes only its top relations); confidence gate drops
  low-confidence entities/edges; byte-identical across two seeded runs.
- Export: embedding input includes the header, stored `content` does NOT (a test asserts the persisted
  chunk text is unchanged while the embedded text differs); `context_header`/`neighbor_entity_ids`
  columns present + populated.
- Optional-stage: a pipeline WITHOUT the stage yields `context_header=None` and byte-identical output
  (the additive-field guarantee).
- Conformance C1–C6 + roster+1; Quality Report gains `chunks_enriched`/`avg_neighbors_per_chunk`/
  `hub_entities_capped`; `bash scripts/verify-local.sh` GREEN; ADR-0039 (context enrichment).

## Definition of Done
- Optional deterministic `context_enrichment` stage + `context.kg_header` provider building a capped,
  confidence-gated `context_header` from the run's existing EntityMention+GraphRecord streams; export
  prepends it to the embedding input only + adds the two columns; additive ChunkRecord field
  (SCHEMA_VERSION bump); byte-identical without the stage; bloat guards enforced + tested; verify-local
  green; ADR. No new model/GPU/key in the default path.

## Reviewer should check
- Header build is pure/deterministic (sorted+capped+content-addressed) → clears Baseline bar.
- The header hits the EMBEDDING input only; stored chunk content is byte-unchanged (LLM window doesn't grow).
- Top-k/hub-entity + confidence caps genuinely bound the header (a hub entity does NOT swamp its chunks).
- Additive field: no stage ⇒ None ⇒ byte-identical; offsets/provenance/contracts untouched.
- Scope discipline: NO query-time loop, NO GraphRAG community summaries — only surface the existing KG
  into the vector + metadata; the LLM-sentence mode stays optional/documented, not the default.
