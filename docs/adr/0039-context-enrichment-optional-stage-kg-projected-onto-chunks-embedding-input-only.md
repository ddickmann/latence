# Context Enrichment is an OPTIONAL Stage that projects the KG back onto chunks into the EMBEDDING input only

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

**Status: accepted** — W13-context-enrichment. Adds the `ContextEnricher` Capability + a
`context_enrichment` Stage (a reference Provider `context.kg_header`, pure-Python, deterministic,
CPU, zero-dep), the additive `ContextHeader` contract + `ChunkRecord.context_header` field, and the
Export wiring that prepends the header to the embedding input only + adds two opt-in metadata
columns. Builds on ADR-0004 (Capability protocols + Provider plugins), ADR-0007 (CPU-first — no
model/GPU/key in the default path), ADR-0016 (thin core, zero extra deps), ADR-0017 (Export writes
files, never a live DB — the retriever owns query-time expansion), ADR-0026/0027 (the
CanonicalEntity/GraphNode/GraphEdge/Evidence streams this reads), ADR-0037 (the OPTIONAL post-Assembly
Stage + additive-field blueprint it mirrors), ADR-0038 (the additive-annotation record precedent —
`induced_labels`), ADR-0036 (Provider ecosystem: profile, conformance, bake-off).

## Context

The connections a chunk needs for retrieval relevance are **already computed** — as the knowledge
graph (entities + evidence-linked relations) — but the RAG Export embeds each chunk **in isolation**
and discards them. A vector built from the chunk text alone loses the entity/relation coherence the
pipeline just spent nine Stages deriving. The SOTA read (Anthropic Contextual Retrieval + KG-augmented
retrieval / HippoRAG) is unambiguous: the winning move is *context into the vector + compact triples
as metadata*, and this framework is one field away from it because it already emits an evidence-linked
KG.

The restoration must not compromise the invariants the framework is built on:

1. **The LLM window must not grow.** Downstream RAG prompts are token-budgeted; inflating the stored
   chunk text with KG prose would silently raise every adopter's prompt cost. The coherence must go
   into the **vector**, not the exported chunk.
2. **No context bloat.** A corpus-wide **hub entity** (an `Acme` that appears in hundreds of chunks
   with hundreds of relations) must never swamp its chunks — a naive "attach all neighbors" projection
   is worse than nothing.
3. **Genuinely optional + byte-identical.** Existing stacks must be unchanged; the new field defaults
   `None` and a stack without the Stage produces byte-identical output.
4. **Deterministic, offline, no new dependency in the default path.** The default build is CPU,
   zero-dep, seeded → byte-reproducible (the Baseline/G1 bar) — no model, no GPU, no key.

## Decisions

### 1. A new `ContextEnricher` Capability — an OPTIONAL corpus-level chunk→chunk Stage after Graph Assembly

Context enrichment is its own Capability (ADR-0004), shaped like `LabelInducer` (a chunk→chunk
transform that yields the SAME chunks with an additive field populated via `chunk.model_copy(update=
...)`) but **corpus-level and multi-input**, like Disambiguation: it consumes the run's `ChunkRecord`s
(what it enriches + passes through 1:1), `EntityMention`s (a chunk's canonical entities, via
`EntityMention.chunk_record_id`), AND the assembled `GraphRecord`s (NODE + EDGE — the entities' KG
neighbors). It depends_on the chunk stream (Chunk / Content-Screening / Schema-Induction) AND Graph
Assembly; the run's EntityMentions are gathered corpus-wide (Entity Extraction is a graph ancestor,
not a direct parent — the same whole-pipeline gather Disambiguation uses for chunks, #94). It is
**genuinely OPTIONAL**: a stack with no `context_enrichment` Stage is byte-identical to today (the
`context_header` field defaults `None`, and the Export prepends nothing). Proven by an e2e test that
runs the same pipeline with and without the Stage and asserts the upstream chunk checkpoint + the
exported chunk `content` are byte-identical.

### 2. The carrier — an additive `ContextHeader` contract + a defaulted record field (SCHEMA_VERSION 15)

A new versioned Pydantic `ContextHeader` (`entities` / `entity_types` / `neighbor_triples` /
`kg_node_ids` / `neighbor_node_ids` + the follow-on `situating_sentence` / `model_id`) is attached to
a chunk via a new optional `context_header: ContextHeader | None` field on `ChunkRecord`. This mirrors
the `risk_markers` / `induced_labels` annotation precedent exactly: a field on the record contract
(not an out-of-band note), so it survives serialize/deserialize + into the corpus and a downstream
retriever reads it off the chunk it already holds. Default `None` ⇒ every existing record + stack is
unchanged; `SCHEMA_VERSION` bumps to 15 (additive, defaulted — a v14 record JSON still validates).

### 3. The reference Provider `context.kg_header` — deterministic, capped, confidence-gated, zero-dep

The in-core reference (pure-Python, CPU, no model, no key — ADR-0007/0016) builds each chunk's header
from the run's existing EntityMention + GraphRecord streams:

- **The chunk's canonical entities**: its mentions (`EntityMention.chunk_record_id == chunk.record_id`)
  → their `CanonicalEntity`/`GraphNode` (via `member_mention_ids`). A **confidence gate** drops
  mentions and entities below `min_entity_confidence`.
- **Its KG-neighbor triples**: for each such node, its `GraphEdge`s (in either direction), rendered
  `A —label→ B`, gated by `min_edge_confidence`, confidence-sorted, and **capped** — top-k
  `max_neighbors_per_entity` per entity, then an overall `max_triples`.
- Everything sorted + content-addressed (the graph's node ids) → byte-identical under seed
  (`deterministic=True`, clears the Baseline/G1 bar). A chunk with no gated canonical entity is left
  `context_header=None` (nothing to add — byte-identical to no-stage for that chunk).

### 4. Bloat discipline — the crux — enforced structurally (all four)

1. **Compact triples, not neighbor prose** — each neighbor is one rendered `A —label→ B`.
2. **Top-k neighbor cap + confidence sort (the hub-entity guard, mandatory)** — per entity at most
   `max_neighbors_per_entity` of its highest-confidence relations survive, and an overall
   `max_triples` caps the header, so a corpus-wide hub entity contributes only its TOP relations,
   never all of them (reuses the `max_evidence_per_node` cap precedent from Graph Assembly).
3. **Keys not raw text** — `kg_node_ids` (the chunk's own entities) + `neighbor_node_ids` (the
   entities the triples reach) are join keys a downstream retriever expands on. **Query-time KG
   expansion is the RETRIEVER's job** (ADR-0017: files, not a live DB); W13 only emits the keys.
4. **Embed-but-don't-store-in-content** — the Export owns the prepend (embedding input only), so the
   header inflates the VECTOR, never the exported chunk. The LLM window never grows unless the adopter
   chooses to inject the header.

### 5. Export wiring — the one place the "flat schema drops rich fields" gap is repaid

`JsonlParquetExport._corpus_text`: when a row carries a `context_header`, the compact rendering is
prepended to the **EMBEDDING input only** (`header_text + "\n" + content`) — the stored
`content`/`masked_content` is byte-unchanged (a test asserts the persisted chunk text is identical
while the embedded text differs). Two **opt-in** Parquet columns (config `context_columns: true`)
surface the projection for a downstream retriever/vector-DB: `context_header` (the header as a JSON
string) and `neighbor_entity_ids` (its `neighbor_node_ids` as a native `list<string>`). They are
opt-in so the default Parquet schema is byte-identical to today (the genuinely-optional guarantee);
the JSONL always carries the full `context_header` (or `null`) via the record dump, exactly like every
other additive field.

### 6. The LLM situating-sentence mode is a DOCUMENTED follow-on, not the default

An optional Contextual-Retrieval-style **situating sentence** (an LLM writes a one-sentence "this
chunk is about …, in the context of …" prefix) is a natural extension behind the same seam, reusing
`label_inducer.llm`'s endpoint pattern (`deterministic=False`, the same honest untrusted-LLM caveat).
It is **deliberately NOT built in W13**: `ContextHeader.situating_sentence` / `model_id` stay defaulted
`None` so the slice is byte-identical + offline (no model, no key in the default path). Building it is a
separate, opt-in Provider — this ADR reserves the field and the seam.

## Consequences

- The RAG corpus a vector DB ingests now carries the KG coherence in its vectors + filterable metadata
  columns, closing the "embed each chunk in isolation" gap — the SOTA Contextual-Retrieval + KG-augmented
  move, on a framework that already emits an evidence-linked KG.
- Enrichment is a **pure decomposition**: only the additive field + the OPTIONAL Stage + the reference
  Provider + the Export prepend are new. No extractor, no offset, no Provenance, no existing contract
  changes; a stack without the Stage is byte-identical.
- The Quality Report gains a counts-only `context_enrichment` roll-up (`chunks_enriched` /
  `avg_neighbors_per_chunk` / `hub_entities_capped` — the anti-bloat witness), safe to persist/share.
- **Follow-ons** (documented, not built): the LLM situating-sentence Provider (§6); a learned
  neighbor-ranker behind the same seam (the bake-off baseline the reference gives). Query-time KG
  expansion stays the retriever's job (ADR-0017) — W13 only emits the join keys.
