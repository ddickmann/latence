# S9 — Graph Assembly + Export

**Parent**: S8 · **Phase 1 · Baseline bar**

## What to build
The corpus-level **Graph Assembly** Stage builds the canonical knowledge graph — nodes, edges, and per-edge **Evidence** — from disambiguated entities and relations. **Export** materializes the two AI-ready deliverables on Storage (ADR-0017), never a live database: (1) a **RAG-ready corpus** — cleaned, chunked, Screened, PII-handled text with full Provenance/Classification as Parquet/JSONL, embeddings optional; (2) the **knowledge graph** as portable files (Parquet + TTL + GraphML) loadable into Neo4j, a triple store, or graph-RAG.

## Ports (algorithmic assets)
- Graph builder + EvidenceRecord provenance (`dataset-intelligence/graph/builder.py`). PORT-AS-IS.

## Acceptance criteria
- [ ] KG nodes/edges built with deterministic IDs; every edge carries Evidence (source docs/mentions/offsets) and confidence.
- [ ] Export produces the RAG-ready corpus (Parquet+JSONL) AND the KG (Parquet+TTL+GraphML) on `file://` and `s3://`.
- [ ] TTL validates as well-formed RDF; GraphML loads in a standard graph tool; Parquet loads as a Delta/Arrow table.
- [ ] Embedding is an optional opt-in Embedder Provider (default Embedder = `multilingual-e5-small`, MIT), not required for Export.
- [ ] KG statistics (node/edge counts, evidence coverage) land in the Quality Report.
- [ ] Baseline bar met — **this completes the first end-to-end messy-docs → KG demo.**

## Blocked by
S8.
