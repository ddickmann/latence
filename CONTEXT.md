# Latence Framework

A model-agnostic, cloud-agnostic framework that turns messy, siloed enterprise data into high-quality, AI-ready data — from document parsing through NER, relation extraction, PII handling, enrichment, and disambiguation to corpus-level knowledge graphs. The framework defines the stack and orchestration contracts; the concrete models and infrastructure are the adopter's choice.

## Language

**Stage**:
A unit of pipeline work with a typed input and output contract (e.g., parsing, entity extraction). Stages declare what they consume and produce; they do not know how they are scheduled or which model fulfils them.
_Avoid_: Service, step, task

**Runner**:
An execution adapter that schedules Stages on a particular substrate. Two ship today: the built-in local runner and the Airflow DAG adapter (`latence-runner-airflow`); further substrates (a Databricks job, a plain container scheduler) are seam-compatible follow-ons rather than shipped packages. Runners own scheduling, retries, and checkpointing; they never define pipeline semantics.
_Avoid_: Orchestrator, worker, control plane

**Pipeline**:
A DAG of Stages over a set of input documents, declared once in the framework and executable by any Runner.
_Avoid_: Workflow, job chain

## Stages

**Source**:
The Capability that lists and fetches documents from where they live (a folder, an object store, or — via connectors — a source system like SharePoint), stamping initial Provenance. The entry point that makes siloed data addressable.
_Avoid_: Connector (a Connector is one Provider of the Source Capability), ingester, loader

**Parse**:
The Stage that turns a source document (PDF, image, office file) into markdown plus a page map preserving original page boundaries.
_Avoid_: Document intelligence, doc intel, OCR (OCR is one Provider technique, not the Stage)

**Chunk**:
The Stage that splits parsed documents into retrieval-sized pieces while preserving offsets, page alignment, Provenance, and Classification.
_Avoid_: Enrichment (banned project-wide — it meant chunking in superpod and feature computation in dataset-intelligence)

**Entity Extraction**:
The Stage that finds typed entity mentions in text (zero-shot NER and friends).
_Avoid_: Extraction (unqualified)

**Relation Extraction**:
The Stage that finds typed relations between entity mentions within a document.
_Avoid_: Ontology (reserved exclusively for formal ontology induction, a post-v1 plugin)

**Redaction**:
The Stage that detects PII and produces masked/replaced variants of the text.

**Screening**:
The Stage that catches dangerous data early, at two checkpoints: Intake Screening before Parse (malware, zip bombs, file-type spoofing, oversized/corrupt files) and Content Screening after Chunk (prompt injection, harmful content, sensitivity escalation). Screening either Quarantines a record or flags it with risk markers that propagate downstream.
_Avoid_: Guardrails (connotes runtime LLM I/O filtering), moderation

**Profiling**:
The corpus-level Stage that computes statistical and quality features (density, readability, Zipf, cross-document entity frequency, co-occurrence).
_Avoid_: Enrichment

**Disambiguation**:
The corpus-level Stage that links entity mentions to canonical entities and merges duplicates across documents.
_Avoid_: Entity linking / entity resolution as stage names (they name the two internal steps, not the Stage)

**Graph Assembly**:
The corpus-level Stage that builds the canonical knowledge graph — nodes, edges, and per-edge Evidence — from disambiguated entities and relations, and re-projects the same input m-ary as Hyperedges (ADR-0057).

**Export**:
The Stage that materializes AI-ready outputs (Parquet, JSONL, TTL, GraphML).

## Cross-cutting

**Provenance**:
The immutable chain from any record back to its source: file name, file type, file size, source system, document id, page and character offsets. Every record at every Stage carries it; Chunk must preserve it losslessly.
_Avoid_: Lineage, metadata (unqualified)

**Classification**:
Descriptive attributes of a document's content — language, content category, sensitivity — attached after Parse and inherited by every downstream record.
_Avoid_: Tagging, labeling

**Quarantine**:
The disposition for records Screening removes from the pipeline: excluded from all downstream Stages and exports, but retained and inspectable with the reason recorded.
_Avoid_: Delete, drop, filter (unqualified)

**Evidence**:
The record linking a knowledge-graph edge (or a Disambiguation merge decision) back to the specific source mentions and offsets that justify it, with a confidence value.
_Avoid_: Provenance (Provenance is source lineage; Evidence is justification for an inference), Hyperedge (a Hyperedge *carries* raw text for retrieval; Evidence never does)

**Hyperedge**:
The m-ary unit of retrieval: a minimal verbatim text span, the full set of canonical entities occurring in it, the directed relations extracted from it, and its token cost — emitted by Graph Assembly as a third `GraphScope` alongside nodes and edges (ADR-0057). Where an edge is binary, a Hyperedge connects m ≥ 2 entities at once, so a joint fact ("A acquired B in 2019 pending approval by C") stays one selectable unit instead of fragmenting into disconnected pairs. It is the atomic thing the chain-of-evidence selector packs into a token budget. Its `tau_text` never holds content the exported corpus would not hold.
_Avoid_: Evidence Unit, EU (collides with Evidence), fact, claim (both assert truth; a Hyperedge asserts only that the text says so)

**Quality Report**:
A first-class artifact emitted per run: per-Stage metrics, corpus-level KG statistics, contract-completeness checks, and optional gold-set precision/recall. For a delta run it also reports what changed — documents added/retracted, entities created/merged/split, edges added/retracted, and drift since the last reconciliation. The substantiation of the "AI-ready" claim.
_Avoid_: Metrics dump, stats, log

## Corpus deltas

**Corpus Version**:
An immutable, numbered committed state of the corpus on Storage. A Delta run reads Version N and transactionally produces Version N+1; any prior Version is inspectable for audit and rollback.
_Avoid_: Snapshot (reserved for the RotatE GraphSnapshot), commit

**Delta**:
The set of changes since the last committed Corpus Version, computed as a content-hash diff of the Source manifest: added, updated (same identity, new hash), deleted, or unchanged documents.
_Avoid_: Diff, changeset, increment

**Affected set**:
The blocking neighborhoods (and their immediate neighbors) touched by a Delta's changed mentions — the only region corpus-level Stages recompute, rather than the whole corpus.
_Avoid_: Dirty set, impacted region

**Reconciliation**:
A full recompute of all blocks, triggered when a drift metric crosses a threshold, that corrects the order-dependent error accumulated by incremental Affected-set recomputation.
_Avoid_: Full rebuild, refresh

**Retraction**:
The default deletion disposition — a soft tombstone: the document and its derived records are excluded from every Export, the knowledge graph, and the RAG corpus, but retained for audit and rollback, with affected clusters re-resolved.
_Avoid_: Soft delete, archive

**Purge**:
The hard deletion operation for legal erasure (GDPR / right-to-be-forgotten): physical removal of the source, all derived records, and their Evidence, followed by reconciliation of the Affected set.
_Avoid_: Hard delete, wipe

**Capability**:
A narrow, model-agnostic interface a Stage depends on to do its work (e.g., Parser, EntityExtractor, PIIDetector), typed against the framework's data contracts.
_Avoid_: Model, service, backend

**Provider**:
A concrete implementation of a Capability — an in-process model, a remote inference endpoint, or a cloud AI service — registered as a plugin. The framework never names a model; it names Capabilities and lets Providers fulfil them.
_Avoid_: Adapter, integration, connector

**Fused Provider**:
A single Provider that fulfils more than one adjacent Stage's Capability in one pass (e.g., joint NER + Relation Extraction). The Pipeline permits this; Capabilities are not forced one-per-Stage.
_Avoid_: Multi-task model (implementation detail, not the seam concept)

**ParserInput**:
The raw-document contract the Parser consumes: Provenance and Classification plus the *undecoded* source `content` (bytes, or str if already textual) and a per-input `config` of Parser hints. Produced by Source, it is the input side of the Parse seam — decoding bytes into markdown belongs to the Parser, which turns a ParserInput into a DocumentRecord (ADR-0019), so the seam fits real bytes-consuming parsers, not just the passthrough.
_Avoid_: Seed record, raw record (unqualified), pre-parsed document

**Raw document**:
A source document as first read by Source — its bytes plus stamped Provenance and Classification, before any Parse decoding — carried as a ParserInput. Contrast the DocumentRecord, which is the *parsed* (markdown) form the Parser produces.
_Avoid_: Blob, payload, input record (unqualified)

## Retrieval tooling

The stateless, downstream layer that gives an adopter's retrieval engine SOTA quality without being one (ADRs 0048–0053). It never holds an index; it only generates signals and transforms candidate lists.

**Retrieval tooling**:
The whole downstream layer (signal generators + candidate processors + the MCP server) that sits on the consumer side of Export. It is *not* a retrieval engine — it never runs first-stage search or holds durable state (ADR-0048).
_Avoid_: Retrieval engine, retriever, search engine, index

**Signal generator**:
An index-time pure transform that turns chunk text into a vector / term-weight map / multi-vector, materialised as files during `latence process` (dense Embedder, SparseEmbedder, MultiVectorEmbedder, FdeConverter). The index-time half of retrieval tooling.
_Avoid_: Encoder (unqualified), vectorizer

**Candidate processor**:
A query-time pure transform over a caller-supplied candidate list: fuse, rerank, pack, or expand. The query-time half of retrieval tooling; holds nothing, fetches nothing.
_Avoid_: Retriever, ranker (unqualified)

**Fuse**:
Merge N ranked candidate lists into one (RRF or weighted). Distinct from rerank and pack.
_Avoid_: Fusion (as a catch-all for rerank/pack), merge

**Rerank**:
Re-score a single candidate list with a stronger model (cross-encoder; MaxSim over multi-vectors is the experimental path, ADR-0050). Distinct from fuse and pack.
_Avoid_: Re-order, fusion

**Pack**:
Select a token-budgeted *subset* of candidates that maximises coverage and relevance minus redundancy — the ported Knapsack QKP context-packer. The final assembly before the LLM's context window. Distinct from fuse and rerank.
_Avoid_: Fusion, context selection (unqualified), truncation

**Expand**:
Graph-augmented widening/rescoring of candidates: 1-hop over the KG columns denormalised onto each chunk (default), or DuckDB multi-hop over the emitted graph parquet (ADR-0051).
_Avoid_: Graph search, traversal (unqualified)

**RetrievalBackend**:
The bring-your-own-engine adapter: a capability-declaring seam (`{dense, sparse, bm25, filter, multivector}`) whose `search` runs the customer's first-stage retrieval. The tooling orchestrates around it but never replaces it (ADR-0052).
_Avoid_: Vector store (as the seam name), connector, driver

**Retrieval MCP**:
The optional, stateless MCP server that wraps a RetrievalBackend with the candidate processors and exposes them as MCP tools; hardened from `memory_engine`, with all agent-facing copy loaded from operator config (ADR-0052).
_Avoid_: Retrieval server, search API

**FDE (Fixed-Dimensional Encoding)**:
A data-oblivious, training-free conversion of a multi-vector into one dense vector whose dot-product approximates MaxSim, so it lands in an ordinary dense index (MUVERA). Ships experimental and off by default (ADR-0050).
_Avoid_: Pooling (unqualified), flattening
