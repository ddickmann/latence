# Latence Framework

A model-agnostic, cloud-agnostic framework that reads documents and writes a corpus that carries its Provenance and a knowledge graph that carries its Evidence: parsing, entity and relation extraction, PII redaction, disambiguation and graph assembly. The framework defines the stack and orchestration contracts; the concrete models and infrastructure are the adopter's choice.

This glossary has two parts. Everything down to *Decided, not yet in code* names something the code does today. That last section holds the vocabulary of the re-scope (ADR-0067, ADR-0069 to ADR-0073), which is decided and not built.

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

**Page map origin**:
Whether a page map's boundaries were MEASURED (the Parser segmented the source, or a page-map sidecar supplied them alongside pre-paginated markdown) or ASSUMED (the document carried no page structure at all, so its single span is a convention and "page 1" is not a claim). Recording it is what makes a missing page map detectable instead of indistinguishable from a real one-page document (ADR-0060).
_Avoid_: page source, page confidence (it is not a score — it is provenance of the boundaries)

**Chunk**:
The Stage that splits parsed documents into smaller pieces (by token budget, by sentence window or by page) while preserving offsets, page alignment, Provenance, and Classification.
_Avoid_: Enrichment (banned project-wide — it meant chunking in superpod and feature computation in dataset-intelligence)

**Entity Extraction**:
The Stage that finds typed entity mentions in text (zero-shot NER and friends).
_Avoid_: Extraction (unqualified)

**Relation Extraction**:
The Stage that finds typed relations between entity mentions within a document.
_Avoid_: Ontology (reserved for the schema level of the re-scope: classes, their hierarchy, properties and constraints. Nothing in the code builds one; see *Decided, not yet in code*)

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
The corpus-level Stage that builds the canonical knowledge graph — nodes, edges, and per-edge Evidence — from disambiguated entities and relations.

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
_Avoid_: Provenance (Provenance is source lineage; Evidence is justification for an inference)

**Embedder**:
A Capability that maps text to a fixed-width vector. It is never a Stage of its own: a Stage names one in a nested `embedder` block, and the resolution phase (Disambiguation, Type Consolidation) uses it as one signal among several (ADR-0068).
_Avoid_: Retrieval embedder, signal generator

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
The default deletion disposition — a soft tombstone: the document and its derived records are excluded from every Export, the knowledge graph, and the exported corpus, but retained for audit and rollback, with affected clusters re-resolved.
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

## Decided, not yet in code

The re-scope (ADR-0067) fixed the vocabulary below in ADR-0069. ADR-0070 to ADR-0073 use it. **No code implements any of these terms.** No record type, Stage or file in the repository carries them. They are here so that pages and discussions use one word for one thing before the first slice is built. Each entry is taken from the decision text and is marked *decided, not yet in code*.

**Ledger**:
The append-only record of Assertions and Decisions, and the engine's one source of truth. Nothing in the semantic layer exists without at least one Assertion behind it, and the layer anyone reads is a deterministic replay of the ledger. A correction is a new Decision, never an edit of an old record.
_Status_: decided (ADR-0069), not yet in code.
_Avoid_: Audit log (a log records that something changed; it cannot rebuild the layer)

**Assertion**:
One statement harvested from one source. It always carries a Locator and the Actor that produced it.
_Status_: decided (ADR-0069), not yet in code.

**Locator**:
Where an Assertion came from: a text span, a Unity Catalog object, a query, a lineage edge or a value sketch.
_Status_: decided (ADR-0069), not yet in code. Today a record points at its source through Provenance, which requires a file and character offsets.

**Decision**:
An accept, reject, merge, split or edit made on top of Assertions, with its Actor and the evidence it used. A human Decision outranks the engine and is never overwritten by a later run.
_Status_: decided (ADR-0069), not yet in code. Today the merge log records what the resolver decided; no record type holds a person's accept or reject.

**Actor**:
Who produced an Assertion or made a Decision: the engine (rules and calibrated bands), a model (with its name and version) or a human.
_Status_: decided (ADR-0069), not yet in code.

**Concept**:
A schema-level element: a class, a property or relation, or a measure or dimension. A Concept is only ever a proposal until a human approves it.
_Status_: decided (ADR-0069, ADR-0073), not yet in code. Today the type vocabulary is a flat list of canonical types with their aliases.

**Entity**:
One canonical real-world thing, made by merging records and mentions. Entities form the instance level.
_Status_: decided (ADR-0069), not yet in code.
_Not to be confused with_ the two ways the Stages above already use the word. An *entity mention* is a typed span of text that Entity Extraction finds. A *canonical entity* is what Disambiguation produces from mentions and what Graph Assembly writes as a node. Both meanings stay, and the Stage names stay. The ADR-0069 Entity is the second meaning widened: it also merges warehouse records, it is rebuilt by replaying the Ledger, and it can carry a Binding. Until that is built, write "mention" for a span and "canonical entity" for a node, and keep the capitalised "Entity" for the ADR-0069 sense.

**Binding**:
A tie from the semantic layer to a Unity Catalog object, keyed on the object's id and carrying a role (for example *defines*, *identifies*, *measures*). A *Concept binding* ties a Concept to a table, column or metric-view measure. An *Entity binding* ties an Entity to the rows that represent it.
_Status_: decided (ADR-0069), not yet in code.

**Accept band**:
The calibrated band inside which the engine may accept an instance-level merge or link by itself. Outside it the engine accepts nothing alone, and a model's verdict accepts nothing by itself either.
_Status_: decided (ADR-0069), not yet in code. Today the resolvers merge on rules and configured thresholds, and no band exists.

**Review band**, **reject band**:
Not fixed by an accepted decision. ADR-0069 names only the accept band. Splitting what lies outside it into a band a person reviews and a band that is dropped is a candidate design that has not become an ADR. Do not write either term as decided.
