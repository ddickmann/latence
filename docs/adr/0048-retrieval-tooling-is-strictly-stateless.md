# Retrieval-quality tooling is strictly stateless — it holds no index, it generates signals and transforms candidates

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

The framework grows a **retrieval-quality tooling** layer (embedding/sparse/multi-vector generation,
reranking, fusion, context-packing, graph-augmented retrieval, an MCP server) that sits **downstream**
of the pipeline export. This ADR records the load-bearing boundary that constrains the entire layer:
the tooling **never owns durable state** — no persisted index, no stored vectors, no standing graph
service. It only **generates signals** (text → vectors/term-weights, as files) and **transforms
candidate lists** (query + candidates → reranked / fused / packed / expanded). This is the operational
meaning of the product promise: *"we provide the machinery for retrieval, not the retrieval engine."*

## Context

Enterprises already run a retrieval engine — Databricks Mosaic AI Vector Search, Qdrant, Weaviate,
Azure AI Search, Elastic, or a proprietary stack. We do not know which, and we do not want to decide
(the per-stack loading story lives in `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`; this ADR originally
cited a planned interop document that was never written — reference corrected, decision unchanged).
The pipeline already stops at *files* (ADR-0017: "the AI-ready deliverable is
a RAG corpus **and** a knowledge graph, both as files" — never a live database). The retrieval tooling
is the first in-repo code to cross to the **consumer** side of that boundary, so it must not quietly
reintroduce the database ADR-0017 refused.

## Decision

Every component is one of exactly two stateless shapes:

- **Signal generator** — pure transform, text → a vector / term-weight map / multi-vector, materialised
  as files (index-time; see ADR-0049). Holds nothing.
- **Candidate processor** — pure transform, `(query, candidate list) → reordered / fused / packed /
  expanded list`. The candidates are supplied by the caller (their engine ran the first-stage search);
  we never fetch them ourselves from a store we own.

Consequences that follow directly, recorded so they are not "fixed" later by mistake:

- **No first-stage search lives here.** The customer's engine executes every ANN / sparse / BM25
  lookup. Our **BM25 is therefore a *rescorer* over a caller-supplied candidate set**, plus an optional
  index-time *artifact* exporter — **never** a first-stage inverted index. Same logic for sparse and
  dense: we produce the vectors as data; their store indexes and searches them.
- **Graph-augmented retrieval expands over data passed in per call** (the KG columns denormalised onto
  chunks, or the emitted graph parquet read read-only for one call), never a standing graph service we
  operate (ADR-0051).
- **The MCP server holds no corpus** — it is a bring-your-own-backend orchestrator (ADR-0052): the
  customer wires *their* engine as the first-stage callback; we apply the stateless tooling around it.

## Considered alternatives

- **An embedded/ephemeral index escape hatch** (build an in-memory index for MCP demos / small corpora /
  offline eval). Rejected: it is the exact crack through which "just a little state" becomes a system of
  record, and it muddies the promise. Self-contained end-to-end operation is explicitly **not** a v1
  goal; the customer's engine is always in the loop for first-stage search.

## Consequences

- The layer is honest to "we don't decide the retrieval technology," and every piece is independently
  adoptable à la carte (call `rerank`/`fuse`/`expand` on results you already have) without buying into
  an orchestrator.
- It sidesteps ADR-0017's no-database line entirely: the tooling holds nothing durable, so there is
  nothing to govern, persist, or operate.
- The cost — accepted deliberately — is that the tooling **cannot answer a query on its own**; without
  a customer engine (or, for multi-hop, the emitted graph files) there is no first-stage retrieval.
