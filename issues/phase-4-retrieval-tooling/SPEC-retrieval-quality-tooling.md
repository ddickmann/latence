# Spec: Retrieval-quality tooling

> **Status:** ready-for-agent. Synthesised from the design grill; decisions recorded in ADRs
> [0048–0053](../../docs/adr/index.md) and [`docs/retrieval-tooling.md`](../../docs/retrieval-tooling.md).
> This spec is the build contract for v1.

## Problem Statement

An enterprise has run its messy documents through the latence pipeline and now holds a clean, enriched,
governed corpus — chunks, metadata, an evidence-linked knowledge graph, and an optional dense embedding —
as files in its own object storage. It already runs a retrieval engine (Databricks Vector Search, Qdrant,
Weaviate, Azure AI Search, Elastic, or a proprietary stack). What it does **not** have is the *quality
machinery* that turns "a vector index" into *SOTA retrieval*: strong multi-signal embeddings, a sparse
signal that exploits the entities and metadata it already extracted, reranking, principled fusion,
token-budgeted context assembly, graph-augmented expansion, and a hardened agentic-search surface. Today
each team reinvents these, badly, and couples them to one stack — or ships plain single-vector retrieval
and calls it done.

The team does not want to be told which retrieval engine to use, and does not want a second database to
operate. It wants the *intelligence and tooling*, plugged into the stack it already has.

## Solution

latence provides a **retrieval-quality tooling** layer that gives any retrieval engine SOTA quality
**without being one**. It never holds an index and never runs first-stage search — the customer's engine
always does that. The layer only **generates signals** (as more AI-ready files, at index time) and
**transforms candidate lists** (at query time), and offers an optional, hardened **MCP server** that wraps
the customer's engine for agentic search.

Concretely, an adopter can:

- Generate richer retrieval signals from the same pipeline run — a structured **SPLADE** sparse vector
  composed from the model plus the pipeline's own NER + metadata, a **MUVERA-FDE** dense encoding of
  multi-vectors, and graph features (centrality, community) — all emitted as columns/sidecars they load
  into their store.
- Call **one-liners** on results their engine returned — `fuse` several ranked lists, `rerank` with a
  cross-encoder, `pack` a token-budgeted subset for an LLM context window, `expand` via the knowledge
  graph — with zero knowledge of any engine.
- Optionally let latence **orchestrate** the whole flow around their engine (`retrieve(query, backend,
  pipeline)`), or expose that orchestration to agents through an **MCP server** that auto-adapts its tool
  descriptions to their stack via config and ports latence's hardened security/observability.

Every piece is independently adoptable, stack-agnostic, and stateless.

## User Stories

1. As a retrieval engineer, I want to generate dense embeddings for my corpus with a model I choose (open-source, self-hosted, OpenAI, or Gemini), so that I control cost, quality, and data residency.
2. As a retrieval engineer, I want to ship my corpus text-only and let my own store compute embeddings, so that I can use my store's native embedding pipeline instead of latence's.
3. As a retrieval engineer, I want a multilingual SPLADE sparse vector generated for every chunk, so that I can run hybrid dense+sparse retrieval in my store.
4. As a retrieval engineer, I want the sparse signal to also incorporate the entities and metadata the pipeline already extracted, so that my sparse retrieval is grounded in domain structure a plain SPLADE model would miss.
5. As a compliance-conscious engineer, I want the sparse signal generated over the redacted chunk text by default, so that it never re-leaks PII the corpus already removed.
6. As a retrieval engineer, I want weak entity extractions excluded from the sparse composition, so that poor NER on some chunks does not pollute the sparse signal.
7. As a retrieval engineer, I want a MUVERA-FDE encoding that turns multi-vector embeddings into a single dense vector, so that I can approximate late-interaction quality inside my ordinary dense index with no special store.
8. As a retrieval engineer, I want the option to also persist the raw multi-vectors, so that I can do exact MaxSim reranking later.
9. As a cautious adopter, I want the multi-vector/FDE path to be clearly experimental and off by default, so that I am never silently relying on an unproven conversion in production.
10. As an engineer evaluating options, I want the experimental FDE path to actually run (not a stub that errors), so that I can measure its quality on my own corpus before trusting it.
11. As a retrieval engineer, I want to rerank a candidate list with a cross-encoder model I select in config, so that I improve top-k precision with a one-liner.
12. As a retrieval engineer, I want to fuse several ranked lists (dense, sparse, BM25) into one, so that I combine signals without writing fusion logic myself.
13. As a retrieval engineer, I want a stronger fusion than plain RRF when I need it, so that fusion can use scores and penalise redundancy rather than only ranks.
14. As a RAG builder, I want to select a token-budgeted subset of candidates that maximises coverage and minimises redundancy, so that I fill an LLM context window with the most useful, least repetitive evidence.
15. As a RAG builder, I want context packing to accept optional richness signals (centrality, roles, clusters) when I have them and degrade gracefully when I do not, so that I get value on day one and better results as I feed it more.
16. As a retrieval engineer, I want a BM25 rescorer over a candidate set my engine already fetched, so that I add exact lexical scoring without standing up a second index.
17. As a retrieval engineer, I want an optional BM25 term-statistics artifact I can load into my store's BM25, so that I can drive first-stage lexical search in the engine I already run.
18. As a knowledge-graph-aware engineer, I want 1-hop graph expansion using the KG columns already on each chunk, so that I get graph-augmented retrieval on any vector store with no graph database.
19. As a knowledge-graph-aware engineer, I want multi-hop traversal without operating a graph database, so that I get deeper graph retrieval at near-zero operational cost.
20. As an engineer with an existing Neo4j deployment, I want an optional Neo4j graph backend, so that I can reuse infrastructure I already run for multi-hop.
21. As a retrieval engineer, I want graph features (centrality, community) computed at index time and attached to chunks and nodes, so that both graph expansion and context packing can use them.
22. As an à-la-carte user, I want to call each transform on results I already have, so that I can adopt one capability without buying into an orchestrator or the MCP server.
23. As a convenience-seeking user, I want a single `retrieve` call that chains query-signal-gen, my engine's search, and the transforms, so that I get a SOTA pipeline in one line.
24. As an integrator, I want to plug my engine in behind a small adapter that declares what it can do, so that the orchestrator uses the best pipeline my store supports and skips what it cannot.
25. As an integrator, I want a reference Qdrant backend adapter, so that I have a working example to copy for my own store.
26. As an agent developer, I want an MCP server that exposes retrieval as tools, so that my agent can search the corpus through a standard protocol.
27. As an agent developer, I want the MCP server to wrap my own retrieval engine, so that agentic search uses my stack, not a bundled one.
28. As an operator, I want all MCP tool names, descriptions, parameter docs, and the tool hierarchy loaded from config at runtime, so that I can tailor the server's guidance to my stack and domain without changing code.
29. As an operator, I want strong default MCP copy out of the box, so that the server is excellent before I customise anything.
30. As an operator, I want the MCP tool descriptions to auto-tailor to my backend's declared capabilities, so that the agent is told about hybrid search only when my store supports it.
31. As a security officer, I want the MCP server to require authentication, bind to loopback, and rate-limit, so that an agentic-search surface does not become an exposure.
32. As a security officer, I want the MCP copy config to be operator-controlled and not agent-mutable at runtime, so that the server's own tool descriptions cannot become a prompt-injection vector.
33. As an SRE, I want structured, rotating logs and per-call records from the MCP server, so that I can observe and debug agentic retrieval in production.
34. As a RAG builder, I want every returned chunk to carry its provenance (source, page, offsets, confidence), so that I can cite sources and show groundedness.
35. As a retrieval engineer, I want to filter candidates on any metadata field, so that I can scope retrieval by matter, tenant, sensitivity, or date.
36. As an adopter, I want to swap the reranker model, fusion algorithm, or sparse model via config, so that tuning retrieval quality is a one-line change.
37. As an architect, I want a guarantee that the tooling holds no index and runs no first-stage search, so that I never acquire a second system of record or a hidden database to govern.
38. As a legal/compliance owner, I want the tooling to be Apache-2.0 with no non-permissive or ambiguously licensed code in the default install, so that I can adopt it without a licensing review blocking launch.
39. As a maintainer, I want the SOTA MaxSim kernels and the Knapsack packer to be portable Apache-2.0 components, so that I can ship them without relicensing friction.
40. As a performance-sensitive engineer, I want BM25 and MaxSim scoring to be fast on large candidate sets, so that reranking many queries stays within latency budgets.
41. As an offline/CI user, I want the core library to work with pure-Python reference implementations and no heavy dependencies, so that I can develop and test without GPUs or large model downloads.
42. As an adopter, I want heavy capabilities behind optional extras, so that I install only what my stack needs.
43. As a lakehouse user, I want graph multi-hop to run over the parquet files the pipeline already emits, so that I add no new artifact and no server to my architecture.
44. As an adopter on Databricks or Azure, I want the backend-adapter contract to be documented so a future adapter can be written, so that I am not blocked when Qdrant is not my store.

## Implementation Decisions

**Boundary & shape (ADR-0048, ADR-0049).**
- The layer is strictly stateless: two component archetypes only — **signal generators** (index-time, text → vector/term-weights/multi-vector, emitted as files) and **candidate processors** (query-time, `(query, candidates) → transformed candidates`). No component holds a durable index, stored vectors, or a standing graph service; none runs first-stage search.
- **Clock-split.** Index-time signal generation extends the pipeline (new Providers under the existing `latence.providers` seam, emitted during `latence process`). Query-time tooling is a new downstream library with its **own** registry seam, `latence.retrieval`, kept off the Stage DAG.

**Index-time signal generators (Providers, files).**
- Dense reuses the existing `Embedder` Capability (any OpenAI-compatible endpoint, sentence-transformers, or the in-core reference), unchanged.
- New sibling Capabilities: `SparseEmbedder` (text → sparse term-weight map), `MultiVectorEmbedder` (text → variable-length token vectors), and `FdeConverter` (multi-vector → one dense vector).
- Emission contract: SPLADE → parallel `sparse_indices` (int list) + `sparse_values` (float list) columns; FDE → an `fde_embedding` dense column intended for the customer's dense index; raw multi-vectors → a **sidecar** table keyed by record id (not in the main corpus table); an optional BM25 term-statistics artifact. All additive and versioned per the framework's contract discipline.
- Graph Assembly is extended to compute **centrality** (degree/PageRank) and **community** (Louvain/Leiden), written as graph-node columns and projected onto each chunk's existing context header — feeding both graph expansion and context packing.

**SPLADE composition (ADR-0053).**
- The sparse vector is a composition, not plain `SPLADE(text)`: `SPLADE(text) ⊕ boost·entity_terms ⊕ boost·metadata_terms` (term-injection default; a weighted-combination mode optional). Input text follows the corpus redaction policy by default. Injected entity terms are gated by the pipeline's `EntityMention` confidence. The multilingual sparse model is config-selected and license-verified (weights and code separately) before wiring; nothing is hardcoded.

**Multi-vector / FDE stub (ADR-0050).**
- Ship the three seams fully wired, plus a **real, training-free, canonical MUVERA FDE reference** and a reference MaxSim scorer — marked experimental and off by default. No fake/passthrough that pretends to be FDE (anti-false-green). Deferred: any regularizer-refined converter, a bake-off that blesses a production default, on-by-default multi-vector, and the SOTA MaxSim kernels as an accelerator.

**Query-time library (`latence-retrieval`).**
- Two consumption levels: **Level 1** — à-la-carte one-liners `fuse`, `rerank`, `pack`, `expand`, and query-signal-gen, each a pure transform over caller-supplied candidates; **Level 2** — an optional orchestrator `retrieve(query, backend, pipeline)` that chains query-signal-gen → the customer's first-stage search → the Level-1 transforms.
- Candidate processors are three distinct operations, named precisely (not lumped as "fusion"): **fuse** (merge N ranked lists → one; RRF and weighted), **rerank** (re-score one list; cross-encoder real, MaxSim experimental), **pack** (token-budgeted subset maximising coverage − redundancy; the ported Knapsack QKP), and **expand** (graph-augmented: 1-hop default, DuckDB multi-hop, optional Neo4j).
- The Knapsack packer accepts optional richness signals (relevance, token cost, redundancy/similarity, centrality, recency, and BYO roles/clusters/fulfilment/coverage) and degrades gracefully when any are absent. Its per-query-token coverage input rides the deferred multi-vector path; the packer runs on every other signal until then.

**Bring-your-own engine (ADR-0052).**
- A `RetrievalBackend` adapter declares its capabilities (a set drawn from dense, sparse, bm25, filter, multivector) and runs first-stage search returning candidates. The orchestrator composes the best pipeline the declared capabilities and emitted signals allow, and drops any leg the store lacks. A reference Qdrant adapter ships first; the protocol is the contract and the library depends on no specific engine.

**Graph-augmented multi-hop (ADR-0051).**
- Default 1-hop over the denormalised KG columns (any store, no graph DB). Zero-ops multi-hop = an embedded engine (DuckDB) querying the graph parquet the pipeline already emits, read-only per call — no new artifact, no server. A pure-Python (networkx) reference for tiny graphs, and an optional Neo4j backend for shops that already run it. (Kùzu was rejected: upstream archived after acquisition.)

**MCP server (ADR-0052).**
- Ports the maintainer's `memory_engine` hardening (bearer-token 0600 + constant-time auth, loopback + UDS transport, rate limiting, WAL, structured rotating JSON logs, the response-envelope/error contract, per-call records, threat model). Replaces the ColPali/MaxSim guts with the `RetrievalBackend` seam; drops subscription/entitlement and the Electron/tunnel bits.
- **All agent-facing copy is operator config loaded at runtime** — tool names, descriptions, parameter docs, operating-policy, and the escalating-ladder hierarchy — with strong defaults modelled on `memory_engine`, auto-tailored by the backend's declared capabilities, overridable per stack/domain. The copy config is operator-trusted, sandboxed, schema-validated, and not agent-mutable. Tool surface keeps the escalating-ladder shape retargeted to stack-agnostic text + graph (`search`, expand/more-like-this, get-context returning grounded chunks + provenance, list-sources, index-status). v1 is local/loopback + bearer; remote transport (OAuth/mTLS) is seamed for later.

**Packaging & licensing (ADR-0049; provenance in `docs/retrieval-tooling.md`).**
- Core `latence-retrieval` ships pure-Python reference implementations (RRF, numpy MaxSim reference, networkx graph, basic BM25). Heavy deps behind extras (splade, multivector, fde, knapsack, graph-duckdb, mcp, backend-qdrant), mirroring the thin-core / per-provider-package discipline.
- The mined assets (`memory_engine`, and `colsearch` = the Knapsack QKP packer + MaxSim kernels) are the maintainer's own projects relicensed Apache-2.0; v1 carries no non-permissive or third-party-encumbered code. The one third-party remnant (an unsloth-adapted FP8 MaxSim kernel) rides the deferred kernel path and carries its own license-clearance task.

## Testing Decisions

- **What a good test is here:** it exercises *external behaviour* at a seam, never implementation details. For a candidate processor, that means "given these input candidates and query, assert the transformed output" — not the internal scoring maths. For the MCP server, "call this tool, assert the response envelope and security behaviour" — not the handler internals.
- **Seam 1 (reused, existing):** index-time signal generators are tested at the **Provider conformance suite (C1–C6)** plus the **export/contract** check plus **`latence stack validate`** — the same seam every pipeline Provider already uses. Assertions: the new columns/sidecars are emitted with the right shapes; the SPLADE composition includes confidence-gated entity/metadata terms; redaction-by-default holds; graph features appear on nodes and chunks. Determinism and additive-contract compatibility are asserted as for existing Providers.
- **Seam 2 (one new seam):** all query-time tooling is tested through the **`RetrievalBackend` adapter + the `latence.retrieval` component interfaces**, driven by a **faithful fake backend**. The fake must reject exactly what a real store rejects (capability mismatches, missing collections, dimension mismatches) — anti-false-green. Coverage: each Level-1 transform at its input→output boundary; the Level-2 orchestrator's capability-adaptive pipeline (including that it drops legs the fake backend does not declare); the experimental FDE/MaxSim path produces real (not faked) output.
- **MCP server** is tested at the **tool-call boundary** using the same fake backend: assert the envelope/error contract, that authentication is required, that rate-limiting engages, that no state is retained across calls, and that config-driven copy renders (and that agent input cannot mutate the copy config).
- **Prior art:** the provider ecosystem's faithful-fake backends (GLiNER, relex, OCR, ULTRA) and their pod-validation pattern; the conformance suite; `latence stack validate`; the anti-false-green rule that offline fakes must mirror/reject what the real API does.
- **Real-model / real-store validation** (like the pipeline's pod sweeps) is done out-of-band with real weights and a real reference store, and its numbers folded back — never fabricated in unit tests.

## Out of Scope

- **The retrieval engine itself.** First-stage ANN/sparse/BM25 search always runs on the customer's store. This tooling is upstream of it and is explicitly not a vector database, search engine, or index.
- **The production multi-vector→dense converter decision.** The LightOn-regularizer-refined converter and any bake-off that blesses a production FDE default are deferred; v1 ships the canonical FDE reference experimental and off by default.
- **The SOTA MaxSim kernels (Triton/MLX/SIMD) as an accelerator** — deferred with the multi-vector path; the unsloth-derived FP8 kernel needs its own license clearance.
- **Remote MCP transport** (OAuth 2.1 / mTLS). v1 is local/loopback + bearer; the transport is seamed for a later adapter.
- **Backend adapters beyond Qdrant** (Databricks, Azure, Elastic, pgvector, Weaviate). Seam-compatible follow-ons; only the contract and the Qdrant reference ship in v1.
- **Multimodal (page-image) MCP results.** A later option; v1 returns grounded text chunks + provenance.
- **Retrieval evaluation harnesses / quality dashboards** beyond what conformance and stack-validate already provide.

## Further Notes

- The whole layer is the consumer-side complement to `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` (which describes how the pipeline's output loads into each stack). It honours ADR-0017 (files, not a database) by holding nothing durable.
- The MUVERA reference implements the canonical, training-free Google MUVERA FDE. Note that the article originally cited as the "FDE spec" is in fact the LightOn hierarchical-pooling *regularizer* (a different, training-required method), which is why it is deferred rather than adopted.
- Build sequencing suggestion (not binding): index-time signal generators first (they extend a well-understood seam and unblock evaluation), then the Level-1 query-time transforms behind the `RetrievalBackend` fake, then the Level-2 orchestrator, then the MCP server. Graph multi-hop (DuckDB) and the Knapsack packer are the two heaviest ports and can proceed in parallel behind their extras.
- Every model choice (sparse, cross-encoder, and any future multi-vector encoder) must pass separate weights-and-code license verification before it is wired, per the research-diligence directive and ADR-0012.
