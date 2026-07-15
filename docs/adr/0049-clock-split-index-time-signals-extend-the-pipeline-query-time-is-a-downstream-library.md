# Clock-split: index-time signal generation extends the pipeline; query-time tooling is a downstream library with its own `latence.retrieval` seam

Retrieval tooling runs on two clocks with opposite constraints, and we split the architecture along
that seam. **Index-time** signal generation (dense, SPLADE-sparse, MUVERA-FDE, raw multi-vectors, BM25
term-stats, graph features) is **part of the pipeline** — produced during `latence process`, emitted as
more AI-ready files. **Query-time** tooling (rerank, fuse, pack, expand, query-signal-gen, the MCP
server) is a **separate stateless downstream library** with its **own provider seam**, `latence.retrieval`,
parallel to but distinct from the pipeline's `latence.providers`.

## Context

The two clocks are genuinely different work: index-time is offline, once-per-corpus, batch,
throughput-bound, and *produces data*; query-time is online, per-query, single-item, latency-bound, and
*transforms data*. The chunks, metadata, and knowledge graph are already produced by the pipeline and
are "the data that becomes the index" — the vectors are the same kind of thing, logically coupled to
that data and useless without it. Dense embedding is **already** a pipeline Capability (`Embedder`,
ADR-0045) emitted as the `embedding` column during Export. Forcing all signal generation into a separate
downstream tool would duplicate the Export/Embedder machinery and split "signal generation" across two
homes.

## Decision

- **Index-time signal generation extends the pipeline.** Dense reuses the existing `Embedder`; new
  sibling Capabilities are added for the shapes `Embedder` cannot express — `SparseEmbedder`
  (text → term-weight map), `MultiVectorEmbedder` (text → variable-length token vectors), and an
  `FdeConverter` (multi-vector → one dense vector). Graph features (centrality, community) extend the
  Graph Assembly stage. All emit **files** (columns / sidecars), so ADR-0017 holds unchanged
  (see ADR-0050 for the emission contract and the multi-vector stub).
- **Query-time tooling is a new stateless library** (`latence-retrieval`) exposing two layers:
  **Level 1** — à-la-carte one-liners (`rerank`, `fuse`, `pack`, `expand`, query-signal-gen) usable with
  zero knowledge of any engine, on candidates the caller already fetched; **Level 2** — a thin optional
  orchestrator, `retrieve(query, backend=…, pipeline=[…])`, that chains query-signal-gen → the
  *customer's* first-stage search → the Level-1 transforms. The MCP server (ADR-0052) is Level 2 with an
  MCP face.
- **A separate registry seam, `latence.retrieval`,** registers the swappable query-time components
  (the reranker model, the fusion algorithm, the sparse model) so they are config-selectable one-liners.
  These are **not** pipeline Stage Capabilities and are deliberately kept off the `latence.providers`
  group / the Stage DAG.

The query-time candidate processors are three distinct operations, named precisely rather than lumped
as "fusion": **fuse** (merge N ranked lists → one), **rerank** (re-score one list with a stronger
model), **pack** (select a token-budgeted subset maximising coverage − redundancy — the Knapsack
context-packer). See the glossary in `CONTEXT.md`.

## Considered alternatives

- **Everything downstream** (freeze the pipeline; the library re-reads `records.parquet` and generates
  *all* signals itself as sidecars). Rejected: it duplicates the Embedder/Export machinery and gives
  "signal generation" two homes. The accepted cost of the chosen split is that regenerating a signal
  means re-running the pipeline, not a standalone tool.

## Consequences

- The "EMBEDDING GENERATION" surface is *mostly the existing `Embedder` seam extended* with two new
  sparse/multi-vector generators — not a new subsystem.
- Heavy query-time deps (cross-encoders, the Rust knapsack, DuckDB, the MCP stack) live behind extras on
  `latence-retrieval`, mirroring the thin-core / per-provider-package discipline (ADR-0016); the core
  ships pure-Python reference implementations.
