# Retrieval-quality tooling (design scope)

> **Status: design, aligned, not yet built.** This page is the agreed scope from the design grill. The
> decisions are recorded as ADRs [0048](adr/0048-retrieval-tooling-is-strictly-stateless.md)–[0053](adr/0053-splade-sparse-signal-is-a-structured-composition-of-model-ner-and-metadata.md);
> this page is the navigational summary and the explicit v1 boundary.

## What this is (and is not)

latence provides the **machinery and intelligence for SOTA retrieval quality — not the retrieval engine
itself.** Enterprises keep their engine (Databricks Vector Search, Qdrant, Weaviate, Azure AI Search,
Elastic, or a proprietary stack); we do not decide or replace it (`docs/INTEROP.md`). The tooling is
**strictly stateless** (ADR-0048): it *generates signals* and *transforms candidate lists*, and holds no
index, no stored vectors, and no standing graph service.

Two clocks (ADR-0049):

- **Index-time** signal generation is **part of the pipeline** — emitted as more AI-ready files during
  `latence process`.
- **Query-time** tooling is a **separate stateless library** (`latence-retrieval`) with its own
  `latence.retrieval` provider seam.

## Index-time — extends the pipeline (emits files)

| Piece | Home | Emits |
| --- | --- | --- |
| Dense embedding | `Embedder` (**exists**, ADR-0045) | `embedding` column |
| SPLADE composition (ADR-0053) | new `SparseEmbedder` | `sparse_indices` + `sparse_values` |
| Multi-vector *(experimental, ADR-0050)* | new `MultiVectorEmbedder` | `multivectors.parquet` sidecar |
| MUVERA FDE *(experimental, ADR-0050)* | new `FdeConverter` | `fde_embedding` column (→ customer's **dense** index) |
| Graph features: centrality, community (ADR-0051) | **extends** Graph Assembly | `graph-nodes.parquet` cols + `context_header` |
| BM25 term-stats *(optional)* | sidecar exporter | `bm25` postings artifact |

## Query-time — the stateless library (`latence-retrieval`)

Three distinct candidate-processor operations, named precisely (not lumped as "fusion"):

- **fuse** — merge N ranked lists → one (RRF / weighted).
- **rerank** — re-score one list with a stronger model (cross-encoder; MaxSim is the experimental stub, ADR-0050).
- **pack** — select a token-budgeted subset maximising coverage − redundancy (the ported Knapsack QKP
  context-packer).
- **expand** — graph-augmented: 1-hop over denormalised columns (default), DuckDB multi-hop (ADR-0051).

Consumed at two levels: **Level 1** à-la-carte one-liners on candidates you already fetched; **Level 2**
an optional `retrieve(query, backend=…, pipeline=[…])` orchestrator around the customer's engine. The
**MCP server** (`latence-retrieval-mcp`, ADR-0052) is Level 2 with an MCP face — capability-adaptive,
hardened (ported from `memory_engine`), with all agent-facing copy loaded from operator config.

Dependency tiering mirrors ADR-0016: the core ships **pure-Python reference** implementations (RRF,
numpy MaxSim reference, networkx graph, basic BM25); heavy deps sit behind extras (`[splade]`,
`[multivector]`, `[fde]`, `[knapsack]`, `[graph-duckdb]`, `[mcp]`, `[backend-qdrant]`).

## In v1

- Dense (extend `Embedder`) + **SPLADE structured composition** (real) + **BM25** (query-time Cython
  rescorer over a candidate set + optional index-time term-stats artifact — **never** a first-stage
  index, ADR-0048).
- **fuse** (RRF/weighted) + **rerank** (cross-encoder real; MaxSim reference) + **full Knapsack packer**
  (ported, Apache).
- **Graph-augmented retrieval**: index-time centrality/community; 1-hop default; DuckDB-over-parquet
  zero-ops multi-hop; Neo4j optional; networkx reference.
- **MUVERA FDE + raw multi-vectors**: real, **experimental, off by default** (ADR-0050).
- **MCP server**: hardened, stack-agnostic, config-driven copy, **local/loopback + bearer**.
- Reference `RetrievalBackend` adapter: **Qdrant** first.

## Explicitly deferred (not v1)

- The **retrieval engine itself** — first-stage ANN/sparse/BM25 search always runs on the customer's
  store (ADR-0048). This tooling is upstream of it.
- The **LightOn-regularizer** multi-vector→dense converter and any bake-off that *blesses* a production
  FDE default; multi-vector on-by-default (ADR-0050).
- The **SOTA MaxSim kernels** (Triton/MLX/SIMD) as an accelerator — deferred with the multi-vector path;
  the unsloth-derived FP8 kernel needs its own license clearance (ADR-0049/0050).
- **Remote MCP transport** (OAuth 2.1 / mTLS) — v1 is local/loopback (ADR-0052).
- Additional backend adapters beyond Qdrant (Databricks, Azure) — seam-compatible follow-ons.

## Provenance

The mined assets are the maintainer's own projects, relicensed Apache-2.0 for this use: `memory_engine`
(the MCP hardening core) and `colsearch` (the Knapsack QKP packer and the MaxSim kernels). The only
third-party remnant is the FP8 MaxSim kernel adapted from unsloth, which rides the *deferred* kernel
path and carries its own license-clearance TODO. Local checkouts may still show stale CC-BY-NC headers;
ports are taken from the Apache-2.0 upstream state.
