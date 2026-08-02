# S0 investigation — retrieval config surface (reader C of 4)

The query-time toolbox is much larger than what M6's eval harnesses exercised. Campaign
facts for the S5 spec:

## Component registry (`latence.retrieval` entry points)

Processors: `processor.{identity, rerank, pack, expand, bm25, maxsim, multihop_expand}`;
fusers: `fuser.{rrf, weighted}`; query encoders: `query_sparse.splade`,
`query_multivector.hashing`. NOT registered (constructed directly): QdrantBackend,
GraphRetriever, CompositeBackend, RetrievalServer, hypergraph_signals, coe.select,
assemble_chains, tuning.

## The full-strength S5 arsenal (all shipped, most unused by M6)

- **Orchestrator**: QueryBundle → per-modality fetch (bm25/dense/graph/multivector/sparse)
  → RRF or weighted fusion (`fuse_weights` per mode) → processor chain (e.g. expand →
  rerank → pack) → final_top_k. `OrchestrationPipeline.from_config` = the YAML-able recipe.
- **Fusion tuning module** (`tuning.py`): evaluate_fusion / sweep_fusion_weights /
  compare_fusers — offline, deterministic; tuned weights transfer verbatim.
- **Cross-encoder rerank** (`processor.rerank`, [rerank] extra): model REQUIRED no default.
- **SPLADE leg**: index via `latence retrieval splade-index` (MANDATORY
  `--accept-noncommercial-license` — naver/splade-v3 is CC-BY-NC-SA; relevant to the
  campaign's commercial posture!) + `query_sparse.splade` (byte-identical composition).
- **Graph legs**: `processor.expand` (1-hop, metadata columns, no DB) and
  `processor.multihop_expand` (real traversal; duckdb/networkx/neo4j sources over
  graph-edges.parquet) and `GraphRetriever` (graph as a FIRST-STAGE peer via
  CompositeBackend — closes "post-fusion expanders can't add missed candidates").
- **MaxSim/multivector** (ADR-0050, off default) + MUVERA FDE (`fde.muvera`, training-free).
- **Qdrant pair**: `latence-sink-qdrant load/verify-absent` (idempotent UUIDv5 ids,
  reconcile sweep, purge audit) ↔ `QdrantBackend` (dense/sparse/filter; text_key
  contract aligns with the sink zero-config).
- **MCP server**: search/more_like_this/get_context/list_sources/index_status; bearer auth,
  rate limit, loopback-only transport, injection-hardened copy; stdio via `run_stdio`
  (NO console script yet — API-only; docs name `latence-retrieval-mcp` as design label).

## Hypergraph/CoE facts (S5 spec constraints)

- **Tier = which ARGUMENTS you pass** to `hypergraph_signals` (no tier config key):
  tier0 = `ann_node_ids` (+ optional `eu_sims`) from the CALLER's ANN; tier1 =
  `alias_surfaces`; both mergeable (alias wins at 1.0, ANN at alpha=0.2).
- **Tier 2 is NOT shipped in the package** — GLiNER2 query NER stayed harness-side
  (ADR-0059 specifies it as a named opt-in extra; a productization item for this campaign).
- HypergraphConfig defaults: hops 2, hub_cap 2000, expand_cap 3000, max_candidates 20000,
  k_h 200, beta 0.5 (forced 1.0 without eu_sims), alpha 0.2, restart 0.2, steps 6,
  theta_v 0.1. CoeConfig: 6 v0 weights + w_conn 0.15, theta_ratio 0.15, weak_discount 0.6,
  verify_on_match false; verification triple = all-three-or-none, loud otherwise.
- trace_sink = plain callable, ids/counts/scores only (never tau_text) — the campaign can
  collect selection traces corpus-wide for the future learned utility model.

## Engine input spec (docs/RETRIEVAL-ENGINE-INPUT-SPEC.md, v20)

export/ tree: records.jsonl+parquet (6 stable columns; embedding/sparse/fde/context
columns only when wired), graph-{nodes,edges}.parquet + ttl + graphml,
graph-hyperedges.{parquet,kv}, multivectors.parquet, bm25-{stats.json,postings.parquet},
splade/index.parquet. `context_header.kg_node_ids` = the chunk→graph join;
`neighbor_node_ids` = 1-hop expansion keys.

## Wizard schema notes

1. Two disjoint config namespaces: index-time `latence.providers` sub-provider slots vs
   query-time `latence.retrieval` {provider, config} blocks.
2. REQUIRED-no-default (must be prompted): Packer.budget, Reranker.model,
   SpladeQueryEncoder.model, TiktokenCounter.encoding, MuveraFdeConverter.input_dimension,
   QdrantBackend.{collection,capabilities,dimension}, QdrantSink.collection.
3. Closed key sets reject typos loudly across the board (good: wizard can validate hard).
4. Injection-point seams to surface as "advanced": TokenCounter, CrossEncoderModel,
   GraphSource, Propagator, UtilityModel, query_signal_gen, trace_sink.
5. Embedders: `embedding.sentence_transformers` (default granite-311m 768d Apache-2.0,
   bf16+FA2 GPU), `embedding.endpoint` (OpenAI-compatible, vLLM-friendly),
   `embedding.hashing` (offline floor). **This closes CLOSEOUT-AUDIT F-VEC/F-BASE**: the
   dense leg for S5 exists — it was just never wired into an eval.
