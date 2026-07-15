# Zero-ops multi-hop graph retrieval via DuckDB over the emitted graph parquet; Neo4j demoted to an optional enterprise adapter

Graph-augmented retrieval defaults to **1-hop over the KG columns already denormalised onto each chunk**
(no extra infrastructure, any vector store). For **multi-hop**, the zero-ops default is **DuckDB running
recursive traversal over the graph parquet the pipeline already emits** — an embedded, serverless engine
querying files that sit next to the vector index, with nothing for the user to run or manage. Neo4j is
demoted from "the multi-hop backend" to an *optional* adapter for shops that already operate it.

## Context

Under the stateless boundary (ADR-0048) the tooling holds no graph service. The pipeline already
denormalises the 1-hop KG neighbourhood onto every chunk (`context_header.kg_node_ids`,
`neighbor_entity_ids`, `neighbor_triples`; ADR-0039), so 1-hop expansion and graph-aware rescoring are
pure metadata operations on the customer's existing index — no graph DB at all. Only *multi-hop*
traversal needs the graph queryable. The requirement was explicit: near-zero operational burden —
"the user does not have to manage a graph database" — because few teams run one well.

"Use Neo4j" and "no graph-DB ops" are in direct tension: **Neo4j is a server** (run it, secure it,
upgrade it — that *is* graph-DB ops). The zero-pain answer is an embedded, serverless engine that ships
as a *file* beside the vector index.

## Decision

- **Index-time:** extend Graph Assembly to compute **centrality** (degree/PageRank) and **community**
  (Louvain/Leiden) over the assembled KG, written as `graph-nodes.parquet` columns and projected onto
  each chunk. These serve **two** consumers at once — graph-augmented retrieval *and* the Knapsack
  packer's `centrality` / `cluster_ids` inputs (ADR-0049).
- **Default tooling path = 1-hop**, stateless, over the denormalised columns — works on any vector store.
- **Zero-ops multi-hop = DuckDB over the emitted `graph-nodes.parquet` / `graph-edges.parquet`**
  (embedded/in-process, MIT-licensed, actively maintained) via recursive-CTE traversal (with DuckDB's
  `USING KEY` for fast BFS/shortest-path), and the optional DuckPGQ extension for full SQL/PGQ path
  queries. Multi-hop needs **no new artifact and no server** — it queries the same graph files the
  pipeline already produces, read-only per call (still stateless).
- **Reference fallback = networkx** (pure-Python, tiny graphs, no binary dep).
- **Enterprise adapter = Neo4j `GraphBackend`** — optional, for shops that already run Neo4j; loads from
  the `graph.ttl` / `graph.graphml` the pipeline already emits.

## Considered alternatives

- **Kùzu** (the "SQLite of graph databases": embedded, MIT, Cypher) was the first choice and is a near-
  perfect technical fit — **rejected because the upstream project was archived after an Apple acquisition
  (Oct 2025)**; only community forks continue. Building a fresh OSS launch on an archived core is a
  supply-chain risk we decline.
- **Neo4j as the default** — rejected: it is a managed server and contradicts the zero-ops requirement.

## Consequences

- Multi-hop "just works" against files the customer already has next to their index — the promised
  almost-zero-pain path — with DuckDB (already common in lakehouse-adjacent stacks) as the only added,
  well-maintained dependency.
- Index-time graph-feature computation widens `graph-nodes.parquet` and couples community-detection into
  `latence process`; the lean alternative (1-hop only, depth solely via the optional Neo4j adapter) is
  available by leaving the feature toggle off.
