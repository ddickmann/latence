# Retrieval-quality tooling — tickets

Tracer-bullet vertical slices generated (mattpocock `to-tickets`) from the spec
[#158](https://github.com/ddickmann/latence-framework/issues/158) and ADRs 0048–0053.
All are GitHub issues labelled `ready-for-agent`; work the frontier (any ticket whose
blockers are all closed). Files here mirror the issues (path-stable copies).

| # | Issue | Ticket | Blocked by |
|---|---|---|---|
| 01 | #159 | query-time seam + `latence-retrieval` package | — |
| 02 | #160 | index-time signal-generation seam + additive export | — |
| 03 | #161 | graph-feature enrichment (centrality + community) | — |
| 04 | #162 | SPLADE structured-composition SparseEmbedder | #160 |
| 05 | #163 | MUVERA-FDE + MultiVectorEmbedder (experimental) | #160 |
| 06 | #164 | BM25 term-statistics artifact exporter | #160 |
| 07 | #165 | fuse (RRF + weighted) | #159 |
| 08 | #166 | rerank (cross-encoder) | #159 |
| 09 | #167 | pack (Knapsack QKP context-packer) | #159 |
| 10 | #168 | expand (1-hop graph) | #159, #161 |
| 11 | #169 | expand multi-hop (DuckDB over graph parquet) | #159, #161, #168 |
| 12 | #170 | query-side sparse + Cython BM25 rescorer | #159, #162 |
| 13 | #171 | rerank MaxSim reference (experimental) | #159, #163 |
| 14 | #172 | Level-2 orchestrator `retrieve()` | #159, #165, #166, #168 |
| 15 | #173 | reference Qdrant `RetrievalBackend` adapter | #159 |
| 16 | #174 | MCP server | #159, #172 |

**Frontier at kickoff:** #159, #160, #161 (no blockers) can start immediately.
