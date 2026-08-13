# Wave 1-A — file-identity-keyed read index for the graph sources (+ cache lifetime policy)

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 1-A). Closes FINDINGS F1.
Pattern source: the ADR-0056 amendment (`HypergraphIndex` / `_INDEX_CACHE`, ticket 01), now
generalized with an explicit lifetime policy. All oracles green; byte-identity proven by running
the gates, not by assertion.

## What changed

- **New module** `packages/latence-retrieval/src/latence_retrieval/read_cache.py` —
  `FileIdentityCache`: process-shared memoization of derived read structures, keyed by
  `(st_dev, st_ino, st_size, st_mtime_ns)` (+ a `variant` sub-key for parameterised decodes).
  Lifetime policy: identity invalidation, **superseded-signature eviction** (decoding a new
  version of a file drops every cached decode of its other versions — a serving process that
  outlives export refreshes holds at most one version per store), **bounded LRU** (default 8
  entries per cache). This is the policy `hypergraph._INDEX_CACHE` lacked (+347 MB wiki /
  +771 MB uda per store version per process, unbounded).
- `hypergraph._INDEX_CACHE` migrated onto it; `clear_index_cache()` and every existing
  cache-sharing / rewrite-invalidation package test unchanged and green.
- `cache: bool = True` on the shipped parquet adapters, mirroring `HyperedgeKV.open`;
  `cache=False` preserves each original per-call path **verbatim** (the parity oracle):
  - `DuckDBGraphSource` — cached decoded undirected edge **adjacency**; per call runs only the
    canonical `_traverse` BFS (split into `_build_adjacency` + `_traverse_adjacency`, same
    statements — caching the raw edge list alone left an O(|E|) adjacency rebuild per call and
    was *slower* than the CTE on wiki2multihop: 70.8 ms vs 51.2 ms median; the adjacency cache
    is what the deepening specified, and it wins decisively).
  - `NetworkxGraphSource` — cached loaded graph; identical per-seed bounded shortest-path reads.
  - `DuckDBGraphDocumentSource` — cached `_DocumentTable` (ascending-`id` rows + node→row
    postings). HAZARD preserved: rows are stored in the uncached query's own `ORDER BY id`
    (same engine, same column), so postings unions re-sorted by row position reproduce the
    ascending-id contract the ticket-01 tiered early-stop depends on, exactly.
  - `build_graph_source` / `build_graph_document_source` grow the `cache` config key
    (bool-validated; `Neo4jGraphSource` unaffected — a live server, nothing to memoize).
- ADR-0056: second dated amendment (2026-08-13) naming the generalized pattern + lifetime
  policy. FINDINGS F1 closed in place. Apache seven-file pin untouched
  (`multihop.py`/`graph_retrieval.py` keep their exact headers; `read_cache.py` is
  PolyForm-Noncommercial like the package — the pin test enumerates exactly seven, green).

## Byte-identity gate (ticket-01 parity.py, extended)

`parity.py` gained two lanes plus `--only=`: `kg_shipped` (shipped DuckDB traversal + DuckDB
document resolution, `cache=True` vs `cache=False`, full ranked `(id, score)` lists `==`) and
`kg_shipped_nx` (shipped networkx traversal A/B, document source held fixed). Runs on the real
local S3 exports:

| dataset | lane | queries | diffs |
|---|---|---|---|
| multihop_rag | kg_shipped | 2,556 (full set) | **0** |
| multihop_rag | kg_shipped_nx | 2,556 (full set) | **0** |
| wiki2multihop | kg_shipped | 12,576 (full set) | **0** |
| wiki2multihop | kg_shipped_nx | 1,000 (deterministic strided sample; the uncached side re-reads + rebuilds a 90k-edge graph per query, ~270 ms/q) | **0** |

Result JSONs: `tickets/01-latency/results/parity-*-kg_shipped*.json`. The committed full-set
baseline `parity-*.json` files are untouched. `benchmark/serving/parity.py` still runs (arg
surface + import path verified; a full run needs live Qdrant and is Wave 5's scope — its bench
shadow-source injection was deliberately NOT rewired here).

## Latency effect on the shipped adapters (adapter_timing.py, per-query median, real exports)

`GraphRetriever.search` end-to-end, seeds precomputed outside the timed region; cached lane 200
queries, uncached 40–60 (results in `tickets/01-latency/results/adapter-timing-*.json`):

| dataset | engine | uncached median (before) | cached median (after) | p95 after | one-time decode |
|---|---|---|---|---|---|
| wiki2multihop | duckdb | 60.0 ms | **0.34 ms** (~175×) | 0.78 ms | ~1.0 s |
| wiki2multihop | networkx | 270.6 ms | **0.34 ms** (~800×) | 0.71 ms | ~1.0 s |
| multihop_rag | duckdb | 34.9 ms | **0.27 ms** (~130×) | 0.41 ms | ~0.09 s |
| multihop_rag | networkx | 26.7 ms | **0.29 ms** (~90×) | 0.42 ms | ~0.05 s |

## Package gates

- `PYTHONHASHSEED=0 .venv/bin/python -m pytest packages/latence-retrieval/tests` —
  **764 passed, 4 skipped**. The count includes the new `test_read_cache.py` + adapter-cache
  tests, and the formerly-skipped real-DuckDB equivalence tests now run because `duckdb` was
  installed into the local venv to drive the real-engine gates (all green on first run) — it is
  NOT added to the dev dependency group; ADR-0016's CPU-first env is unchanged, everything
  degrades to skip without it.
- `ruff check packages/latence-retrieval` clean; `mypy --strict packages/latence-retrieval/src`
  clean (0 errors, 30 files). License pin `test_publish_readiness.py` 25 passed;
  latence-core KV interop tests 31 passed; `benchmark/s5/tests` 125 passed.

## Residuals (logged, not chased)

- `benchmark/s5/legs.py`'s docstring rationale for `EdgeGraphSource` ("the shipped
  NetworkxGraphSource re-reads and rebuilds the edge graph on every call") is now outdated —
  the bench shadow sources remain byte-equivalent and in use; rewiring the bench onto the
  shipped cached adapters is Wave 5 territory.
- The architecture-pass janitorial rider "ADR index cross-ref for the 0049/0056 solver-lineage
  confusion" targets the ADR index file, which this wave does not touch; left for the wave that
  does.
