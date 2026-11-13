# The hypergraph Evidence sidecar is read per call over a point-lookup KV; no global edge-adjacency matrix is ever built

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

The Evidence-Unit (EU) hypergraph store that backs chain-of-evidence retrieval is emitted by the
pipeline as **files** (ADR-0017, ADR-0049) and read by the query-time tooling **per call, holding
nothing between calls** (ADR-0048) — the same discipline `DuckDBGraphSource` already follows
(ADR-0051). The online read path is a **point-lookup KV**, not a columnar scan. The spec's global
edge-adjacency matrix `A_E` (and the global incidence CSR) is **not built at all**: edge overlap is
recomputed on the query-local candidate set. This ADR records the measured evidence, because the
decision was made on numbers rather than on principle.

## Context

The hypergraph store (entities, hyperedges/EUs, incidence, per-entity postings, EU text) is the
structural index the chain-of-evidence selector needs. Its source spec assumes the online path
slices a **global** incidence CSR and does `O(1)` lookups against a **global** `A_E = BᵀB`,
describing the latter as "the single most important precomputation for solver speed."

That assumption collides with ADR-0048, which holds the query-time tooling stateless and explicitly
**rejected** an embedded/ephemeral index escape hatch as "the exact crack through which 'just a
little state' becomes a system of record." Holding those globals resident is that crack; loading
them per query looked prohibitively slow. The tension could not be resolved by argument, so it was
measured.

## Evidence

A synthetic-corpus simulation (`issues/wayfinder-hypergraph-coe/prototypes/residency/`, Zipf(1.07)
entity popularity, arity 2–12, 100 queries per scale, |V0|=10, |E_c|≤2000, budget 8000 tokens)
compared three strategies at 10k / 100k / 1M EUs: **A** strict per-call over parquet, **A′** strict
per-call over an LMDB point-lookup KV, **C** fully resident.

1. **Global `A_E` is infeasible.** `nnz(A_E) ≈ Σ_v d_v²`, so hub entities make it grow
   quadratically in hub degree: 38M nnz at 10k EUs, **2.9B (32 GiB) at 100k**, **232B (2.6 TiB) at
   1M**. It dies between 10k and 100k EUs.
2. **Global `A_E` is also worthless where it fits.** At 10k EUs, slicing the stored matrix cost
   **2.08 ms** against **1.06 ms** to recompute the candidate-local overlap from scratch. The
   precomputation was slower than its absence.
3. **Residency does not change what is selected.** 100/100 identical selections, mean Jaccard
   1.0000, at every scale, for both strict strategies against resident. The choice is purely
   latency/ops; quality cannot arbitrate it.
4. **The latency cost was columnar scan, not statelessness.** p50 at 1M EUs: strict-parquet
   **69.5 ms** (63 ms of it I/O — 2000 scattered `edge_id`s touch most row groups, and it degrades
   with corpus size), strict-KV **2.26 ms** (0.51 ms of it I/O, flat in corpus size), resident
   **1.70 ms**. Holding the whole store in memory buys **0.56 ms**.

## Decision

- **The sidecar is a pipeline artifact — files.** Emitted during `latence process` as an index-time
  signal (ADR-0049), governed by ADR-0017 like every other deliverable. Not a service.
- **The online path holds nothing between calls.** Each query opens the store, reads what it needs,
  and closes it — statelessness stays *structural*, as in `DuckDBGraphSource`. **ADR-0048 is not
  amended**; the escape hatch it rejected stays shut.
- **The online read path is a point-lookup KV**, not a columnar scan. Columnar scan is ruled out by
  measurement, not taste: it is 30× slower and degrades with corpus size.
- **No global `A_E` and no global incidence CSR are built or stored.** `edge_adjacency.npz` and
  `incidence.npz` leave the layout. Edge overlap is recomputed on the candidate set (|E_c| ≤ 2000)
  per query; the query-local incidence `B_q` is built from the fetched entity lists.
- **Per-entity postings carry γ inline**, so top-k candidate truncation needs no row fetch.
- **The vector engine holds exactly two ID-payload collections** (`entity_vecs`, `eu_vecs`) and
  serves ANN entry only. Swapping engines never touches sidecar or solver code.

Resulting sidecar contract (supersedes the source spec's §1.6/§1.7 layout):

```
entities.parquet     entity_id, name, type, aliases, description, degree
hyperedges.parquet   edge_id, entity_ids[], relations[], gamma, token_cost,
                     support, has_relations, provenance
eu.kv                edge_id  -> {entity_ids, token_cost, gamma, support, has_relations}
                     entity_id -> {edge_ids[], gammas[]}     (postings, γ-sorted, γ inline)
                     edge_id  -> tau_text
```

Parquet remains the durable, inspectable, engine-agnostic form (and the Export/audit surface); the
KV is a derived read-path artifact rebuildable from it.

## Considered alternatives

- **Fully resident store (with or without global `A_E`)** — rejected: buys 0.56 ms at 1M EUs, costs
  an ADR-0048 amendment and a memory footprint that grows with the corpus.
- **Caller-owned handle** (adopter holds the loaded store; our processors stay pure functions of
  it) — moot once resident loses on its own merits; it was only ever a way to relocate the state,
  not to justify it.
- **Strict per-call over columnar parquet scan** (the literal reading of the spec's constraint) —
  rejected on measurement: 69.5 ms p50 at 1M EUs and worsening with scale.
- **Building `A_E` anyway for small corpora** — rejected: a size-conditional code path whose fast
  branch is measurably slower than the general one.

## Consequences

- The online path is **flat in corpus size** (~2.3 ms p50 from 10k to 1M EUs): compute runs on the
  query-local subgraph only, and the KV reads do not grow.
- The spec's "the index is built for the solver" intent survives for everything per-EU and
  per-entity (`token_cost`, `gamma`, `support`, `has_relations`, γ-sorted postings) — those are
  columns and point lookups. Only the *global pairwise* structure is refused.
- Two artifacts instead of one (parquet + derived KV) mean the delta/Retraction/Purge machinery
  (ADR-0029) must cover both; the KV is rebuildable, so Purge may rebuild rather than edit in place.
  Detail belongs to the deltas ticket, not this ADR.
- **Concurrency: measured after the fact and fine as-is** (ticket 19, 2026-07-31, real
  `HyperedgeStoreReader` over real records at 50k/500k hyperedges): per-call open+mmap costs
  0.039–0.062 ms p50 — ~1–2% of a query — flat from 1 to 16 concurrent processes and across a
  10× corpus-size step; total p50 ≈ 2–2.9 ms at c≤4. Degradation at c=16 lands in the reads
  (CPU saturation on a 10-logical-CPU host), so the contingency fix named here (a cheaper open)
  would save nothing and stays unbuilt. Cold-cache remains unmeasured (needs sudo on the dev
  host) — stated, not hidden.
- Absolute latencies come from a Python prototype on synthetic data and are not product SLAs. The
  *comparisons* hold — all strategies shared the same compute code — but the numbers will move.

## Amendment (2026-08-12, wayfinder-enterprise-serving ticket 01): a process-shared, derived read index

The "resident buys 0.56 ms" ruling above was measured over an **LMDB** point-lookup KV (C-speed
lookups) with candidate sets of |E_c| ≤ 2000. The shipped query path is the **pure-Python**
`latence_retrieval.hypergraph.HyperedgeKV` reader, and the verified expansion (ticket 20:
hop-priority, hub-capped, `max_candidates=20000`) touches far more rows than the prototype did.
Measured on real exports (cProfile, Mac dev host): at 677k hyperedges (uda) one query performed
~4.7M binary-search key comparisons and ~230k JSON decodes — **3.4 s median** for the ladder's
hypergraph lane, seconds not milliseconds. The regime the original numbers were taken in does not
exist in the shipped code, so the conclusion drawn from them no longer binds.

Decision: `HyperedgeKV.open` now attaches a **process-shared `HypergraphIndex`** — the
entity→hyperedge postings and hyperedge→entity incidence (+ γ and its truncation rank) decoded
once per store FILE per process, keyed by file identity (device/inode/size/mtime) and therefore
invalidated by any rewrite. This does **not** reopen ADR-0048's escape hatch:

- The index is a **derived, rebuildable memoization** of the emitted artifact, never a system of
  record — nothing can be written to it, and byte-identical results with and without it are pinned
  by tests and by the ticket-01 full-query-set parity runs on real exports (0 diffs).
- Statelessness stays behavioral: identical inputs yield identical outputs; the store file's
  lifetime is still the caller's per-call `open`/`close`.
- `open(path, cache=False)` and `from_bytes` keep the original scan-per-call path (preserved
  verbatim), which remains the parity oracle.
- The global pairwise structure stays refused: no `A_E`, no global incidence CSR — the index holds
  exactly the postings and incidence the store already contains, in int-mapped form.

## Amendment (2026-08-13, architecture pass Wave 1-A): the read index is a generalized pattern with a lifetime policy

The 2026-08-12 amendment's shape — a **derived, rebuildable memoization of an emitted artifact,
keyed by file identity (`st_dev`/`st_ino`/`st_size`/`st_mtime_ns`), with `cache=False` preserving
the original per-call path verbatim as the parity oracle** — turned out not to be specific to the
hyperedge store. The shipped parquet graph adapters had exactly the friction the index fixed here:
`NetworkxGraphSource.neighbors` re-read the parquet and rebuilt the whole graph per call,
`DuckDBGraphSource.neighbors` re-ran the recursive CTE over the parquet per call, and
`DuckDBGraphDocumentSource.documents` re-scanned the full chunk parquet per call — which the
ticket-01 tiered `GraphRetriever.search` now invokes up to `max_hops + 1` times per query
(FINDINGS F1: the tiering win was measured against run-lifetime bench sources; on the shipped
adapters the per-call re-read multiplied instead).

Decision: the pattern gets **one home** — `latence_retrieval.read_cache.FileIdentityCache` — and
an explicit **lifetime policy** the first instance lacked (`_INDEX_CACHE` grew by one full index
per store version per process, ~347 MB (wiki) / ~771 MB (uda), without bound across export
refreshes):

1. *Identity invalidation* — a rewrite changes the signature; stale content is never served.
2. *Superseded-signature eviction* — decoding a NEW version of a file drops every cached decode of
   its other versions (same `(st_dev, st_ino)`, any variant), so refresh churn never accumulates.
3. *Bounded LRU* — a hard per-cache entry cap (default 8) as the safety valve for many distinct
   live artifacts.

Consumers, each with `cache: bool = True` and the untouched verbatim path under `cache=False`:
`HyperedgeKV.open` (the `HypergraphIndex`, migrated onto the shared cache), `DuckDBGraphSource`
(decoded undirected edge adjacency + the canonical `_traverse` BFS), `NetworkxGraphSource` (the
loaded graph; identical per-seed bounded shortest-path reads), `DuckDBGraphDocumentSource` (the
decoded ascending-`id` row table + node→row postings — reproducing the `ORDER BY id` contract the
tiered early-stop depends on). Decode parameters (configurable column names) enter the key as a
`variant`, so two decodings of one file never collide.

The boundary is unchanged: nothing can be written through the cache, results are byte-identical
with and without it (package equivalence tests + the ticket-01 `kg_shipped`/`kg_shipped_nx`
parity lanes over real exports, 0 diffs), ADR-0048's escape hatch stays shut, and the global
pairwise structure stays refused. Measured effect (adapter_timing.py, per-query median,
`GraphRetriever.search` end-to-end): wiki2multihop DuckDB 60.0 ms → 0.34 ms, networkx 270.6 ms →
0.34 ms; multihop_rag DuckDB 34.9 ms → 0.27 ms, networkx 26.7 ms → 0.29 ms; one-time decode ~1.0 s
(wiki) / ~0.1 s (multihop_rag) on the first call per file version.
