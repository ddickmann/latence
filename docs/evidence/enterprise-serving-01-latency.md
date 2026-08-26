# Ticket 01 report — hypergraph + graph lane latency at SOTA

Status: DONE on the Mac-local gate — measurements and parity complete on all five local
exports; uda-at-full-sweep confirmation deferred to the post-sweep S5 re-run (map ruling 1 —
the pod was not touched).

## What shipped (packages/latence-retrieval — never benchmark code)

1. **`hypergraph.py` — `HypergraphIndex`** (ticket mitigation 1, "precomputed hyperedge
   indices"): the entity→hyperedge postings and hyperedge→entity incidence (+ γ and its global
   `(-γ, id)` truncation rank) decoded **once per store file per process** and shared through a
   file-identity-keyed cache (device/inode/size/mtime — a rewritten store re-indexes, never
   serves stale). `HyperedgeKV.open` attaches it transparently; `open(path, cache=False)` and
   `from_bytes` keep the original scan-per-call path, preserved **verbatim** as the parity
   oracle. Expansion + s_hyp prune run in int space (int order == id string order because the
   KV's keys are byte-sorted and UTF-8 byte order equals code-point order, so every tie-break
   reproduces the string-keyed original exactly; every float expression is replicated
   operation-for-operation). Kernels are numpy-vectorised when numpy is importable — including
   an O(n) first-occurrence layer dedupe (scatter/gather, no argsort) — with pure-Python
   fallbacks, so the package core stays pydantic-only. Retained-set solver fields are read by
   direct KV row addressing (the `e|` rows are one contiguous range), not per-id binary search.
2. **`graph_retrieval.py` — tiered `GraphRetriever.search`**: the reached neighbourhood is
   resolved to documents tier by tier (closest hop first) and resolution stops as soon as
   `top_k` hits are ranked. Valid exactly when hop scores strictly decrease (guarded;
   `hop_decay == 1.0` / underflow keep the pre-ticket exhaustive path, preserved verbatim).
   A contract-honouring `GraphDocumentSource` returns each document at its closest tier and
   tiers arrive id-ascending, so the early stop is byte-identical to ranking everything. At uda
   this is the difference between resolving ~10² and ~10⁵ documents (≈93k pydantic candidates
   per query built pre-ticket, then truncated to 100).

ADR-0056 carries a dated amendment: the original "resident buys 0.56 ms" rejection was measured
over an LMDB C-speed KV at |E_c| ≤ 2000; the shipped pure-Python reader at verified expansion
widths (`max_candidates=20000`) measures seconds, so the cached **derived, rebuildable** index is
recorded there, with ADR-0048's escape hatch explicitly kept shut (nothing writable, results
byte-identical, opt-out preserved).

## The measured cost structure (cProfile, before)

- Hypergraph lane at uda (677,257 hyperedges): ~4.7M pure-Python binary-search key comparisons
  (each materialising key bytes) + ~230k `json.loads` **per query**, in
  `solver_fields`/`postings` during expansion — ≥90% of the lane.
- Traversal lane at uda: ~93k documents resolved and built into pydantic `Candidate`s per query
  before top-100 truncation (~1.9M pydantic constructions per 10 queries).
- After the index landed, the residual hot spot was `np.unique`'s internal argsort on ~230k-row
  layer streams (64% of the remaining lane) — replaced by the O(n) scatter/gather dedupe.

## BEFORE / AFTER latency (Mac dev host, 50 evenly-strided gold queries per dataset, per-lane wall time)

Lanes, exactly as the S5 harness pays them (`benchmark.s5.legs` / `rescue` seams; per-query
store open included): `kg` = alias seeding + `GraphRetriever` traversal (top-100);
`hyper_ladder` = `hypergraph_signals(alias_surfaces=query_ngrams)` + chunk resolution + CoE
selection; `hyper_rescue` = the rescue entry (alias + tier-0 `ann_node_ids` from query aliases
plus top-BM25-document entities, seeds precomputed outside the timed region and shared
byte-identically by every path).

| dataset | lane | before median | before p95 | after median | after p95 | median speedup | p95 speedup |
|---|---|---|---|---|---|---|---|
| multihop_rag | kg | 1.5 ms | 23.2 ms | 0.78 ms | 1.66 ms | **2x** | 14x |
| multihop_rag | hyper_ladder | 43.1 ms | 103.8 ms | 7.90 ms | 14.33 ms | **5x** | 7x |
| multihop_rag | hyper_rescue | 85.3 ms | 120.5 ms | 6.50 ms | 14.19 ms | **13x** | 8x |
| vidoseek | kg | 0.2 ms | 0.4 ms | 0.19 ms | 0.45 ms | 1x | 1x |
| vidoseek | hyper_ladder | 5.8 ms | 13.8 ms | 3.55 ms | 9.19 ms | **2x** | 1x |
| vidoseek | hyper_rescue | 10.6 ms | 17.8 ms | 3.89 ms | 9.03 ms | **3x** | 2x |
| ohr_bench | kg | 0.3 ms | 5.3 ms | 0.34 ms | 1.21 ms | 1x | 4x |
| ohr_bench | hyper_ladder | 20.9 ms | 90.0 ms | 5.65 ms | 12.39 ms | **4x** | 7x |
| ohr_bench | hyper_rescue | 50.1 ms | 140.3 ms | 6.23 ms | 11.41 ms | **8x** | 12x |
| wiki2multihop | kg | 7.9 ms | 116.4 ms | 0.81 ms | 3.24 ms | **10x** | 36x |
| wiki2multihop | hyper_ladder | 489.8 ms | 826.4 ms | 14.90 ms | 22.03 ms | **33x** | 38x |
| wiki2multihop | hyper_rescue | 959.5 ms | 1095.8 ms | 19.48 ms | 24.54 ms | **49x** | 45x |
| uda | kg | 435.3 ms | 1777.9 ms | 3.92 ms | 30.11 ms | **111x** | 59x |
| uda | hyper_ladder | 3570.5 ms | 4476.1 ms | 37.13 ms | 43.78 ms | **96x** | 102x |
| uda | hyper_rescue | 3732.9 ms | 4576.4 ms | 33.03 ms | 39.68 ms | **113x** | 115x |

**"Orders of magnitude" is claimed exactly where the numbers show it**: at uda scale — the
677,257-hyperedge regime the ticket was opened for — every graph lane is **~100× (96–113×
median, 59–115× p95)**. The speedup grows with corpus size because the AFTER cost is nearly
flat in scale (hyper ladder: 7.9 ms at 138k → 37.1 ms at 677k hyperedges, vs 490 ms → 3,570 ms
before); the small corpora were never seconds-per-query and honestly gain 2–13×.

One-time index build, at first open per process (transparent-at-load per the ticket), excluded
from per-query numbers and amortised over any real run: multihop_rag 0.2 s, vidoseek 0.2 s,
ohr_bench 0.9 s, wiki2multihop 2.8 s, uda 10.5 s (quiet machine). Resident index footprint
(measured): wiki2multihop +347 MB (137,986 hyperedges / 271,261 nodes / 542,077 postings),
uda +771 MB (677,257 / 516,509 / 1,955,311) — inside the sweep's 29 GB / 4-worker budget.

## Parity proof (the gate)

Method: for every usable gold query (full `benchmark.s5.goldmap`/`datasets` sets), each lane is
executed through every code path and compared with `==` — the ranked chunk `(id, score)` lists,
the CoE selection, AND the complete `HypergraphSignals` (ρ ids/order/float scores, solver
fields, anchors, vertex confidences, stage counters). Paths: per-call scan (pre-ticket code,
verbatim) vs shared accelerated index vs pure-Python index kernels; tiered `GraphRetriever` vs
the preserved exhaustive resolution. Runner: `tickets/01-latency/parity.py`; raw results in
`tickets/01-latency/results/parity-*.json`.

| dataset | queries parity-checked | coverage | diffs |
|---|---|---|---|
| vidoseek | 1,142 | full usable gold set | **0** |
| multihop_rag | 2,556 | full usable gold set | **0** |
| ohr_bench | 4,541 | full usable gold set | **0** |
| wiki2multihop | 12,576 | **full usable gold set** (4 shards of 3,144; per shard 1,100 checked before an infra interruption — the committed `*.interrupted.log`s are the zero-diff evidence — and the remaining 2,044 by the resumed run's result JSON; 1,100 + 2,044 = 3,144, 4 × 3,144 = 12,576) | **0** |
| uda | 400 | evenly-strided sample of the 8,583 usable queries (200 pre-interruption per the committed log + 200 resumed) — full-uda confirmation rides the post-sweep S5 re-run | **0** |

**Total: 21,215 real gold queries, 3 lanes each, every code-path comparison equal — 0 diffs.**
Each hypergraph lane was compared across three paths per query (scan vs accelerated index vs
pure-Python kernels — two equality checks over the full signals + ranked lists + CoE
selection), the kg lane across two (tiered vs preserved exhaustive, ids and scores).

Additionally pinned in the package suites (house style): 3-way byte-equality across entry
tiers/eu_sims/hops=3/tight caps/γ ties/unknown anchors on synthetic stores; cache sharing +
rewrite invalidation; tiered-vs-oracle equality at five truncation depths with an early-stop
call-count pin, filter-before-truncation, pure-expansion, and the `hop_decay=1.0` degenerate.

## Config surface

- `HyperedgeKV.open(path, cache=True)` — default ON (map ruling: shipped default = strong
  configuration). `cache=False` restores the pre-ticket scan-per-call path bit-for-bit.
- `HypergraphIndex(store, accelerated=None)` — auto-selects numpy kernels when importable;
  `accelerated=False` forces the pure-Python kernels (identical results, pinned).
- `clear_index_cache()` — drops the process-shared indices (tests / long-lived processes).
- `GraphRetriever` — no new knobs: tiering engages automatically iff hop scores strictly
  decrease; degenerate configs keep the exhaustive path. `numpy` is an optional accelerator,
  never a dependency: the package core remains pydantic-only.

## Mitigations from the ticket, disposition

1. Precomputed entity→hyperedge / hyperedge→chunk indices — **shipped** (`HypergraphIndex`;
   the hyperedge→chunk map was already a preloaded dict in the harness seam).
2. Community-scoped traversal — **not shipped**: scoping changes results, so it could only ever
   be opt-in; the parity-preserving optimizations above already deliver the latency target, so
   no non-default quality-affecting knob was added.
3. Policy gate — landed pre-ticket; orthogonal to and not counted in these numbers (they
   measure the lane itself when it runs).
4. Profile-revealed extras — **shipped**: γ-rank precomputation, O(n) layer dedupe, direct KV
   row addressing for solver fields, tiered document resolution.

## Deferred / residuals

- **uda at full sweep scale**: the pod's running S5 rescue sweep was not touched. The local uda
  export copy (677,257 hyperedges — verified equal scale) provided the at-scale BEFORE/AFTER
  above; the sweep re-run AFTER this ticket lands provides the full-pipeline confirmation
  number against the 5.24 s/query sweep measurement, per the map's operating contract.
- The shipped `NetworkxGraphSource`/DuckDB graph sources still re-read `graph-edges.parquet`
  per call (the S5 harness bypasses them with a run-lifetime source). Out of this profile's
  scope; a candidate for the architecture pass / ticket 04 documentation.
- Index memory: the decoded index is held per process for the file's lifetime (+347 MB wiki,
  +771 MB uda — see the table section); the S5 sweep budget (29 GB / 4 workers) absorbs it,
  and `cache=False` remains the zero-footprint escape for memory-constrained embedders.
