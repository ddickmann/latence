# Ticket 01 report — hypergraph + graph lane latency at SOTA

Status: measurements complete on all five Mac-local exports; uda-at-full-sweep confirmation
deferred to the post-sweep S5 re-run (map ruling 1 — the pod was not touched).

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

TABLE_PLACEHOLDER

The one-time index build runs at first open per process (transparent-at-load per the ticket) and
is excluded from per-query numbers; it is reported per dataset above and amortises over any real
run (the S5 sweep pays it once per worker per dataset).

## Parity proof (the gate)

Method: for every usable gold query (full `benchmark.s5.goldmap`/`datasets` sets), each lane is
executed through every code path and compared with `==` — the ranked chunk `(id, score)` lists,
the CoE selection, AND the complete `HypergraphSignals` (ρ ids/order/float scores, solver
fields, anchors, vertex confidences, stage counters). Paths: per-call scan (pre-ticket code,
verbatim) vs shared accelerated index vs pure-Python index kernels; tiered `GraphRetriever` vs
the preserved exhaustive resolution. Runner: `tickets/01-latency/parity.py`; raw results in
`tickets/01-latency/results/parity-*.json`.

PARITY_PLACEHOLDER

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
- Index memory: the decoded uda index is held per process; the S5 sweep budget (29 GB /
  4 workers) absorbs it — measured resident growth noted in the profile logs.
