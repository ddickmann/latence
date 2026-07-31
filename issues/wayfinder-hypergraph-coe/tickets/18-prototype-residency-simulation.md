# 18 — Prototype: residency simulation (latency + quality evidence for the sidecar ruling)

Type: prototype
Status: resolved
Blocked by: —

## Question

Ticket 03 (statelessness ruling) cannot be decided from first principles: the choice between
**A strict per-call** / **B caller-owned handle** / **C memory-resident (ADR-0048 amendment)** turns on
latency and on whether the strategies differ in *what they select*. Dennis: "without knowing the
quality differences and latency impacts we cannot decide that."

Build a throwaway simulation over a synthetic hypergraph at realistic scales that measures:

1. **Latency** — full online path (entry → candidate universe → propagation → CoE select) per query,
   p50/p95/p99, phase-broken-down, per strategy, per corpus scale.
2. **Quality** — is the selected EU set *identical* between strict and resident? (Hypothesis: yes,
   they differ only in access path; a measured divergence would be a genuine finding.) Includes
   whether dropping global `A_E` changes the 2-hop candidate universe.
3. **Offline cost** — time and on-disk/in-memory size of building global `A_E` and the incidence CSR
   vs not building them. `A_E` density under hub entities is the suspected scaling wall.
4. **Memory** — resident footprint per scale, and what that implies for the adopter.

B is latency-identical to C by construction (same loaded objects, different owner), so measure A vs C
and treat B as C-on-latency, differing only in ownership/ops.

Scope: throwaway prototype, synthetic data, no pipeline integration. Not the M6 quality gate — the
"does CoE beat chunk-topk" question belongs to ticket 17.

## Answer

Prototype: [`prototypes/residency/sim.py`](../prototypes/residency/sim.py); raw numbers in
`prototypes/residency/results.json`. Synthetic hypergraph, Zipf(1.07) entity popularity (hub
entities are what stress `A_E`), arity 2–12, token_cost 40–120, 100 queries/scale, |V0|=10,
|E_c|≤2000, budget 8000 tokens, RWR λ=0.2 K=5. Strategies measured: **A** strict per-call over
parquet (DuckDB predicate pushdown), **A′** strict per-call over an LMDB point-lookup KV
(the spec's own `eu_text.lmdb` shape, env opened+closed per call), **C** fully resident.

### Finding 1 — global `A_E` is infeasible at any real scale, and buys nothing anyway

| EUs | projected nnz(`A_E`) | size | materialised? | max entity degree |
|---|---|---|---|---|
| 10,000 | 38.3M | 0.43 GiB (257 MiB actual) | yes | 4,589 |
| 100,000 | 2.90B | **32.4 GiB** | **no** | 40,566 |
| 1,000,000 | 232.8B | **2,601 GiB (2.6 TiB)** | **no** | 366,397 |

`nnz(A_E) ≈ Σ_v d_v²`, so hub entities make it grow quadratically in the hub degree. It dies
between 10k and 100k EUs. Worse, where it *did* exist (10k), **slicing it cost 2.08 ms vs 1.06 ms
to recompute the candidate-local overlap from scratch** — the stored matrix was *slower* than not
having it. Spec 1.7's "single most important precomputation for solver speed" is refuted on both
feasibility and value.

### Finding 2 — residency is not a quality decision

**100/100 identical selections, mean Jaccard 1.0000, at every scale, for both A and A′ vs C.**
The strategies differ only in access path; the selected evidence set is the same. Quality
therefore cannot arbitrate this decision — it is purely latency/ops.

### Finding 3 — latency: the cost is columnar *scan*, not statelessness

p50 / p95 ms, full online path:

| EUs | A strict-parquet | A′ strict-KV | C resident |
|---|---|---|---|
| 10,000 | 11.44 / 25.17 | 2.57 / 10.60 | 2.59 / 12.25 |
| 100,000 | 20.53 / 28.23 | 2.36 / 10.09 | 1.96 / 9.43 |
| 1,000,000 | **69.52** / 80.90 | **2.26** / 10.61 | **1.70** / 8.57 |

At 1M EUs strict-parquet spends 63 of its 69 ms on I/O (13.6 entry + 49.3 row fetch) — fetching
2000 scattered `edge_id`s from a 1M-row parquet touches most row groups, and it degrades with
corpus size. The KV point-lookup path does the same reads in **0.51 ms total** and is **flat in
corpus size**. Compute (propagate 1.3 + overlap 0.9 + select 1.9 ≈ 4 ms) is scale-invariant
because it runs on the query-local subgraph only.

**A′ costs 0.56 ms more than holding everything in memory (2.26 vs 1.70 ms at 1M) — a 33%
relative gap on a sub-3ms budget, i.e. architecturally irrelevant.** Store on disk: 124 MiB KV at
1M EUs vs a resident footprint that grows with the corpus.

### Conclusion

Strategy **A′ — strict per-call over a point-lookup KV — dominates**: statelessness is preserved
structurally (nothing held between calls, exactly ADR-0051's `DuckDBGraphSource` discipline),
selections are identical to resident, latency is within 0.6 ms of resident and flat in corpus
size, and no ADR-0048 amendment is needed. Strategy A as literally specced (columnar scan) is the
one to avoid; C's advantage does not survive contact with the numbers; B is moot once C is.

The sidecar layout changes: `edge_adjacency.npz` and `incidence.npz` **drop out** (recomputed
query-locally), `entity_postings` carries γ inline so top-k candidate truncation needs no row
fetch, and the KV becomes the primary online read path rather than a text blob store.

### Caveats (honest limits of this evidence)

1. Synthetic corpus; real entity-popularity skew may differ from Zipf(1.07). The *shape* of the
   `A_E` blowup is robust to that (it needs only hubs to exist), the latency ratios less so.
2. Warm OS page cache — favourable to both file strategies; cold-start numbers will be worse.
3. Single process, no concurrency: resident amortises load across queries, per-call KV open may
   contend under parallel load. Untested.
4. Python prototype; absolute ms are not product latencies. All strategies share the same compute
   code, so the *comparison* holds even though the absolutes will move.
5. `A_E` projection Σd_v² overestimates actual nnz by ~14% (measured at 10k) — immaterial to the
   verdict.
