# S5 retrieval — the full-scale re-run on optimized code

**What this is.** The complete S5 matrix — 7 datasets × 10 legs = 70 cells, every query, no
`--limit` — measured on current head after the ticket-01/02/03 latency work and the eleven
architecture-pass deepenings. It exists because the previous matrix was measured on code that
no longer exists, and a benchmark of superseded code answers no question worth asking.

**Where it ran.** The pod, single process per lane, `OMP_NUM_THREADS=1`, two memory-sized pools
(uda ×2, everything else ×2) inside a **5.1-core / 31 GB cgroup**. `nproc` reports 48 and `free`
reports 251 GB on that box; both are the host, not our slice, and sizing to them is how the first
launcher came within 5 GB of an OOM kill. Wall clock: 11:08 → 21:05 UTC.

**The prior matrix is preserved** in `results-preopt/` and is the oracle for everything below.

---

## 1. Did the optimizations change any answer?

`check_preopt_parity.py` compares the two matrices cell by cell, stripping `latency_ms_*` (the
thing the work was allowed to change) and `limit` (run provenance, not measurement).

| Outcome | Cells |
|---|---|
| Identical on every quality number | **39** |
| Moved, fourth decimal place only | 7 |
| Not comparable — the query population changed | 20 |
| No pre-optimization counterpart | 4 |

**The 20 incomparable cells are the honest part.** Twenty pre-optimization cells were measured
under `--limit 1000` on the three large corpora. The re-run measured them whole: hotpotqa
1,000 → 3,703, uda 1,000 → 8,583, wiki2multihop 1,000 → 12,576. A recall computed over a
different query set is a different quantity. The oracle reports these as NOT COMPARABLE rather
than counting them as either a pass or a failure, because either would be a lie about what was
measured. The re-run is strictly more complete; it is not a contradiction of the old numbers.

**The 7 that moved** are `musique__{bm25,kg,graph_rescue,graph_rescue_packed,graph_rescue_coe}`
(`recall@10` 0.3405 → 0.3404, `ndcg@10` 0.4162 → 0.4161 — the same shift, repeated identically
across every budget in every curve) and two `vidoseek` `tokens_packed_mean` values moving by
0.1–0.2 tokens in ~10,000. The affected musique legs are exactly the BM25-involving ones;
`dense`, `dense_content_only`, `coe`, `packer` and `hyperedges` did not move. See §5 — this is
under measurement, not assertion.

---

## 2. The baseline this had to beat

The table that started this work showed every graph leg *regressing* against dense retrieval.
Same metric, full query sets, current head:

| dataset | dense (baseline) | packer (the old floor) | **graph_rescue** | vs dense |
|---|---|---|---|---|
| vidoseek | 0.9466 | 0.7538 | **0.9726** | +0.026 |
| ohr_bench | 0.5890 | 0.2793 | **0.6589** | +0.070 |
| uda | 0.0594 | 0.0266 | **0.1276** | **+0.068 (2.1×)** |
| multihop_rag | 0.3328 | 0.2028 | **0.3759** | +0.043 |
| wiki2multihop | 0.6753 | 0.5446 | **0.7096** | +0.034 |
| musique | 0.3353 | 0.2500 | **0.3404** | +0.005 |
| hotpotqa | 0.5576 | 0.4103 | **0.5786** | +0.021 |

`recall@10`. **Graph rescue beats the dense baseline on all seven datasets.** The regression the
baseline table recorded is gone: no leg that appends graph evidence now lands below dense.

Document-level recall tells the same story — `doc_recall@10` for graph_rescue: vidoseek 0.9956,
ohr_bench 0.9738, uda 0.4466, multihop_rag 0.8551, wiki2multihop 0.7198, musique 0.6068,
hotpotqa 0.8119 — dense-or-better everywhere.

### Read against the right baseline, though

Dense is not the strongest rung in the matrix; `bm25` is, and `graph_rescue`'s own first stage
already contains it. Compared against **`bm25`** rather than against dense, the same cells say
something much weaker:

| dataset | bm25 | graph_rescue | Δ recall@10 | Δ nDCG@10 |
|---|---|---|---|---|
| vidoseek | 0.9705 | 0.9726 | +0.0021 | **−0.0412** |
| ohr_bench | 0.6609 | 0.6589 | **−0.0020** | **−0.0288** |
| uda | 0.1261 | 0.1276 | +0.0015 | +0.0006 |
| multihop_rag | 0.3772 | 0.3759 | **−0.0013** | −0.0050 |
| wiki2multihop | 0.7018 | 0.7096 | +0.0078 | −0.0166 |
| musique | 0.3404 | 0.3404 | 0.0000 | 0.0000 |
| hotpotqa | 0.5786 | 0.5786 | 0.0000 | 0.0000 |

So the "+0.07 on uda" headline is what **adding BM25 to dense** buys, not what the graph buys. The
graph lanes' own contribution at k=10 is within ±0.008 on recall and *negative* on nDCG for four of
seven datasets.

**There is a structural reason, and it is not a tuning accident.** The harness runs `POOL_K = 100`
(`benchmark/s5/legs.py:286`) and the pipeline's `select_pool_k` is also 100. `_append_and_dedupe`
(`packages/latence-retrieval/src/latence_retrieval/rescue.py:1071`) places the *entire* fused base
ahead of every rescued candidate by construction, and the base is already 100–200 candidates —
so the first item stamped `rescued` sits at rank ≥ 100, and `select()` then truncates the union to
`union[:100]` (`rescue.py:878`). **No rescued candidate can reach a top-10 cut, or the selector's
pool, at this configuration.**

The cells confirm the prediction rather than merely permitting it: `graph_rescue_coe` is bitwise
identical to `graph_rescue` on **7 of 7** datasets, and `graph_rescue_packed` on 4 of 7 (differing
only where the token budget truncates, always downward). A selector fed an all-base pool ordered by
a strictly decreasing score can only drop items, never reorder them.

**What this section's numbers therefore measure** is the base stage — dense+BM25 RRF, max-normalised,
with the Expander rescore — not graph traversal. `graph_rescue` − `bm25` is, to within the rounding
floor, exactly the Expander's contribution. The traversal and hypergraph lanes are architecturally
live, correct, and **measurably inert at k=10 on this matrix**: they cost latency and buy nothing the
metric can see.

### That experiment has now been run, and the lanes lost

Ticket 06 (`issues/wayfinder-enterprise-serving/tickets/06-graph-inertia/`) ran all three
configurations named above plus the interleave family, over three datasets. Its `base` arm
reproduces this matrix's `graph_rescue` cells to four decimals on full populations, so it is
comparable to the table above rather than a re-measurement of it.

The finding is not that a knob was set wrong — it is that **there is almost nothing behind the
knob**. An oracle that promotes every rescued gold chunk to the front of the list (no policy can
beat it) is worth **+0.0056** recall@10 on multihop_rag, **+0.0077** on wiki2multihop, and
**exactly 0.0000** on vidoseek, where the graph lanes reach *not one* gold chunk the base missed
across 1,142 queries. Each option in turn:

- **`select_pool_k > |base|`** — works and saturates (0 rescued candidates selected at 100, 59 at
  250, no change at 1000), and costs the packer −0.0032 recall over its own selected context.
- **larger `k`** — `append` is bitwise identical to `base` at *every* cut including k=100, on all
  three datasets. The inertia is not a k=10 artifact.
- **smaller `pool_k`** — a rescued candidate reaches the top ten only when `|base| < k`, i.e. at
  `pool_k=8`; it buys +0.0003 against a base that shrinking already cost 0.0127 (net −0.0124).
- **interleave / fuse instead of append** — monotonically worse in the fusion weight, down to
  −0.1121 at `w=1.00`. **The append-not-fuse design choice is confirmed by measurement.**

And the counterfactual, on the same queries in the same units: a perfect reranker over the base's
own top-100 is worth **+0.3724** recall@10 (multihop_rag) and **+0.0905** (wiki2multihop) — 66×
and 12× the graph's entire ceiling. The base already *retrieves* 87% of multihop_rag's gold in its
top 100 and only *ranks* 38% of it into the top 10: a ranking problem, not a reach problem.

**The claim these cells support** is that graph-as-rescue never *lowers* recall — a real property
of the append design, and one fusing demonstrably lacks — not that graph traversal improves
retrieval quality here. It does not, at any configuration tested. See
[`REPORT.md`](../../../../issues/wayfinder-enterprise-serving/tickets/06-graph-inertia/REPORT.md).

---

## 3. Does the packer do its job?

Yes, and the one place it looks like it does not is the budget, not the packer.

### What the append discipline is worth under a budget, in chain terms

Ticket 06 measured the graph payload in recall and found it empty. The complementary question —
what the *discipline* is worth once a token budget binds — is answered by the selection legs at an
**equal 4,000-token budget**. `packer` packs a graph-**fused** ranking; `graph_rescue_packed` packs
the append-not-fuse union. Same knapsack, same spend, different candidate list:

| dataset | chain_coverage@10 | Δ | full_chain_hit@10 | Δ |
|---|---|---|---|---|
| musique | 0.4575 → 0.5905 | **+0.1330** | 0.1092 → 0.2266 | **+0.1174** |
| hotpotqa | 0.6181 → 0.7960 | **+0.1779** | 0.3508 → 0.6130 | **+0.2622** |
| multihop_rag | 0.6206 → 0.7506 | **+0.1300** | 0.3282 → 0.4736 | **+0.1454** |
| wiki2multihop | 0.5547 → 0.7199 | **+0.1652** | 0.2417 → 0.4216 | **+0.1799** |

Thirteen to twenty-six points of full-chain completion at identical token spend, on the four
genuinely multi-hop corpora.

**Read this with ticket 06, not against it.** Since the rescued payload reaches almost no gold the
base missed, this gain cannot be the graph *adding* evidence — it is the cost of graph-fusion being
*removed*. Ticket 06 measured the same effect from the other side, as a monotonic decline in the
fusion weight down to −0.1121 at `w=1.00`. Both arms agree: fusing graph scores into the ranking
actively damages retrieval, and the damage is largest exactly where chains must be completed under
a budget. The graph's contribution in this system is negative-avoidance, and it is worth a lot.

**One anomaly, unexplained:** `multihop_rag` chain coverage peaks at 4,000 tokens (0.7506) and
falls at 8,000 and beyond (0.7136, 0.7112). More budget makes it worse. Not investigated.

`graph_rescue_packed` matches `graph_rescue` exactly on five of seven datasets. It loses recall on
vidoseek (0.8424 vs 0.9726) and slightly on ohr_bench (0.6352 vs 0.6589). The vidoseek budget
curve says why:

| budget | recall@10 | doc_recall@10 | tokens actually packed |
|---|---|---|---|
| 2,000 | 0.2747 | 0.9387 | 1,920 |
| 4,000 | 0.5599 | 0.9921 | 3,931 |
| 8,000 | 0.7545 | 0.9947 | 7,936 |
| 10,000 | 0.8424 | 0.9947 | 9,932 |
| 16,000 | 0.9631 | 0.9956 | 15,931 |
| 32,000 | **0.9726** | 0.9956 | 31,913 |

Two things are visible. **The packer spends what it is given** — 9,932 of 10,000; 31,913 of
32,000 — which is the backfill change working; the pre-backfill greedy stop left ~4,500 tokens
unspent at the 10k budget. And **at 32k the packed leg equals the unpacked leg exactly**
(0.9726). vidoseek is a page-image corpus with long pages; its answers do not fit in 10k tokens.
The ceiling is the budget, and `doc_recall@10` stays ≥0.9387 even at a 2k budget — the right
document is found, it is the chunk-level evidence that gets truncated.

---

## 4. What the optimizations bought

64 cells carry a comparable latency pair.

- **Median speedup 9.8×.** Maximum 464× (`ohr_bench__bm25`, 1,303 → 2.8 ms).
- At uda scale, the one that mattered: `graph_rescue` **6,648 → 646 ms (10.3×)**,
  `hyperedges` 5,362 → 79.6 ms (67×), `kg` 859 → 37.1 ms (23×).
- `wiki2multihop__bm25` 2,058 → 8.3 ms (247×); `ohr_bench__kg` 1,301 → 3.3 ms (395×).
- **Five cells got slower**, all reported rather than dropped: `vidoseek__graph_rescue_packed`
  +15.5 ms, `graph_rescue_coe` +10.8 ms, `graph_rescue` +1.7 ms (on a ~470 ms baseline, ~3%),
  and two sub-2 ms dense cells (`ohr_bench__dense` +1.0 ms, `vidoseek__dense_content_only`
  +0.18 ms). vidoseek is the smallest graph in the set, so the index-time work has little to
  bite on, and 3% on a shared 5.1-core box is inside run-to-run variance.

`uda__graph_rescue_coe` is the clearest single result: it never finished at all on the old code
across 10+ hours. It completes now.

---

## 5. The 7 moved cells — measured, and the answer was not the obvious one

The hypothesis was that the vectorized BM25 reassociated floating-point summation and tipped a
value sitting on a rounding boundary. **That is wrong, and the measurement killed it.**

Three findings, in the order they landed:

1. **The arithmetic cannot reach it.** musique/bm25's unrounded `recall@10` is
   0.3404457170718723 — it sits 4.283e-06 below the 0.34045 rounding boundary, while the
   metric's own float-summation error is 1.1e-15. Six orders of magnitude short. No
   ranking-independent arithmetic produces this move; the per-query recall multiset must have
   differed. Likewise `tokens_packed_mean` is `round(S/N, 1)` over an **integer** sum of
   `HeuristicTokenCounter` counts (`N = 1142`, verified all-int), so float accumulation cannot
   move it at all — the 16k vidoseek cell requires ≥170 tokens of genuinely different packed
   content.
2. **The ranking code is byte-identical between the two sweeps.** `git diff fa4818f..400bbce`
   over `DenseBackend`, `Bm25Backend`, `_Bm25TermMatrix`, `fuse_signals`, `POOL_K`, `_tokenize`,
   `Runner`, `fuser.py`, both `bm25.py`, `metrics.py`, `goldmap.py`, `embed.py`: zero functional
   diff. The BM25 vectorization predates *both* sweeps.
3. **The variable that changed was the environment.** The pre-optimization sweep ran four
   contended unpinned processes; the re-run ran one process at `OMP_NUM_THREADS=1`. Flipping
   only that switch on current head reproduces the effect exactly:

   | threads | `recall@10` unrounded | cell | `ndcg@10` | cell |
   |---|---|---|---|---|
   | 1 | 0.3404457170718723 | 0.3404 | 0.4161434108235836 | 0.4161 |
   | 4 | 0.3405835720484370 | 0.3406 | 0.4161918995551152 | 0.4162 |

   Same code, same corpus, same 1,209 queries: 33 get a different dense pool, 4 a different
   top-10 membership, and exactly **one** (`musique-2hop__67660_158861`, `|gold|=6`) a different
   gold-hit count. That one query is the entire delta — `1/(1209x6) = 1.37855e-04`, matching the
   observed difference to the digit.

This also corrects the premise in §1. `hyperedges`, `coe` and `packer` *do* involve BM25 and did
not move; the five that moved report identical metrics on musique because on that corpus they
collapse to **one ranking** (dense+bm25 RRF — the graph lanes contribute nothing and the 10k
budget never binds). One ranking flipped, five cells reported it.

**So: a real ranking change, not a regression, and not caused by the optimization work.** The
correct reading is 39 of 46 cells exact and 7 inside the harness's own reproducibility floor —
roughly ±1 query in 1,209 — which until now was **undeclared**.

### What was fixed

The determinism contract asserted `PYTHONHASHSEED=0` and said nothing about thread count. It now
refuses unless `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are all pinned to
1, at the API entry so the campaign stage is bound by it too. A benchmark that lets the thread
scheduler edit its results is not a benchmark, and the fix belongs in the harness rather than in
a note telling operators to remember.

---

## Reproducing this

```
python -m benchmark.s5.run_matrix --datasets all --legs all
python benchmark/sota-campaign/stages/s5-retrieval/check_preopt_parity.py
```

Cells are cached; `--force` re-measures. The parity script is the oracle and exits non-zero on
any non-latency difference between the two matrices.
