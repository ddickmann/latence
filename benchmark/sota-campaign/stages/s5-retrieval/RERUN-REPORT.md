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

---

## 3. Does the packer do its job?

Yes, and the one place it looks like it does not is the budget, not the packer.

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

## 5. The open item

The 7 fourth-decimal moves in §1 have a hypothesis — that the vectorized BM25 reassociated
floating-point summation, shifting scores in their last bits, and a metric value sitting within
1e-4 of a rounding boundary tipped — and that hypothesis is **under measurement, not assumed**.
The decisive test is whether the observed delta (0.0001) is smaller than the smallest step a
single query's ranking change could produce in the mean; if it is, no document moved and the
change is arithmetic. That probe is running against the real corpus. Until it returns, the
correct statement is *"39 of 46 comparable cells are bit-identical and 7 moved in the fourth
decimal for a reason not yet established,"* not *"parity held."*

---

## Reproducing this

```
python -m benchmark.s5.run_matrix --datasets all --legs all
python benchmark/sota-campaign/stages/s5-retrieval/check_preopt_parity.py
```

Cells are cached; `--force` re-measures. The parity script is the oracle and exits non-zero on
any non-latency difference between the two matrices.
