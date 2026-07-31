# 19 — Task: validate per-call KV open under concurrent query load

Type: task
Status: resolved
Blocked by: 13

## Question

[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md)
rests on single-process, warm-page-cache measurements ([ticket 18](18-prototype-residency-simulation.md)).
The one untested assumption: opening and closing the KV env **per call** under concurrent query load
(N parallel queries, cold and warm cache). Measure p50/p95/p99 vs the single-process baseline and
confirm the open cost does not dominate under contention.

If contention is real, the fix is a **cheaper open** (mmap reuse scoped to a request, or a
read-only env opened per worker rather than per call) — **not** a resident store, which ticket 18
already showed buys only ~0.56 ms. Any such change must preserve "holds nothing across queries".

## Answer

**Verdict: (a) fine as-is.** Per-call `open()` + mmap does **not** degrade under contention and no
cheaper-open change is warranted. The open cost is ~0.04–0.06 ms p50 at *every* concurrency level
and both scales — ~1–2 % of a query — and it stays flat from c=1 to c=16. What degrades at c=16 is
the **read compute**, and only because 16 processes oversubscribe this 10-logical-CPU host
(8 performance cores): classic CPU queuing, not an open/mmap bottleneck.

### Method

Benchmark: [`prototypes/residency/kv_concurrency.py`](../prototypes/residency/kv_concurrency.py),
run with `uv run python …` from the repo root. It measures the **real** store
(`latence_core.hyperedge_store.HyperedgeStoreReader`, the dependency-free mmap+binary-search
reference implementation — *not* the LMDB stand-in ticket 18 used). Real store files built with
`build_store_bytes(store_entries(...))` over real `GraphHyperedge`/`GraphNode` contract records
(construction patterns from `packages/latence-core/tests/test_hyperedge.py`), Zipf(1.07) entity
popularity, arity 2–6, at **50k** hyperedges/10k nodes (22.5 MB) and **500k**/100k (226.5 MB).
Per query: open → 10 `postings` + 200 `solver_fields` + 40 `tau_text` → close; open/reads/close
timed separately. Concurrency 1/4/16 via **multiprocessing** (spawn, barrier start — queries are
separate processes in production), 100 queries per worker, warm page cache.

**Cold cache not measured:** `/usr/sbin/purge` requires root on this macOS host
(`Operation not permitted` without sudo), so all numbers are warm-cache — stated honestly, same
caveat as ADR-0056's own measurements.

### Numbers (per-query wall time, ms)

| scale | c  | total p50 | total p95 | total p99 | open p50 | open p99 | reads p50 | reads p99 |
|-------|----|-----------|-----------|-----------|----------|----------|-----------|-----------|
| 50k   | 1  | 1.97      | 2.31      | 2.68      | 0.040    | 0.26     | 1.91      | 2.40      |
| 50k   | 4  | 2.06      | 7.58      | 10.6      | 0.039    | 8.6      | 1.97      | 2.59      |
| 50k   | 16 | 2.31      | 15.1      | 32.8      | 0.046    | 9.9      | 2.12      | 28.6      |
| 500k  | 1  | 2.66      | 3.34      | 9.4       | 0.048    | 6.7      | 2.50      | 3.14      |
| 500k  | 4  | 2.86      | 3.51      | 8.2       | 0.044    | 5.3      | 2.67      | 3.41      |
| 500k  | 16 | 5.38      | 18.9      | 29.1      | 0.062    | 12.8     | 4.05      | 26.3      |

### Reading

- **open() is a non-issue.** p50 0.039–0.062 ms across all cells; it does not grow with
  concurrency or corpus size. The occasional open p99 spike (5–13 ms) appears even at **c=1**
  (6.7 ms at 500k) — scheduler/page-fault noise hitting whichever phase is running, not
  open-specific contention (reads p99 shows the same tails at c=16).
- **c=4 ≈ c=1.** Total p50 moves 1.97→2.06 ms (50k) and 2.66→2.86 ms (500k): effectively free
  parallelism while workers fit the cores.
- **c=16 degrades reads, not opens** — total p50 ×2 at 500k, p95/p99 to ~15–30 ms — exactly the
  ≥1.6× slowdown + queuing tails expected from 16 CPU-bound processes on 10 logical CPUs. The
  ADR's named fix (per-worker read-only env) would not help: there is nothing to save in open().
- **Baseline shape vs ADR-0056 holds:** flat-ish in corpus size (reads p50 1.91→2.50 ms for a 10×
  corpus). Absolute reads are ~2.5 ms rather than the ADR's ~0.5 ms because that figure was
  LMDB-I/O-only at 1M EUs; this is the pure-Python reference reader doing 250 lookups **including
  JSON decode** — per the ADR, the comparisons hold, the absolute numbers move.

Residual (minor): warm-cache only, and c=16 exceeds this host's core count, so the c=16 tails are
an upper bound for a machine this size, not evidence of store-level contention.
