# 19 — Task: validate per-call KV open under concurrent query load

Type: task
Status: open
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
