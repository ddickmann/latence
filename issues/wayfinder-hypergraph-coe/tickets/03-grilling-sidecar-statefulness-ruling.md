# 03 — Grilling: where the hypergraph sidecar sits under the stateless boundary

Type: grilling
Status: resolved
Blocked by: 01, 18

## Question

The spec's sidecar store (entities/hyperedges parquet, incidence CSR, A_E, postings, eu_text KV) must not quietly reintroduce the database ADR-0017 refused or break ADR-0048's "tooling holds no index". Proposed shape to grill: index-time = pipeline emits the sidecar as files (ADR-0049 extension); query-time = `latence-retrieval` loads them read-only per call (ADR-0051 DuckDB precedent). Does memory-resident CSR/postings between calls count as "holds no index"? Where is the line for the entity_kb/alias table? Outcome: an ADR (amendment or new) naming the boundary + the sidecar artifact contract; vector engine holds exactly `entity_vecs`/`eu_vecs` with ID-only payloads.

## Answer

**Ruling: strategy A′ — strict per-call over a point-lookup KV. Recorded as
[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md).**

Dennis approved the strict / hold-nothing family; realized as **A′ (KV point lookups)** because
plain A (columnar parquet scan) is ruled out by the same measurement — 69.5 ms p50 at 1M EUs,
degrading with corpus size, vs 2.26 ms flat for A′.

Decided on the evidence from [ticket 18](18-prototype-residency-simulation.md), not from principle:

- **ADR-0048 is NOT amended.** Statelessness stays structural — open, read, close, per call, the
  same discipline as `DuckDBGraphSource`. The escape hatch ADR-0048 rejected stays shut, now with
  measurement behind it rather than only conviction.
- **The sidecar is a pipeline artifact (files, ADR-0017/0049)**; the KV is a derived read-path
  artifact rebuildable from the parquet.
- **No global `A_E`, no global incidence CSR** — infeasible above ~10k EUs (2.6 TiB projected at
  1M) *and* slower than local recompute where it fits. Both drop out of the layout.
- **Per-entity postings carry γ inline** so candidate top-k needs no row fetch.
- **Vector engine holds two ID-payload collections only** (`entity_vecs`, `eu_vecs`), ANN entry only.
- **Entity KB / alias table** reaches the query path as part of the same per-call KV read — no
  separate resident structure. (Model loading for query-side NER/linking is a different question:
  ticket 06.)

Open follow-up spun out, not blocking: per-call KV open under **concurrent** load is untested
(ticket 19).
