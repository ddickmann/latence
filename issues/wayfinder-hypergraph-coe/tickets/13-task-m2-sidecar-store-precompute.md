# 13 — Task: M2 — sidecar store + precomputation

Type: task
Status: open
Blocked by: 03, 05, 12

## Question

Implement the sidecar per **[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md)**
(which supersedes spec §1.6/§1.7's layout) and ticket 05's tokenizer ruling:

- `entities.parquet`, `hyperedges.parquet` — durable, inspectable, engine-agnostic form.
- `eu.kv` — derived point-lookup read path: `edge_id → {entity_ids, token_cost, gamma, support,
  has_relations}`, `entity_id → {edge_ids[], gammas[]}` (postings γ-sorted, **γ inline**),
  `edge_id → tau_text`. Rebuildable from the parquet.
- `token_cost` precomputed per ticket 05's ruling.
- Two ID-payload vector collections (`entity_vecs`, `eu_vecs`) behind the existing engine seam.

**Explicitly NOT built** (ADR-0056, measured in [ticket 18](18-prototype-residency-simulation.md)):
global `A_E` edge-adjacency and global incidence CSR. `edge_adjacency.npz` / `incidence.npz` do not
exist. Do not reintroduce them as an optimisation — they were measured slower than local recompute
and infeasible above ~10k EUs.

Tests: KV is byte-reproducible from the parquet (rebuild equals original); postings are γ-sorted
with γ inline; incremental append equals full rebuild; nothing is retained across calls (the store
handle holds no state between queries).
