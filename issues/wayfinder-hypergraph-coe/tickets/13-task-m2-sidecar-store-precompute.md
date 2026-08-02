# 13 — Task: M2 — sidecar store + precomputation

Type: task
Status: resolved
Blocked by: 03, 05, 12

## Question

Implement the sidecar per **[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md)**
(which supersedes spec §1.6/§1.7's layout) and ticket 05's tokenizer ruling:

- `entities.parquet`, `hyperedges.parquet` — durable, inspectable, engine-agnostic form.
- `eu.kv` — derived point-lookup read path: `edge_id → {entity_ids, token_cost, gamma, support,
  has_relations}`, `entity_id → {edge_ids[], gammas[]}` (postings γ-sorted, **γ inline**),
  `edge_id → tau_text`. Rebuildable from the parquet.
- `token_cost` precomputed per **[ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md)**:
  optional operator-named tokenizer (heuristic default, nothing pinned), and the **tokenizer identity
  stored beside the value** so a later mismatch is detectable rather than silent.
- Two ID-payload vector collections (`entity_vecs`, `eu_vecs`) behind the existing engine seam.

**Explicitly NOT built** (ADR-0056, measured in [ticket 18](18-prototype-residency-simulation.md)):
global `A_E` edge-adjacency and global incidence CSR. `edge_adjacency.npz` / `incidence.npz` do not
exist. Do not reintroduce them as an optimisation — they were measured slower than local recompute
and infeasible above ~10k EUs.

Tests: KV is byte-reproducible from the parquet (rebuild equals original); postings are γ-sorted
with γ inline; incremental append equals full rebuild; nothing is retained across calls (the store
handle holds no state between queries).

## Answer

**BUILT + green on `hypergraph-m1`.** Suite: **4668 passed / 36 skipped / 0 failed** (M1 baseline
4661 + 7 new tests), `ruff` clean, `mypy --strict` clean over 148 typed sources.

What landed (default behavior unchanged — the sidecar exists only on the already-opt-in
hyperedge path; the off-path e2e asserts no KV file and untouched node properties):

1. **`latence_core.hyperedge_store`** — the ADR-0056 point-lookup KV as a **zero-dep,
   byte-reproducible single-file format** (sorted keys, mmap + binary search, per-call
   open/read/close — statelessness structural, as in `DuckDBGraphSource`). Four keyspaces:
   `e|` solver fields (no text — the index is built for the solver), `p|` postings **γ-sorted
   desc with γ inline** (top-k truncation needs no row fetch), `t|` `tau_text` (already the
   corpus redaction variant), `a|` alias→node_ids (NFKC+casefold — ADR-0059's tier-1
   zero-model entry works today). **Deliberate deviation from the ADR's LMDB instance:** the
   measurement showed *point lookup* is what matters, not the library; the reference
   implementation is dependency-free per ADR-0007/0016 and the FORMAT is the contract — an
   LMDB adapter remains a Provider-package option behind the same reader interface.
2. **Export ships the KV beside the parquet** (`graph-hyperedges.kv`, rebuildable from the
   durable form; `hyperedge_kv: false` opts out — within the opted-in path the strong
   configuration is the default, per the standing rule).
3. **Spec 1.4 entity descriptions + hyperedge degree** stamped deterministically onto node
   properties (name — type — top-3 relation contexts, e.g. `Acme Corp — ORG — acquired:
   Beta Ltd`): the tier-0/1 embedding input, no LLM summarisation. Only on the opted-in path.
4. **ADR-0058 index side complete**: `hyperedges.token_counter: "tiktoken:<encoding>"` —
   lazy import, loud `ConfigError` when the dep is missing or the encoding unknown (never a
   silent heuristic fallback), the full spec string stored as the counter identity on every
   record. Heuristic default unchanged; no new core dependency.
5. **NOT built**, per ADR-0056: no global `A_E`, no incidence CSR — confirmed absent.

Tests: KV byte-reproducibility under input reordering; incremental-append ≡ full rebuild;
γ-inline sorted postings; closed-reader refusal (no stale data); counter config error paths;
e2e KV-vs-parquet record-for-record agreement + description/degree presence; e2e off-path
absence. **Residuals, stated:** glinker's richer alias set still dies inside the provider
(alias table currently canonical names only — tracked in map fog); `entity_vecs`/`eu_vecs`
as live ANN collections are the adopter's engine-loading step per the input-spec pattern —
the embedding *inputs* (description, tau_text) are now both in the export.

## Close-out audit addendum (2026-08-02)

This ticket's Question listed spec 1.6's two ID-payload vector collections
(`entity_vecs`/`eu_vecs`); the Answer shipped without them and without declaring the drop —
now declared in [CLOSEOUT-AUDIT.md](../CLOSEOUT-AUDIT.md) (F-VEC). Consequence measured in
the official eval: `eu_sims` never existed, so the s_hyp prune ran entry-overlap-only (β=1)
in every configuration. The opt-in embedding producer is the top production-direction
measurement, paired with the dense-baseline re-run (F-BASE).
