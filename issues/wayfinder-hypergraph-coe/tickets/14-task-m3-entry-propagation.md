# 14 — Task: M3 — entry + confidence propagation

Type: task
Status: claimed
Blocked by: 06, 13

## Question

Implement spec Part 2 per ticket 06's model policy, on the **ADR-0056** read path:

- Entry entities **tiered per [ADR-0059](../../../docs/adr/0059-query-side-entry-is-tiered-zero-model-default-with-gliner2-strictly-opt-in.md)**:
  tier 0 = ANN over `entity_vecs` with the caller's own query embedding (no new model, the default);
  tier 1 = alias point-lookups against the KV (zero models); tier 2 = GLiNER2 query NER + `w_relq`
  from one load, opt-in behind an extra, no default checkpoint. The extra wraps the upstream library
  directly — **never** depend on `latence-extract-gliner2`/`latence-core`.
- Restart vector anchored on the linked branch (`V_llm`) only; when tiers 1–2 are off it degrades to
  ANN-recalled anchors, and `w_relq`'s weight must **redistribute cleanly** rather than assume the
  signal exists.
- Candidate universe from the KV postings (γ inline → top-k truncation with no row fetch), cap
  |E_c| ≤ 2000.
- **`B_q` is built from the fetched entity lists** — there is no global incidence CSR to slice
  (ADR-0056). Edge overlap, where the selector needs it, is recomputed on the candidate set.
- **Paper-faithful order (ticket 20 addendum — verified 2× recall, 2.7× faster):**
  expand (hop-priority truncation, hub cap) → score `s_hyp = β·s_entry + (1−β)·s_rel` →
  **prune to TopK_h BEFORE propagating** (K_h ~ 200) → RWR on the small query-aware hypergraph
  behind a `Propagator` seam (heat-kernel per spec 6.2 stays fog) → **vertex gate `c* ≥ θ_v` +
  coverage over retained entities** (paper Eq 30–33). Never propagate over the raw expansion.

Tests per spec M3: `S` column-stochastic; convergence within K≤10 at λ=0.2; a synthetic 2-hop
corpus ranks the bridging hyperedge above a lexically-similar distractor. Plus: the online path
holds nothing between queries.
