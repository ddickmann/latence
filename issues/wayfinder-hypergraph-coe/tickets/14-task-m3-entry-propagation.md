# 14 — Task: M3 — entry + confidence propagation

Type: task
Status: resolved
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

## Answer

**BUILT + green on `hypergraph-m1`.** Suite: **4679 passed / 36 skipped / 0 failed** (M2
baseline 4668 + 11 new), `ruff` + `mypy --strict` clean. Default behavior untouched — this is
a NEW module in `latence-retrieval` (`hypergraph.py`); nothing existing changed.

What landed:

1. **`latence_retrieval.hypergraph`** — entry + propagation over the sidecar KV, in the
   **ticket-20-verified order**: expand (hop-priority, hub-capped) → `s_hyp` prune to K_h
   (Eq 16–19) → two-stage RWR on the small `B_q` (Eq 26–28) → mean-pooled ρ + vertex gate
   θ_v + coverage (Eq 29–33). **Zero-dep and pure Python** — viable precisely BECAUSE the
   verified order prunes before propagating; the package stays pydantic-only.
2. **ADR-0059 tiers 0/1 live**: tier 0 = caller's `ann_node_ids` (their own query embedding,
   no new inference, weight α); tier 1 = `alias_surfaces` against the KV's NFKC/casefold
   alias table (weight 1.0 — spec 2.1's anchored restart). Tier 2 plugs into the same two
   arguments from a future opt-in extra; nothing imports a model. `eu_sims` absent → the
   prune degrades to entry-overlap (β=1), documented not silent.
3. **`HyperedgeKV`** — a second, independent implementation of the ADR-0056 store format
   (the format is the contract; the package cannot import latence-core). **Interop pinned**:
   core's writer ↔ this reader, byte round-trip, in core's suite.
4. **`Propagator` seam** (spec 6.2): `RWRPropagator` default; heat-kernel remains fog until
   the eval can arbitrate. `HypergraphSignals` exposes ρ, per-vertex c*, anchors, and
   stage-narrowing counts (honest observability).

**Two spec deviations, math-backed** (pinned in test docstrings, not hidden):

- Spec M3's "converges <1e-6 within K≤10 at λ=0.2" is **impossible in the worst case**:
  contraction is (0.8)^t → 1e-6 at t≈62 on a slow-mixing chain. Pinned instead: the
  geometric bound holds, AND the default K=6 is **ranking-converged over the mass-bearing
  head** — ordering is what selection consumes.
- "ρ alone ranks the bridge above a lexically-similar distractor" is **structurally false**
  when the distractor touches the anchor (any diffusion puts anchor-touching evidence
  closer; the paper's Eq-32 score adds entry/coverage/textual terms above ρ). Pinned
  instead: the 2-hop entity **gains confidence from zero** (the "reasoning beyond cosine"
  claim), the bridge survives all gates at ≥0.3×ρ(direct), and the distractor gains no
  structural advantage. The selection-level assertion moves to M4's utility tests, where
  the spec itself puts the final ranking.

M4 consumes `HypergraphSignals` directly (ρ + fields + vertex confidences are exactly the
selector's inputs; token_cost + counter identity ride in `fields` for the ADR-0058
verification pass).
