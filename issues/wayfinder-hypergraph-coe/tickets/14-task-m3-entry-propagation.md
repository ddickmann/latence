# 14 — Task: M3 — entry + confidence propagation

Type: task
Status: open
Blocked by: 06, 13

## Question

Implement spec Part 2 per ticket 06's model policy, on the **ADR-0056** read path:

- Entry entities: linked branch (query NER + linking) ∪ ANN branch; restart vector anchored on the
  linked branch only.
- Candidate universe from the KV postings (γ inline → top-k truncation with no row fetch), cap
  |E_c| ≤ 2000.
- **`B_q` is built from the fetched entity lists** — there is no global incidence CSR to slice
  (ADR-0056). Edge overlap, where the selector needs it, is recomputed on the candidate set.
- RWR propagation behind a `Propagator` seam (RWR default; heat-kernel per spec 6.2 stays fog).

Tests per spec M3: `S` column-stochastic; convergence within K≤10 at λ=0.2; a synthetic 2-hop
corpus ranks the bridging hyperedge above a lexically-similar distractor. Plus: the online path
holds nothing between queries.
