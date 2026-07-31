# 15 — Task: M4 — knapsack chain-of-evidence selector

Type: task
Status: open
Blocked by: 14

## Question

Implement spec 3.1–3.3: UtilityModel protocol + HeuristicUtility, lazy-greedy marginal loop over v(e|A) with A_E-backed redundancy/connectivity, theta_min stop, and knapsack_refine reusing the existing Packer unchanged on frozen values. Also implement **[ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md)**'s
**exact query-time verification pass**: after selection, re-count the 15–40 chosen Hyperedges with
the *generation* tokenizer and trim the tail if the true total overflows. Mandatory (not optional)
when the stored tokenizer identity differs from the configured query tokenizer. Never apply a
cross-family scale factor.

Note [ticket 01](01-research-spec-vs-repo-audit.md): the existing Packer is the same greedy family,
not DP/ILP, so the spec's `knapsack_refine` "boundary repair" may be a no-op — decide whether it
survives, is replaced by a real DP pass, or is dropped, and record which.

Tests per spec M4: budget never exceeded; identical relation signature never selected twice; greedy+refine ≥ greedy on value; determinism; **the emitted set never exceeds the real budget under the generation tokenizer**, including when the index tokenizer under-counted.
