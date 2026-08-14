# 15 — Task: M4 — knapsack chain-of-evidence selector

Type: task
Status: resolved
Blocked by: 14

## Question

Implement spec 3.1–3.3 **on the gated candidate set from M3** (post `s_hyp`-prune + vertex/coverage gate — ticket 20 addendum; selection over the raw expansion is the measured failure mode). Use the **incremental sparse overlap** (running max, one sparse product per acceptance) — the dense `|E_c|²` matrix is the measured latency cap. Then: UtilityModel protocol + HeuristicUtility, lazy-greedy marginal loop over v(e|A) with A_E-backed redundancy/connectivity, theta_min stop, and knapsack_refine reusing the existing Packer unchanged on frozen values. Also implement **[ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md)**'s
**exact query-time verification pass**: after selection, re-count the 15–40 chosen Hyperedges with
the *generation* tokenizer and trim the tail if the true total overflows. Mandatory (not optional)
when the stored tokenizer identity differs from the configured query tokenizer. Never apply a
cross-family scale factor.

Note [ticket 01](01-research-spec-vs-repo-audit.md): the existing Packer is the same greedy family,
not DP/ILP, so the spec's `knapsack_refine` "boundary repair" may be a no-op — decide whether it
survives, is replaced by a real DP pass, or is dropped, and record which.

Tests per spec M4: budget never exceeded; identical relation signature never selected twice; greedy+refine ≥ greedy on value; determinism; **the emitted set never exceeds the real budget under the generation tokenizer**, including when the index tokenizer under-counted.

## Answer

**BUILT + green** (built from a full handoff; independently verified — commit
[`5136383`] contains exactly the four permitted files, and the 14 new tests pass on a fresh
run). Full detail in [M4-RESULT.md](../M4-RESULT.md); the load-bearing points:

1. **The M3-deferred selection-ranking test PASSES unweakened**: the planted 2-hop chain
   takes both seats; the maximally-advantaged lexical distractor (sim 1.0, γ 0.95,
   anchor-touching) loses the contested seat via the weak discount + redundancy with the
   anchored evidence + the connectivity bonus. The end-to-end claim the whole design makes —
   structure beats lexical similarity AT SELECTION — is now pinned by test.
2. **`knapsack_refine` DROPPED, with recorded rationale** (the open question from ticket 01):
   the existing Packer is the same greedy family (composition = structural no-op); a
   frozen-value DP optimises a linear proxy that ignores the redundancy/connectivity
   interaction terms, so its "wins" are unverifiable; and theta_min + the ADR-0058 trim
   would re-break any repaired boundary. The graduated fog item closes as *dropped-by-
   evidence-and-argument*, per the standing rule.
3. Incremental sparse redundancy (running max-shared + inverted index) — the ticket-20
   measured requirement; no dense matrix, zero new deps, package stays pydantic-only.
4. **ADR-0058 verification pass live**: exact re-count of the selected set, mandatory on
   counter-identity mismatch, lowest-marginal-first trim, result carries
   verified/trimmed/total_cost — never a scale factor.
5. Dedup invariant pinned: identical-signature duplicates hash to the same content-addressed
   id (ADR-0057), so spec 3.2's red=1.0 hard-skip case is unrepresentable by construction.
6. One implementation defect caught in dev (documented in M4-RESULT): bound-only acceptance
   preferred an isolated edge over a chain-adjacent one; acceptance now uses true marginal
   density.

Suite: **4694 passed / 36 skipped / 0 failed** (M3 baseline 4679 + 14 new + 1 from the
ticket-10 purge test), `ruff` + `mypy --strict` clean. Default behavior unchanged (new
module only). M5 (chains + traces) is the last build step before the eval.

## Close-out audit addendum (2026-08-02)

Answer #5's claim that spec 3.2's hard-redundancy case is "unrepresentable by construction"
was WRONG — dedup collapses only identical (entity set, signature) pairs; strict-subset
signatures and same-signature/different-entity-set records coexist. Corrected disposition
(declared deviation, entity-overlap discount as the approximation, no unmeasured hard skip)
in [CLOSEOUT-AUDIT.md](../CLOSEOUT-AUDIT.md) (F-COE); coexistence + residual pinned in
`test_coe.py::test_subset_signature_records_coexist_and_get_the_overlap_discount`.
