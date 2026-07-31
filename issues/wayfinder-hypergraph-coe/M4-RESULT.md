# M4 result — chain-of-evidence selector (ticket 15)

Status: built + green. Module: `packages/latence-retrieval/src/latence_retrieval/coe.py`; tests:
`packages/latence-retrieval/tests/test_coe.py` (14, all deterministic, PYTHONHASHSEED=0).

## What was built

- `UtilityModel` Protocol (spec 3.5's seam: `base` + `marginal`) + `HeuristicUtility` (spec 3.2
  v0). One recorded deviation from spec 3.5's `(query_feats, edge_feats)` signature: every
  feature is already query-conditional (ρ, sim, anchor overlap are computed for THIS query), so
  the query folds into `EdgeFeatures` rather than travelling as a separate argument.
- Absent-signal handling (ADR-0059 discipline): `eu_sims` / `relation_bonus` are optional caller
  dicts; when a signal is absent for the whole call its weight **redistributes** (v0 is the
  weighted mean over active signals; with all six active the divisor is 1.0 and the value equals
  the spec's plain weighted sum). Weak Hyperedges get the spec's ×0.6 discount.
- `select(...)`: budgeted lazy-greedy over marginal **density** `v(e|A)/token_cost`, with
  `theta_min = 0.15 · max(v0)`. Redundancy/connectivity is **incremental sparse** (ticket 20):
  a running max-shared-entities vector + connectivity flags, updated by one sparse pass over an
  inverted node→candidate index per acceptance — no dense `|E_c|²` matrix anywhere. Connectivity
  can grow as A grows (breaks strict submodularity), so heap keys seed an optimistic-connectivity
  bound and every pop re-evaluates the true marginal before accepting — the initial
  bound-only-accept variant selected an isolated candidate over a chain-adjacent one and was
  caught by the connectivity test during development. Determinism: no randomness, ascending-id
  tiebreaks (like `Packer`).
- ADR-0058 verification pass: with the caller-supplied triple (`tau_lookup`, `TokenCounter`,
  `counter_identity`), the SELECTED set is exactly re-counted and trimmed lowest-marginal-first
  on overflow. Mandatory when any selected record's stored `token_counter` differs from the
  verification identity (no config can switch it off); optional (`CoeConfig.verify_on_match`)
  on a full match. No scale factor, ever. `CoeSelection` reports `verified` / `trimmed` /
  `total_cost` (exact when verified, stored otherwise) — honest observability. A partial triple
  is a loud `ValueError`; a selected id whose `tau_text` cannot be fetched fails loudly.
- The spec-3.2 "identical relation signature ⇒ red = 1.0 hard skip" is encoded as documentation
  + a test, not code: M1's dedup (ADR-0057, `latence_core.hyperedge`) content-addresses the id
  over (corpus, entity-node set, sorted relation signature), so the would-be duplicate IS the
  same record (`support += 1`) and the hard-redundancy case is unrepresentable in any store this
  selector reads.

## knapsack_refine decision: DROPPED (option a)

Rationale (recorded in the module docstring as well):

1. Ticket 01's finding stands: the existing `Packer` is the same greedy marginal-density family,
   not DP/ILP — composing it as a "boundary repair" phase re-runs the same decision rule on
   frozen values, a structural no-op.
2. A real bounded DP over frozen `v(e|A_final)` would optimise a linear proxy that ignores
   exactly the interaction terms (redundancy, connectivity) that motivated the marginal
   formulation; a frozen-value swap can raise the proxy while lowering the true objective — an
   unverifiable "improvement" I will not ship as a win.
3. `theta_min` deliberately leaves budget unspent; a repair phase that back-fills the boundary
   with sub-threshold filler contradicts it.
4. The ADR-0058 verification pass may trim the tail anyway, re-breaking any repaired boundary.
5. Ticket 20's measurements located selection quality in pipeline order and candidate reach,
   never boundary packing.

The spec-M4 value criterion is therefore pinned as: marginal-aware greedy ≥ (strictly > in the
planted scenario) a plain value-density greedy baseline on chain value
`Σ_i v(e_i | {e_1..e_{i-1}})` (`test_greedy_beats_plain_value_density_baseline_on_chain_value`).

## The deferred selection-level ranking test (e2e)

`test_e2e_selection_picks_the_planted_chain_over_the_lexical_distractor` — the test M3's
`test_two_hop_evidence_gains_confidence_from_zero` explicitly deferred. Full path:
`_build_kv` fixture → `hypergraph_signals` → `select` → ADR-0058 verify. The distractor is the
classic lexical-noise shape: it mentions the queried entity, hogs ALL the lexical signal
(sim 1.0), carries the highest γ (0.95) — but states no relation (weak Hyperedge). The planted
chain carries relations, corpus support (3), and the realistic bridge-text signal ticket 20's
addendum established (the question names the chain's endpoints; sims 0.6/0.7). Under a 2-seat
budget, selection assembles **{h_direct, h_bridge}**; the distractor loses the contested seat
(weak discount + redundancy with the already-selected anchor evidence + the chain's
connectivity bonus). Passed without weakening the assertion; verification ran (identity match +
`verify_on_match=True`), no trim.

## Numbers

- `PYTHONHASHSEED=0 uv run pytest packages/latence-retrieval/tests/`: **639 passed, 17 skipped**
  (skips pre-existing, dep-gated), including the 14 new coe tests.
- `uv run mypy --strict packages/latence-retrieval/src`: Success, 27 source files.
- `uv run ruff check packages`: All checks passed.
- Full suite (`PYTHONHASHSEED=0 make test`): **4694 passed, 36 skipped, 0 failed** in 10:17
  (run 2026-07-31 on this branch, including the concurrent session's new latence-core
  hyperedge tests present in the working tree).

## Deviations / notes

- Anchor overlap uses the tiered restart weights (alias 1.0, ANN α) as a weighted mean —
  generalising spec 3.2's binary `V_llm` indicator; documented on `EdgeFeatures`.
- `conn` counts V0 (the anchors) from the first item, resolving spec 3.3's self-contradictory
  parenthetical ("0 for the first item" vs "V0 counts") in favour of the anchored-start reading.
- Files touched: NEW `coe.py`, NEW `test_coe.py`, `__init__.py` (exports only), this file.
