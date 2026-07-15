## Parent
Spec: retrieval-quality tooling (#158)

## What to build
Port the Apache-licensed Knapsack QKP context-packer as a `pack` candidate-processor that selects a token-budgeted subset maximizing coverage/relevance minus redundancy, with optional richness signals and graceful degradation.

## Acceptance criteria
- [ ] `pack` returns a token-budgeted subset under a configurable budget, maximizing a coverage/relevance objective minus redundancy.
- [ ] Optional signals (centrality, recency, roles, clusters, similarity) are used when supplied and gracefully absent otherwise; runs on defaults day one.
- [ ] The ported solver carries clean Apache-2.0 provenance and is deterministic on CPU by default (ADR-0049).
- [ ] Tested at the input→output boundary (budget respected, redundancy penalized) with prior-art fakes.

## Blocked by
- #159 — query-time seam
