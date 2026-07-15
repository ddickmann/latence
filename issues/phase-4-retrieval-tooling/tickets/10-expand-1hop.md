## Parent
Spec: retrieval-quality tooling (#158)

## What to build
An `expand` one-liner that widens and/or re-scores candidates using the 1-hop KG columns denormalized onto each chunk — works on any vector store, no graph database.

## Acceptance criteria
- [ ] `expand` returns neighbor-expanded and/or graph-rescored candidates using the denormalized KG columns present on candidates.
- [ ] A pure transform; no graph database required; works regardless of the customer's store (ADR-0051).
- [ ] Tested at the input→output boundary with fixture candidates carrying KG columns.

## Blocked by
- #159 — query-time seam
- #161 — graph-feature enrichment
