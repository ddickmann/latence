## Parent
Spec: retrieval-quality tooling (#158)

## What to build
Extend Graph Assembly to compute node centrality and community and attach them to graph nodes and to each chunk's context header, so downstream expand and pack can consume them.

## Acceptance criteria
- [ ] Graph Assembly computes a centrality measure (degree/PageRank) and a community/cluster id per node.
- [ ] The features are emitted as graph-node columns and projected onto chunks additively; byte-identical when the enrichment is disabled.
- [ ] Determinism holds across runs; covered by conformance + `latence stack validate` (ADR-0051).

## Blocked by
- None — can start immediately
