## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A reference MaxSim reranker over the raw-multivector sidecar, plus query-side multi-vector/FDE generation — experimental, off by default.

## Acceptance criteria
- [ ] Query-side multi-vector and FDE generation produce query representations matching the index-time contracts.
- [ ] A reference MaxSim reranker scores candidates against their sidecar multi-vectors; experimental and off by default; real (not faked) math (ADR-0050).
- [ ] Tested for correctness on synthetic multi-vectors.

## Blocked by
- #159 — query-time seam
- #163 — MUVERA-FDE
