## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A real, capability-declaring Qdrant `RetrievalBackend` adapter, as the reference example adopters copy for their own store.

## Acceptance criteria
- [ ] A Qdrant adapter declares its capabilities and implements `search` (dense/sparse/filter as supported), returning `Candidate`s.
- [ ] Ships behind an extra; documented as the reference adapter pattern.
- [ ] Protocol conformance is unit-tested; live-store validation against Qdrant done out of band.

## Blocked by
- #159 — query-time seam
