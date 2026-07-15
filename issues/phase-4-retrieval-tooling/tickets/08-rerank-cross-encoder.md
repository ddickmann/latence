## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A one-liner `rerank` that re-scores a candidate list with a config-selected, license-verified cross-encoder model.

## Acceptance criteria
- [ ] `rerank` re-scores and reorders a candidate list using a cross-encoder chosen by config; license verified (weights and code).
- [ ] Works as a pure transform over caller-supplied candidates; batch-efficient.
- [ ] Tested with a faithful fake cross-encoder that rejects what the real API rejects; real-model validation out of band.

## Blocked by
- #159 — query-time seam
