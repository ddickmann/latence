## Parent
Spec: retrieval-quality tooling (#158)

## What to build
An optional Level-2 orchestrator `retrieve(query, backend, pipeline)` that chains query-signal-gen → the customer's first-stage search → the candidate processors, adapting to the backend's declared capabilities and dropping legs it lacks.

## Acceptance criteria
- [ ] `retrieve` runs an opinionated, config-overridable pipeline (fetch-per-modality → fuse → expand → rerank → pack) over a supplied `RetrievalBackend`.
- [ ] The pipeline composes only the legs the backend's declared capabilities + emitted signals support, dropping the rest gracefully.
- [ ] Orchestration is stateless (first-stage search runs on the backend); tested end-to-end against the faithful fake backend (ADR-0052).

## Blocked by
- #159 — query-time seam
- #165 — fuse
- #166 — rerank (cross-encoder)
- #168 — expand 1-hop
