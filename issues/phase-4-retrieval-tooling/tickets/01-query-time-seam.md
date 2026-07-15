## Parent
Spec: retrieval-quality tooling (#158)

## What to build
The tested foundation every query-time capability builds on: a new `latence-retrieval` package exposing the `latence.retrieval` registry group, the `Candidate` contract, the capability-declaring `RetrievalBackend` protocol, a faithful fake backend for tests, and a trivial identity candidate-processor that proves register → resolve → transform end-to-end.

## Acceptance criteria
- [ ] A `latence-retrieval` package installs with a pure-Python core (no heavy deps) and registers components under a new `latence.retrieval` entry-point group, parallel to and separate from `latence.providers`.
- [ ] A `Candidate` contract carries at least id, score, text, and metadata, and round-trips through serialization.
- [ ] A `RetrievalBackend` protocol declares a capability set (dense/sparse/bm25/filter/multivector) and a `search` returning candidates; a faithful fake backend implements it and rejects capability/dimension mismatches exactly as a real store would (anti-false-green).
- [ ] An identity candidate-processor is registered, resolved by config, and returns its input unchanged; a test drives it over fake-backend candidates.
- [ ] The package holds no durable index and runs no first-stage search — asserted by the seam's shape and tests (ADR-0048).

## Blocked by
- None — can start immediately
