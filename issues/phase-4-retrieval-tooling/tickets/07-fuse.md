## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A one-liner `fuse` that merges several ranked candidate lists into a single ranked list, via RRF or weighted score fusion, registered under `latence.retrieval`.

## Acceptance criteria
- [ ] `fuse` merges N ranked lists into one, supporting RRF and weighted-score modes, selectable by config.
- [ ] Deterministic ordering for tied inputs; handles disjoint and overlapping candidate ids.
- [ ] A pure transform over caller-supplied lists; tested at the input→output boundary.

## Blocked by
- #159 — query-time seam
