## Parent
Spec: retrieval-quality tooling (#158)

## What to build
Query-time sparse generation (reusing the SPLADE composition) and an exact Cython BM25 rescorer over a caller-supplied candidate set — never a first-stage index.

## Acceptance criteria
- [ ] Query-side sparse generation mirrors the index-time SPLADE composition for the query text.
- [ ] A BM25 rescorer scores a caller-supplied candidate set exactly (IDF from supplied stats or the candidate pool), optimized for large candidate sets.
- [ ] Never builds or holds a first-stage index (stateless, ADR-0048); tested at the input→output boundary.

## Blocked by
- #159 — query-time seam
- #162 — SPLADE composition
