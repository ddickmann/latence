## Parent
Spec: retrieval-quality tooling (#158)

## What to build
Extend `expand` with zero-ops multi-hop traversal via DuckDB over the emitted graph parquet, plus a networkx reference and an optional Neo4j backend.

## Acceptance criteria
- [ ] Multi-hop expansion runs over the emitted graph parquet via an embedded engine (DuckDB), read-only per call, with no server to operate.
- [ ] A networkx reference path exists for small graphs; an optional Neo4j backend is available behind an extra.
- [ ] Hop depth is bounded; stateless (holds nothing between calls); tested with a fixture graph (ADR-0051).

## Blocked by
- #159 — query-time seam
- #161 — graph-feature enrichment
- #168 — expand 1-hop
