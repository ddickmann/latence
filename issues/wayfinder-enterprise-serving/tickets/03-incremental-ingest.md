# 03 — Incremental ingest: document churn without full rebuild

Type: task
Status: claimed
Blocked by: —

## Why (Dennis, 2026-08-12)

The campaign proves build-and-test; enterprises live with document churn — incremental
upserts, re-extraction, graph drift. Nothing in the architecture blocks it; nothing proves it
either. This ticket makes churn a first-class, verified capability.

## Scope

1. **Design first** (ADR, house grill-quality): the increment unit (document), what an upsert/
   delete invalidates (chunks → mentions → entities → graph nodes/edges/hyperedges → context
   headers → embeddings → BM25 stats/postings → exports), and the recompute strategy per
   stage. Key tensions to resolve explicitly:
   - corpus-level statistics (BM25 idf/avgdl, profiling) shift on every churn — exact
     recompute vs bounded-staleness, and where the boundary is;
   - disambiguation/consolidation decisions can flip when new surfaces arrive — scope of
     re-resolution;
   - graph features (degree, community, centrality) drift — when are they refreshed.
2. **Implement the increment path** in `latence-core`: `add/update/remove` documents against
   an existing run's store, reusing the Runner's checkpoint machinery where it fits.
3. **The equivalence proof (the gate)**: on a real S3 export corpus, split it (e.g. 90% base
   build + 10% incrementally added; plus an update and a delete batch) and prove the
   incremental store's **exported retrieval state is equivalent to a full rebuild** of the
   same final corpus: identical row sets, identical graph topology, identical retrieval
   rankings on the S5 query set (embeddings bitwise where determinism permits, else
   documented tolerance with rationale). Any deliberate divergence (bounded staleness) is
   measured and documented, never silent.
4. **Wire into the production lane**: CLI verbs + campaign wizard awareness (config block),
   documentation section (hand to ticket 04 for final placement).

## Non-negotiables

- The equivalence proof runs on real data (a real S3 dataset locally available), not toys.
- Default batch behavior stays byte-identical; incremental is additive.
- No LLM calls required for churn on hand-authored-schema corpora; induced-schema corpora
  reuse the frozen schema (churn must not re-trigger induction silently).
- Tests, ruff, mypy --strict clean; the ADR lands with the code.

## Gate artifact

ADR + equivalence report (row/topology/ranking diffs, all zero or documented) + CLI/config
surface. Dennis inspects.
