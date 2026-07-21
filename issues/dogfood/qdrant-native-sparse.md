# dogfood-01 — Qdrant-native sparse vectors (SPLADE first-class ANN)

**Severity:** major (blocks tutorial §4.3) · **Area:** `latence-sink-qdrant` · **Kind:** feature (not a bug)

## What is wrong / missing
The Qdrant sink writes the Export's dense vector and carries `sparse_indices` / `sparse_values` as
**payload only** — `sink.py:242` explicitly declares named sparse vectors out of scope. The
RFC-retrieval-proof tutorial (§4.3) needs SPLADE-v3 sparse vectors loaded as **queryable Qdrant
named sparse vectors**, so the `fused` retrieval config (§6) can run a real sparse ANN leg instead of
a payload-side rescore.

## What to build
- A `--vector sparse --sparse-source <name>` load path in the sink that writes
  `qdrant_client.models.SparseVector` under a **named sparse vector** on the collection, keyed by the
  same record id as the dense points (so dense + sparse coexist on one collection).
- Collection creation declares the named sparse vector config when sparse is requested; when sparse is
  absent the dense-only collection is **byte-identical** to today (off-by-default invariant).
- Idempotent upserts; a re-load does not duplicate. A Retraction/Purge removes the sparse vector too.

## Acceptance criteria
- [ ] `latence-sink-qdrant load ... --vector sparse --sparse-source splade` loads named sparse vectors.
- [ ] A sparse query against the collection returns candidates ranked by SPLADE dot-product (asserted
      against a real local Qdrant in a test gated by `LATENCE_QDRANT_URL`, plus a `FakeQdrantClient`
      path that faithfully models named sparse vectors and rejects what real Qdrant rejects).
- [ ] Dense-only load path unchanged / byte-identical (regression test).
- [ ] Idempotent re-load; purge removes sparse (regression test).
- [ ] `ruff` + `mypy --strict` clean; `py.typed`; heavy `qdrant-client` stays in the `[client]` extra.

## Notes
- License: the sparse vectors here derive from `naver/splade-v3` (CC-BY-NC-SA-4.0). This ticket is
  model-agnostic — it wires the *mechanism*; the operator names the checkpoint and accepts its license
  (tutorial Appendix A). Do not pin splade-v3 anywhere as a default.
- This is pre-identified in the tutorial as an expected first finding; may be pre-built before the
  dogfood loop, or fixed on first encounter at §4.3.
