# Ticket 03 report — Incremental ingest: document churn without full rebuild

Status: draft (proof running — this file is finalized when the real-data gate completes)

## What landed

* **ADR-0061** — `docs/adr/0061-incremental-ingest-exact-corpus-recompute-over-an-order-preserving-reuse-union.md`
  Resolves the ticket's named tensions: the invalidation cascade is the doc-level/corpus-level
  descriptor partition; corpus statistics (BM25 idf/avgdl, profiling, graph features, context
  headers, embeddings) are **exact-recompute every churn, bounded staleness rejected**;
  re-disambiguation scope is full-union in-run (exact) with the committed Version staying
  affected-set-scoped as the audit artifact; graph features refresh every churn.
* **Enrichment-pipeline reuse** (`latence_core/delta_runner.py`, `latence_core/runner.py`) —
  the `_reuse_supported` refusal is gone; the flagship s3 profile (context enrichment + KG +
  exports) now gets O(changed) extraction on churn.
* **Order-preserving overlay fold** (`LocalRunner._merge_overlay_doc_ordered` + the one fold
  rule in `_stream_inputs_of_type` / `_stream_all_records_of_type` /
  `_stream_parent_inputs_of_type` / the Export dispatch) — the reuse overlay is merged
  document-interleaved in source order at the first doc-level producer of each carrier, so every
  corpus-level Stage, the Export row order and the embedder's batch composition see exactly the
  full rebuild's arrival order. This upgrades ADR-0043's equivalence from "same ids/sets" to
  byte-identical streams (display spellings, hyperedge tie-breaks, profiler order-dependences
  included).
* **Manifest tombstone fix** (`DeltaRunner._manifest_of`) — a DELETED entry now shadows the
  ancestor Version chain. Before, an in-place edit's retired content id resurrected from an
  ancestor Version's manifest and was re-classified DELETED on every later delta
  (`documents_deleted` over-counted forever after any edit). Found by the new churn-sequence
  test; edit-then-revert still counts live (newer re-add wins over the older tombstone).
* **CLI**: `latence delta --dry-run` — churn preview (unchanged / new-or-changed / missing by
  content-addressed id) with no Stage run, no model load, no commit; `--verbose` lists ids.
  The operations themselves remain `latence delta` (add/update/remove-by-file-deletion),
  `latence retract`, `latence purge`.
* **Docs**: `docs/guides/incremental-ingest.md` (registered in mkdocs nav under "Operate a
  stack" and in `docs/guides/index.md`; final placement is ticket 04's).
* **Tests**: `packages/latence-core/tests/test_e2e_delta_enrichment_incremental.py` — the
  CI-scale gate: add+update+remove churn through an enrichment+export pipeline is
  **byte-for-byte** equal (export jsonl + graph checkpoint) to a batch rebuild of the same final
  corpus, reuse provably engaged; export no-double-count; induced-schema churn never re-induces
  unchanged documents (lexicon inducer witness). Plus a `--dry-run` CLI test.

## The equivalence proof (the gate)

Corpus: **multihop_rag** — 609 real news articles, the s3 enterprise-SOTA profile byte-for-byte
(chunk.markdown 640/80/8; GLiNER2 fused extraction @0.5, hand-authored schema — ZERO LLM calls;
GLiNER2 redaction with guardrails; exact-surface type consolidation + disambiguation;
graph.canonical + weak hyperedges (max_arity 12) + degree features; context.kg_header;
knowledge-graph export with hyperedge KV + jsonl/parquet corpus export with granite-311m
embeddings and BM25) — only `device: cuda→cpu` and local paths changed
(`issues/wayfinder-enterprise-serving/tickets/03-proof/make_pipeline.py`).

Protocol (`03-proof/run_proof.py`, all phases via the real CLI, `PYTHONHASHSEED=0`, CPU):

1. base: 548/609 docs (90%), `latence delta` bootstrap
2. add: +61 docs, delta — churn witness gate: 548 reused / 61 extracted
3. update: 20 docs edited in place, delta — 589 reused / 20 extracted / 20 old ids DELETED
4. remove: 20 docs deleted, delta — 589 reused / 0 extracted / 20 DELETED (final state, with
   embeddings)
5. rebuild: full batch `latence run` of the same final 589-doc corpus, fresh store

Comparison (`03-proof/compare.py` → `03-proof/results.json`):

| Artifact | Result |
| --- | --- |
| corpus.jsonl (rows incl. embeddings) | PENDING |
| corpus.parquet row sets + embedding vectors | PENDING |
| graph-nodes / graph-edges / graph-hyperedges parquet | PENDING |
| graph-hyperedges.kv / graph.ttl / graph.graphml (sha256) | PENDING |
| bm25-stats.json / bm25-postings.parquet | PENDING |
| BM25 rankings, top-20, 200 real MultiHopRAG queries | PENDING |
| Dense cosine rankings, top-20, 200 queries | PENDING |

Determinism controls measured up front (M1 Pro, CPU): granite-311m encode is bitwise
reproducible run-to-run for a fixed batch composition; batch composition SHIFTS vectors by up to
1.5e-3 (padding) — which is exactly why the order-preserving fold matters and why the proof can
demand bitwise equality rather than a tolerance. GLiNER2 extraction is bucket- and
batch-invariant by provider contract (asserted in its own tests).

## Measured costs (CPU, M1 Pro; the pod GPU ran this corpus's full pipeline in 0.19 h)

* GLiNER2 extraction ≈ 1.15 s/chunk; granite-311m embedding ≈ 3.5 s/chunk (the dominant churn
  cost on CPU — recomputed per export by decision, ADR-0061 §2).
* PENDING: per-phase wall times from the proof logs.

## Deferred, with reasons

* **Campaign-wizard churn block** — deliberately NOT added. The campaign is a batch benchmark
  pipeline; no campaign stage consumes churn configuration, and `latence stack check` explicitly
  flags config keys that go nowhere. The churn surface is the delta CLI; if a future campaign
  stage exercises churn, the wizard block rides that stage (the s3 six-part pattern is
  documented in the campaign wizard).
* **Embedding cache keyed by embedding-input text** — would cut the O(corpus) per-churn
  embedding cost; correct by construction (the input is content + rendered header, both
  deterministic) but carries real invalidation subtleties (headers change when the chunk's KG
  neighborhood changes). Do when measured need arises; ADR-0061 §2 records it.
* **Incremental BM25 df maintenance** — reconstructible from the existing artifact; not worth
  divergence risk for a model-free recompute.
* **`retract`/`purge` survivor reuse** — the erasure paths still re-extract survivors
  (split-on-delete through the real pipeline); deliberate compliance-over-speed, documented.
* **Multi-Source byte-order guarantee** — the order-preserving fold guarantees rebuild byte
  order for single-Source pipelines (every reference stack); interleaving multi-Source URI
  ranges keeps set/ranking equivalence only (ADR-0061 §5, honest limit).

## Gate results

PENDING — filled from `03-proof/results.json` when the runs complete.
