# Ticket 03 report — Incremental ingest: document churn without full rebuild

Status: **done — gate PASSED** (2026-08-13; every comparison identical, zero divergences,
zero tolerances used)

> Run log: the first full-scale attempt (2026-08-12) was killed mid-base-extraction at 2 720 of
> ~2 939 chunks when its host session ended (the supervisor was session-tied). No divergence —
> a clean interruption; the WAL/checkpoint discipline left the run resumable. Relaunched
> 2026-08-13 08:46 DETACHED (`nohup`), with the final delta (t03-v4) and the full rebuild
> overlapped after the delete mutation (separate stores, read-only shared input, same torch
> thread count ⇒ bitwise CPU numerics). All phases completed 12:13; the detached finalizer ran
> `compare.py` automatically — `EQUIVALENCE: PASS`, exit 0.

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

## Real-data smoke (already complete — the harness self-test)

Before the full run, the identical protocol on a 14-doc multihop_rag subset (full s3 profile,
CPU): base 12 docs → add 2 → delta (witness: 12 reused / 2 extracted) vs batch rebuild of all
14. Result: **all six graph artifacts byte-identical** (nodes/edges/hyperedges parquet, KV
store, TTL, GraphML), corpus row sets identical, **embeddings bitwise identical**, bm25-stats
byte-identical, 200/200 BM25 and dense top-20 rankings identical. The smoke also CAUGHT a real
divergence — the reuse store's lexicographic id-sort put `#chunk-10` before `#chunk-2` in a
reused document's within-doc order — fixed read-side by the natural-order overlay fold
(commit 4249620) and pinned by a >10-chunk document in the CI equivalence test.

## The equivalence proof (the gate)

Corpus: **multihop_rag** — 609 real news articles, the s3 enterprise-SOTA profile byte-for-byte
(chunk.markdown 640/80/8; GLiNER2 fused extraction @0.5, hand-authored schema — ZERO LLM calls;
GLiNER2 redaction with guardrails; exact-surface type consolidation + disambiguation;
graph.canonical + weak hyperedges (max_arity 12) + degree features; context.kg_header;
knowledge-graph export with hyperedge KV + jsonl/parquet corpus export with granite-311m
embeddings and BM25) — only `device: cuda→cpu` and local paths changed
(generated by the ticket's own pipeline-builder script, which is not published).

Protocol (the ticket's proof harness, all phases driven through the real CLI, `PYTHONHASHSEED=0`,
CPU):

1. base: 548/609 docs (90%), `latence delta` bootstrap
2. add: +61 docs, delta — churn witness gate: 548 reused / 61 extracted
3. update: 20 docs edited in place, delta — 589 reused / 20 extracted / 20 old ids DELETED
4. remove: 20 docs deleted, delta — 589 reused / 0 extracted / 20 DELETED (final state, with
   embeddings)
5. rebuild: full batch `latence run` of the same final 589-doc corpus, fresh store

Every churn-witness probe passed exactly: add = 548 reused / 61 extracted; update = 589
reused / 20 extracted / 20 old ids DELETED; remove = 589 reused / 0 extracted / 20 DELETED.
Final corpus: 589 documents, 3 181 chunks, 22 779 graph nodes, 2 429 edges, 18 547 hyperedges.

Comparison of the final incremental export (t03-v4) vs the full rebuild (t03-rebuild)
(`03-proof/compare.py` → `03-proof/results.json`) — **all diffs zero**:

| Artifact | Result |
| --- | --- |
| corpus.jsonl (3 181 rows incl. embeddings) | **byte-identical file (sha256 equal — rows AND row order)**; multiset diff 0/0 |
| corpus.parquet row sets + embedding vectors | 3 181 = 3 181, only-in-either 0, changed 0 |
| graph-nodes / graph-edges / graph-hyperedges parquet | 22 779 / 2 429 / 18 547 rows each side, only-in-either 0, changed 0 |
| graph-hyperedges.kv / graph.ttl / graph.graphml | **sha256-identical** |
| bm25-stats.json | **sha256-identical** (idf/avgdl/df exactly equal) |
| bm25-postings.parquet | 3 181 = 3 181, only-in-either 0, changed 0 |
| Dense embeddings (granite-311m, 768-d) | **bitwise identical** (`max_abs_diff = 0.0`) |
| BM25 rankings, top-20, 200 real MultiHopRAG queries | **200/200 identical lists** |
| Dense cosine rankings, top-20, 200 queries | **200/200 identical** (provable from bitwise-equal vectors; asserted) |

No deliberate divergence exists; the documented tolerance mechanism was never needed.

Determinism controls measured up front (M1 Pro, CPU): granite-311m encode is bitwise
reproducible run-to-run for a fixed batch composition; batch composition SHIFTS vectors by up to
1.5e-3 (padding) — which is exactly why the order-preserving fold matters and why the proof can
demand bitwise equality rather than a tolerance. GLiNER2 extraction is bucket- and
batch-invariant by provider contract (asserted in its own tests).

## Measured costs (CPU, M1 Pro; the pod GPU ran this corpus's full pipeline in 0.19 h)

Per-phase wall times (supervisor log; delete + rebuild ran concurrently):

| Phase | Docs extracted | Wall |
| --- | --- | --- |
| base build (548 docs, no embedder) | 548 | 97.3 min |
| **add +61 docs** | 61 | **8.4 min** |
| **update 20 docs** | 20 | **3.3 min** |
| **remove 20 docs** (incl. full-corpus embedding + BM25 export) | 0 | **19.5 min** |
| full rebuild (589 docs, embedder on) | 589 | 97.4 min |

The O(changed) claim, measured: an add of 61 documents costs 8.4 min vs 97.4 min for the
rebuild (**11.6×**); an update of 20 costs 3.3 min (**29×**); even the removal churn that pays
the full exact-recompute export with embeddings costs 19.5 min (**5×**). Single-probe rates:
GLiNER2 extraction ≈ 1.15 s/chunk; granite-311m embedding ≈ 3.5 s/chunk isolated (in-run
length-sorted batching brings the full-corpus embedding pass to ~15 min for 3 181 chunks).

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

* **Equivalence: PASS** (`compare.py` exit 0) — every artifact identical, both ranking legs
  200/200, embeddings bitwise, no tolerance used. Full data in `03-proof/results.json`;
  runs + logs under `benchmark/sota-corpus/ticket03-proof/` (local, untracked).
* CI: the whole `packages/latence-core/tests` suite is green on the final tree + the
  roster-complete venv, verified in two legs with unmasked exit codes (this Mac, heavy provider
  packages installed, `PYTHONHASHSEED=0`): (1) everything except `test_goldset_property.py` —
  every collected test ran to `[100%]`, **0 failed, 0 errors**, 10 skips; the *process* then
  segfaulted in native-library teardown (torch+spacy+pyarrow atexit on macOS) AFTER the test
  phase — the last four files re-run standalone exit 0 clean; (2) `test_goldset_property.py`
  standalone: **17 passed**, exit 0 (its dense-cluster stress test costs 288 s of CPU alone —
  flagged as a follow-up complexity fix, out of ticket scope). `ruff check` clean;
  `mypy --strict --explicit-package-bases` clean on every touched file.
* Collateral fixes that rode along: the five front-door pages' ADR count moved 59→60
  (ADR-0061 joined the tree — the truthfulness test caught it); the two `test_perf_seam`
  torch-free-fallback tests now PIN torch absence with a `sys.modules` monkeypatch (they
  asserted the host venv, not the contract); the local venv was completed to the declared
  provider roster (`latence-embedder-st`, `latence-linkpred-ultra`, `latence-pii-presidio`,
  `latence-splade` installed — the conformance roster tests were failing here since before this
  ticket for that reason).
* Commits: 6db0508 (ADR-0061 + implementation + CLI + docs + tests), 4249620 (natural-order
  overlay fold after the smoke caught the id-sort divergence), 89d3cc1 (detached supervisor +
  concurrent finals), plus this report finalization.
