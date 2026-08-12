# Ticket-03 equivalence proof — incremental churn == full rebuild (ADR-0061 §8)

Real-data gate for incremental ingest, run entirely on this Mac (CPU, `PYTHONHASHSEED=0`,
zero LLM calls). Corpus: `benchmark/sota-corpus/multihop_rag` (609 docs, hand-authored schema),
profile: the committed s3 pipeline byte-for-byte except `device: cuda→cpu` + local paths.

```bash
PROOF=benchmark/sota-corpus/ticket03-proof            # untracked local data root
.venv/bin/python issues/.../03-proof/prepare_corpus.py --proof-root $PROOF --phase init
.venv/bin/python issues/.../03-proof/make_pipeline.py --proof-root $PROOF
.venv/bin/python issues/.../03-proof/run_proof.py $PROOF          # ~10 h CPU, resumable
.venv/bin/python issues/.../03-proof/compare.py \
    $PROOF/store-incr/_latence/runs/t03-v4/export \
    $PROOF/store-rebuild/_latence/runs/t03-rebuild/export         # writes results.json
```

* `prepare_corpus.py` — deterministic base(548)/add(61)/update(20)/delete(20) batches
  (`batches.json`), in-place phase mutations of the working input dir.
* `make_pipeline.py` — the three pipeline variants (intermediate deltas skip the dense
  embedder; the two compared runs carry it — embeddings are a pure per-run export
  augmentation, ADR-0061 §2).
* `run_proof.py` — sequential supervisor with per-phase logs, durable `phases.json`, and hard
  churn-witness probes (a delta whose reuse counters deviate FAILS the proof).
* `compare.py` — artifact + ranking comparison; exits non-zero on any unmeasured divergence.

Results land in `results.json` here and are summarized in `../03-report.md`.
