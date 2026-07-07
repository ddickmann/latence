# Streaming, memory-bounded local Runner (accepted)

**Status: accepted** — built (from G1-RESULTS Lane 6 / issue #63). The two biggest named
transients are now bounded and the quadratic guard is in place; see **Built vs. remaining**
below for exactly which mechanisms landed and which stay flagged for the perf rig.

## Problem
The local Runner is a whole-corpus-in-memory DAG executor: `runner.py` holds `outputs: dict[str, list[StageOutput]]` (every stage's full record list simultaneously), each stage is materialized `list(provider.…)`, and `stages/export.py` does `pa.Table.from_pylist(flat)` (the entire corpus as one Python list + Arrow table before write). G1 measured peak RSS ≈ **62 MB per 1k docs, linear**; ceiling = `RAM / 62MB-per-1k`; a low-cardinality corpus drove records **quadratic** in docs (~0.084·n²; 5k docs → ~12 GB). No backpressure.

## Decision (two stage classes + incremental IO)
The DAG splits into two kinds of stage, routed by CapabilityKind:

1. **Streaming transforms** (doc-level: Source, Screening, Parse, Chunk, Entity, Relation, Redaction). These are per-document and independent, so the Runner pushes the corpus through them in **bounded batches** (configurable `batch_docs`, default e.g. 256) instead of whole lists. A batch flows the full doc-level chain, its outputs are appended to checkpoints, then it is released before the next batch. Provider signatures are already `Iterable`-in/`Iterable`-out, so provider code barely changes — the change is in the Runner's execution loop.
2. **Accumulators** (corpus-level: Profiling, Disambiguation, Graph Assembly). These inherently need the whole corpus, but their **output is bounded by the knowledge-graph size, not the corpus size** — canonical entities, KG nodes/edges, corpus features — which is `O(entities + edges) << O(all mentions)`. The design formalizes the contract that an accumulator consumes a stream and holds only bounded aggregate/canonical state (Profiling already spills features; Disambiguation/Graph already emit canonical records, not raw mentions). They receive the doc-level output as a stream (read back from checkpoints batch-by-batch), not a materialized list.

**Incremental Export.** Replace `from_pylist(whole)` with a `pq.ParquetWriter` writing **row-group batches** + JSONL append. This kills the single biggest transient (peak RSS currently coincides with Export).

**Incremental, batch-granular checkpoints.** `_write_checkpoint(..., list(own_output))` becomes append-per-batch (each batch written temp→rename, preserving WAL atomicity); resume reads completed batches and skips them. Resume semantics stay exact at batch granularity.

**Quadratic-blowup guard.** Independent of streaming: cap relation / co-occurrence fan-out per document (a configurable cap with an audited "capped" marker), bounding the `O(n²)` a low-cardinality corpus produces. (#33 already capped profiling pair-state; this extends the discipline to relation extraction / graph edges.)

**Determinism preserved.** Batches flow in stable input order; accumulators still sort their bounded output before emit. Identical input+config ⇒ identical Corpus Version fingerprint (the G1 determinism bar) must hold — a required test.

## Honest boundary (what this does NOT do)
Memory drops from `O(all records)` to `O(batch × pipeline-depth + KG state)` — a large, real improvement — but **not `O(1)`**: the accumulators' KG state grows with unique entities/edges. For truly massive corpora the KG itself would need sharding / external (on-disk or DB-backed) accumulator state — a bigger, separate effort, explicitly out of this ADR. So: doc-level pipeline becomes memory-bounded; corpus-level memory is bounded by the KG, not the corpus. The real 100k/1M peak-RSS + no-OOM-at-ceiling numbers still need a dedicated perf rig (#63).

## Blast radius / risk
Runner execution loop + checkpoint format (batch-granular) + Export writer change. Providers mostly unchanged (iterable contracts already hold). Risks to guard with tests: checkpoint/resume correctness at batch granularity, determinism (fingerprint stability), no regression to Provenance/Evidence/offset guarantees, and the delta/Purge path (which reads checkpoints) staying correct. Accumulators must be confirmed to hold only bounded state (audit Disambiguation/Graph for any hidden full-corpus list).

## Built vs. remaining (as of #63)

**Built (this issue):**
* **Incremental Export** — `JsonlParquetExport` streams the input in bounded batches
  (`_EXPORT_BATCH_ROWS`): each batch is one JSONL flush + one Parquet **row group** via a
  `pq.ParquetWriter`, written to a temp file and atomically published (new
  `Storage.temp_uri` / `Storage.finalize_atomic`). The whole-corpus `pa.Table.from_pylist(flat)`
  transient — the peak G1 measured — is gone; the Export's peak is now O(batch). Byte-identical
  output (verified). This is the single biggest transient the ADR called out.
* **Batch-granular streamed checkpoint writes** — `LocalRunner._write_checkpoint` serializes a
  Stage's output in bounded batches (`_CHECKPOINT_BATCH_RECORDS`), appending each to a temp file
  (so the write transient is O(batch), not the old whole-string build), logging a `stage_batch`
  WAL event per batch, then publishing temp→rename. The on-disk format is byte-identical
  (newline-delimited `model_dump_json`), so the delta engine's checkpoint reader and the Airflow
  XCom decode are untouched. Resume stays exact at Stage granularity.
* **Quadratic-fan-out guard** — `PatternRelationExtractor` caps relations emitted per document
  (`max_relations_per_document`, default 10_000), applied after the deterministic sort (stable
  prefix ⇒ determinism preserved), with the capped/dropped counts audited into
  `RelationQuality` (`fanout_capped_documents` / `fanout_dropped_relations`). This bounds the
  `~0.084·n²` blow-up a low-cardinality corpus drove into records **and** downstream graph edges
  (extending #33's profiling discipline to relations). The two new `RelationQuality` fields bump
  `QUALITY_SCHEMA_VERSION` 12 → 13 (additive; a v12 report still validates). The Runner seam that
  threads the tally from the Provider (`_execute_stage` → `_fanout_total` → `_relation_quality`)
  into the **persisted** Quality Report is covered by an end-to-end regression test
  (`test_fanout_cap_audit_reaches_quality_report_end_to_end`), not just the Provider-level cap —
  so a broken wiring can no longer silently zero the audited marker.
* **Reduced-scale memory-scaling validation** — a `tracemalloc` test proves the Export's peak is
  bounded (not corpus-linear) when the corpus is fed as a generator; the determinism (byte-stable
  checkpoints), provenance/offset, Evidence, and delta/Purge-read guarantees are pinned by
  regression tests.

**Remaining (explicitly out of this issue, still flagged):**
* The Runner's DAG executor still holds the per-Stage `outputs` dict in memory for the
  corpus-level Quality Report and the accumulator Stages (Profiling / Disambiguation / Graph
  Assembly), which consume the whole doc-level output. Reading those inputs back from checkpoints
  batch-by-batch (so the doc-level `outputs` are released before the accumulators run) is the
  next increment — the doc-level pipeline's own *checkpoint/Export* transients are bounded now,
  but the executor's retained output list is not yet streamed. Corpus-level memory remains
  bounded by the KG state, as the Honest boundary states, not by O(1).
* The real **100k / 1M peak-RSS + no-OOM-at-ceiling** numbers still need a dedicated perf rig
  (the DoD Hardening "no OOM at the documented ceiling" is validated at reduced scale here; the
  headline numbers stay flagged, per the issue).

## Rejected
- Record-by-record streaming (batch amortizes overhead and matches batching providers).
- A queue/thread backpressure model (the local Runner is single-process; bounded batches ARE the backpressure — a real distributed backpressure story is the Databricks/Airflow adapter's job, ADR-0003).
- Rewriting accumulators to be incremental-across-runs (that's the delta engine's job, #58/#42) — orthogonal.
