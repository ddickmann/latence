# Streaming, memory-bounded local Runner (partially accepted — doc-level stream both ways, EXCEPT the Capabilities whose Provider materialises)

**Status: partially accepted.** The **IO-transient** half landed first (G1-RESULTS Lane 6 / issue
#63, completed by #69, input-side relation transient bounded by #68). The **doc-level output
materialisation** — the `return list(provider…)` in every `_dispatch_stage` branch that this ADR
flagged as the remaining unbounded transient — was removed by **C6**, which lands on top of the
**C3** Stage-execution seam: `_dispatch_stage` returns the Provider's `Iterator` uncollapsed and
the Runner writes it record-by-record into the checkpoint, so a doc-level Stage's records never
exist as a list. `outputs` retains nothing at all by default; the finalized checkpoint is the
source of truth (ADR-0017) and every reader streams it back. The matching **doc-level input**
materialisation — `_gather_inputs` concatenating every parent's records into the list a Stage held
for its whole execution — was removed by **audit R2** for the Capabilities whose Protocol allows
it (`_STREAMED_INPUT_KINDS`: Parse, Chunk, Schema Induction, Entity Extraction, Fused, Redaction).
It is **still whole-corpus for the rest** — Screening, Relation Extraction, Context Enrichment,
Export — because their Provider materialises the input anyway. See **Honest memory boundary**
below for the per-transient status; that residue is why this ADR is *partially*, not fully,
accepted, and why the `batch_docs` scheduling loop (issue #110) is still the mechanism that would
close it.

> **C6 + R2 update.** The H-E1 Part 4 correction below is preserved for the record. Its
> `O(corpus-slice)` per-doc-level-stage finding was *half* addressed by C6 (the output side) and,
> for six of the ten doc-level Capabilities, the other half by audit R2 (the input side). What
> remains true from it, and what an intervening rewrite of this section violated, is the
> discipline — **do not claim a bound the code does not hold**, and do not retire the issue that
> tracks the missing bound on the strength of the claim.
>
> **H-E1 Part 4 correction (audit THEME E).** This ADR previously read "built" with a
> `batch_docs`-default-256 doc-level batching loop as a landed mechanism. That was an overclaim:
> only the Export and the checkpoint serialization are streamed/bounded; the doc-level stage chain
> materialises `O(corpus-slice)` per stage. A reader who sized RAM for `O(batch × depth)` doc-level
> memory (per the old claim) would OOM. The Decision below keeps the *design* (it is still the
> intended end state) but the doc-level batching is now explicitly **Deferred (issue #110)**, to
> land after the THEME-F Runner refactor so it isn't written twice.

## Problem
The local Runner is a whole-corpus-in-memory DAG executor: `runner.py` holds `outputs: dict[str, list[StageOutput]]` (every stage's full record list simultaneously), each stage is materialized `list(provider.…)`, and `stages/export.py` does `pa.Table.from_pylist(flat)` (the entire corpus as one Python list + Arrow table before write). G1 measured peak RSS ≈ **62 MB per 1k docs, linear**; ceiling = `RAM / 62MB-per-1k`; a low-cardinality corpus drove records **quadratic** in docs (~0.084·n²; 5k docs → ~12 GB). No backpressure.

## Decision (two stage classes + incremental IO)
The DAG splits into two kinds of stage, routed by CapabilityKind:

1. **Streaming transforms** (doc-level: Source, Screening, Parse, Chunk, Entity, Relation, Redaction). These are per-document and independent, so the Runner *should* push the corpus through them in **bounded batches** (configurable `batch_docs`, default e.g. 256) instead of whole lists: a batch flows the full doc-level chain, its outputs are appended to checkpoints, then it is released before the next batch. Provider signatures are already `Iterable`-in/`Iterable`-out, so provider code barely changes — the change is in the Runner's execution loop. **STATUS: the `batch_docs` *scheduling* loop is still DEFERRED (issue #110); the per-Stage materialisation it was wanted for is gone for six of the ten doc-level Capabilities.** What is built: the *release* of a doc-level Stage's retained output once its consumers have run (#69), the streamed **output** dispatch (C6) and the streamed **input** for `_STREAMED_INPUT_KINDS` (audit R2) — so those Stages are `O(checkpoint batch)` on both sides without any scheduling change. Screening / Relation / Context Enrichment / Export still materialise their input by Provider Protocol, and only `batch_docs` bounds those.
2. **Accumulators** (corpus-level: Profiling, Disambiguation, Graph Assembly). These inherently need the whole corpus, but their **output is bounded by the knowledge-graph size, not the corpus size** — canonical entities, KG nodes/edges, corpus features — which is `O(entities + edges) << O(all mentions)`. The design formalizes the contract that an accumulator consumes a stream and holds only bounded aggregate/canonical state (Profiling already spills features; Disambiguation/Graph already emit canonical records, not raw mentions). They receive the doc-level output as a stream (read back from checkpoints batch-by-batch), not a materialized list.

**Incremental Export.** Replace `from_pylist(whole)` with a `pq.ParquetWriter` writing **row-group batches** + JSONL append. This kills the single biggest transient (peak RSS currently coincides with Export).

**Incremental, batch-granular checkpoints.** `_write_checkpoint(..., list(own_output))` becomes append-per-batch (each batch written temp→rename, preserving WAL atomicity); resume reads completed batches and skips them. Resume semantics stay exact at batch granularity.

**Quadratic-blowup guard.** Independent of streaming: cap relation / co-occurrence fan-out per document (a configurable cap with an audited "capped" marker), bounding the `O(n²)` a low-cardinality corpus produces. (#33 already capped profiling pair-state; this extends the discipline to relation extraction / graph edges.)

**Determinism preserved.** Batches flow in stable input order; accumulators still sort their bounded output before emit. Identical input+config ⇒ identical Corpus Version fingerprint (the G1 determinism bar) must hold — a required test.

## Honest memory boundary (what is and isn't bounded today)

This is the section a reader must size RAM against. Corrected in H-E1 Part 4 to state the **real,
per-transient** boundary, not the aspirational end state.

**Bounded today (built):**
* **Export.** `JsonlParquetExport` streams the corpus in `_EXPORT_BATCH_ROWS` batches (one Parquet
  row group + one JSONL flush per batch); its peak transient is `O(batch)`, not `O(corpus)`. This
  was the single biggest transient G1 measured (peak RSS coincided with `from_pylist(whole)`).
* **Checkpoint serialization.** `_write_checkpoint` serializes a Stage's output in
  `_CHECKPOINT_BATCH_RECORDS` batches appended to a temp file; the *write* transient is `O(batch)`,
  not the old whole-string build.
* **Retained doc-level `outputs` set (#69).** A doc-level Stage's output is *released* from the
  executor's `outputs` dict once its consumers have run, so the *simultaneously-retained* live set
  is bounded by the DAG fan-in (+ accumulator KG state), not by `O(all doc-level records)`.
* **Relation fan-out (#63/#68).** Per-document output and input candidate transients are capped,
  bounding the `~0.084·n²` low-cardinality blow-up.

* **The doc-level stage OUTPUT materialisation (C6).** `_dispatch_stage` returns
  `Iterator[StageOutput]` — the Provider's own iterator, uncollapsed — and `_run` feeds it
  straight into `_write_checkpoint`. A doc-level Stage's records are therefore serialized in
  `_CHECKPOINT_BATCH_RECORDS` batches as the Provider produces them and are never a list. Nothing
  is retained in the executor's `outputs` dict either (the `_release_doc_level_outputs` knob
  flips back to the pre-#69 retain-everything path for a byte-for-byte differential check). The
  per-doc-level-stage OUTPUT transient is `O(checkpoint batch)` + whatever the Provider holds.
  * **The two exceptions, and why they are inherent.** An **Export** must hold the records it
    materialises (it writes a file, and streams it internally in `_EXPORT_BATCH_ROWS` row groups).
    A **fused** or **Type-Consolidation** Stage returns two carriers from one Provider call, which
    the Provider already held. Neither is a Runner-side buffer.
  * **The drain ordering this required.** Provider teardown (endpoint-retry / induction / fan-out
    tally drain, `del provider`, `empty_cuda_cache`) used to fire when the dispatch *returned*.
    Under lazy iteration that is *before the Provider has produced anything* — every tally would
    read zero and the model would be freed mid-generation. The teardown therefore moved into the
    producing generator's own `finally`, inside `StageExecution` (C3), which owns the Provider's
    whole lifetime. This is the load-bearing detail of C6: streaming without it is silently wrong.

* **The doc-level stage INPUT materialisation — for six of the ten Capabilities (audit R2).**
  C6 bounded only the output side; the input side was still `_gather_inputs` concatenating every
  parent's records into one list that lived in `ctx.incoming` for the Stage's entire execution,
  with `_as_type` building a second list of the same size and (under `extract_on: masked`)
  `_extract_input_chunks` a third of fresh `model_copy` objects. Measured on the shipped code:
  an Entity Stage's `ctx.incoming` held 48 records at 4 documents and 192 at 16 — exactly linear.
  For the Capabilities in `runner._STREAMED_INPUT_KINDS` — **Parse, Chunk, Schema Induction,
  Entity Extraction, Fused, Redaction** — the Runner now hands the Stage a lazy stream over the
  parents' checkpoints (`_stream_doc_inputs` + `_as_type_stream` +
  `_extract_input_chunks_streamed`); pulling the *k*-th input record has read exactly *k* records
  back. Both sides of those Stages are `O(checkpoint batch)`. `records_in` stays exact without a
  buffer: `_streamed_input_count` counts the checkpoint's lines (one record per line by
  construction of `_write_checkpoint`) rather than decoding the corpus a second time.
  * **The page-map state those Stages carry across the stream (audit R3).** `O(checkpoint batch)`
    on both sides is a claim about the RECORDS; it was not true of the state a Stage derives from
    them. `PageIndexResolver` — one per Stage invocation, and the invocation is handed the whole
    corpus — memoised a `PageMap` + `PageOffsetIndex` per `document_record_id` and evicted
    neither, so a doc-level Stage retained ~284 KB per 519-page document for its whole execution
    (~284 MB at 1k documents, ~2.8 GB at 10k) inside the bound this section states. It is now an
    LRU window with a budget of `DEFAULT_MAX_LIVE_PAGES` (50,000) **page spans** — a ~27 MB
    ceiling that does not move with the corpus. An absorbed map is reachable only from its
    index, so eviction frees the map; the most recently used index is never evicted (a document
    whose map alone exceeds the budget still resolves); the evicted index's drift is folded into
    the corpus roll-up first (the drift section's numbers are unchanged, ADR-0008); and a lookup
    for an already-evicted document is counted and warned, counts-only, rather than degrading
    silently (`PageIndexResolver.evicted_reentries`, ADR-0034).
  * **Why the budget is pages and not documents (audit R4).** R3 first stated the window as 64
    DOCUMENTS, justified by "every shipped batching Provider uses `batch_size` 16, so 64 is 4×
    the widest in-flight set". Both halves were false — `latence_extract_gliner2` clamps
    `batch_size` to 64 and the PII Providers clamp it not at all, and `batch_size` does not bound
    the DISTINCT DOCUMENTS observed before a batch resolves, because a Provider must `observe()`
    every chunk including the ones its `min_chars` floor drops. A corpus of one-page memos
    therefore streamed hundreds of documents through the window while one batch of long-document
    chunks accumulated and evicted the long document that batch was still resolving; its later
    chunks carry no map of their own (schema v18), so their mentions inherited the chunk's page
    range — a plausible-looking WRONG page, i.e. exactly the ADR-0021/0031 resolution regression
    the C1 seam exists to prevent, at an in-range operator config. A page span is what the
    retained state is made of (~550 bytes), so pricing a slot by pages makes the bound a real
    memory bound instead of a proxy, and makes the short documents that dominate a real stream
    cost ~1/500th of what a long one does. Measured on the shipped wiring in
    `packages/latence-extract-gliner2/tests/test_gliner2_page_window.py`.
  * **The one property traded for the bound.** Those branches' carrier check is now per-record,
    so a mis-wired Pipeline raises the same typed `ContractError` at the first record of the
    wrong carrier instead of before the Provider is called. Same accept/reject set (`_as_type`
    already rejected an input if *any* record was wrong), and still strictly before the Stage's
    checkpoint is published — no torn commit, no silent acceptance (ADR-0034).

* **The end-of-run report phase (C5 + audit R4).** Assembling the Quality Report happens *after*
  every Stage has been streamed and released, and it re-reads the corpus from the checkpoints. C5
  made the thirteen per-kind gathers a fold (`QualityAggregator`), but the report's other input —
  the DAG's SINK records, which the completeness check and the document count are folded from —
  was still handed to `build()` as one materialised whole-corpus list, because
  `QualityReportBuilder.final_records` did `final.extend(reader.stage_records(...))` and both
  Runners call it at their report call site. Measured on a real `LocalRunner` run, report-time peak
  was strictly linear at ~4.8 KB per sink record while the fold itself added 0.2–0.4%: ~99.6% of
  the peak was the list C5 was supposed to have removed, and at the 1M-document ceiling this
  section is sized against the report phase alone would allocate tens of GB — an Export- or
  Redaction-sink run would OOM at report time with every Stage still `O(checkpoint batch)`.
  `final_records` now yields from the reader instead (`build()` consumes it exactly once, in
  `_fold_final_records`, and nothing else reads it), so the report phase holds no per-record
  whole-corpus copy. What it does still hold is per-DOCUMENT and bounded: one document id per
  distinct source document (that IS the document count) and the drift aggregator's
  `PageIndexResolver` window above. Record order — and therefore the report's bytes (ADR-0008) —
  is unchanged.

**NOT bounded today:**
* **The doc-level INPUT of the remaining four Capabilities.** **Intake/Content Screening**
  (`screen_*` returns a `ScreeningOutcome` holding passed + quarantined lists, and Content
  Screening's page-map re-homing needs the input set to know what was dropped), **Relation
  Extraction** and **Context Enrichment** (one input stream partitioned into two carriers — one
  side must be buffered whatever the Runner does) and **Export** (it writes a file) still receive
  a gathered `O(corpus-slice)` list. Streaming the Runner's half would move no memory, because the
  Provider materialises the same records anyway — so this is a **Provider-Protocol** bound to win,
  or a scheduling one (#110), not a dispatch one. A corpus that OOMs a Content-Screening or a
  Relation Stage still OOMs.
* **Peak RSS at 100k / 1M documents.** Still needs a dedicated perf rig (#63); every bound above is
  validated at reduced scale in-suite.

**Corpus-level (accumulators) — the audit this ADR required, now actually performed.** The
"Blast radius / risk" section asked for Disambiguation/Graph to be audited "for any hidden
full-corpus list". They were, and the ADR's original claim was **too strong**:

* **Profiling** and **Graph Assembly** hold bounded aggregate / canonical KG state —
  `O(entities + edges) << O(all mentions)` — as claimed.
* **Disambiguation** (`CascadeDisambiguator.disambiguate`) holds `list(mentions)` **plus** a
  `by_id` dict over the same records. Its retained state is `O(mentions)`, not `O(entities)`.
* **Type Consolidation** (`CascadeTypeConsolidator.consolidate`) holds `list(mentions)` and
  `list(relations)`, and re-emits both 1:1 relabelled. Its retained state is `O(mentions +
  relations)`.

Neither is a defect: a canonical entity cannot be elected from a prefix of the mentions, nor a
canonical type vocabulary from a prefix of the labels. The defect was the *claim*. What every
accumulator DOES hold to is that it consumes each input stream **exactly once** — the property
the Runner's streamed read-back depends on, since `_stream_inputs_of_type` hands each carrier a
fresh one-shot generator and a second pass would silently read an empty corpus.

Both facts are now pinned by `test_accumulator_memory_bound.py` rather than asserted in prose, so
the doc and the code cannot drift apart again. For truly massive corpora the mention set (and the
KG) would need sharding / external accumulator state — a separate effort, out of this ADR.

**Net:** IO transients (Export + checkpoint write) and the retained live set are bounded; a
doc-level Stage's transient is bounded on **both** sides for the six Capabilities in
`_STREAMED_INPUT_KINDS` and on the **output side only** for Screening / Relation / Context
Enrichment / Export, whose input is still `O(corpus-slice)` because their Provider materialises it;
corpus-level memory is bounded by the KG for Profiling/Graph and by the **mention set** for
Disambiguation/Type-Consolidation. The real 100k/1M peak-RSS + no-OOM-at-ceiling numbers still need
a dedicated perf rig (#63).

**Scope of the in-suite memory test.** `test_export_memory_scaling.py` bounds **Python-object
allocations** via `tracemalloc` (the streamed Export's peak stays sub-linear as the corpus grows
4×). It does **not** measure Arrow/Parquet **off-heap RSS** (pyarrow buffers, OS page cache) — that
is the perf rig's job (#63). It also validates only the Export transient. The doc-level-stage
transient is pinned by shape rather than by allocation: `test_runner_executor_streaming.py` for
the output side (nothing retained) and `test_doc_level_input_streaming.py` for the input side (the
Stage never receives a materialised list, and pulling the *k*-th input record has read exactly *k*
records back from the parent's checkpoint). The four Capabilities whose input is still gathered
have no bound to test — see **NOT bounded today**.

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
* **Input-side mention cap (#68).** The emitted-fan-out cap above bounds the *output*, but
  `_find_candidates` still materialised the full `O(mentions²)` candidate list on the **input**
  side before the sort trimmed it — unbounded when a large operator-configured `window_chars`
  defeats the inner-loop break (operator-config-gated, the same trust tier as the output cap).
  Mirroring #33's input-side discipline exactly (profiling caps the *distinct entities* that enter
  its `O(n²)` pairing), `PatternRelationExtractor` now caps the *mentions per document* that enter
  the candidate scan (`max_mentions_per_document`, default 5_000), so the transient candidate list
  is `O(cap²)` regardless of `window_chars`. The cap keeps the lowest-ordered document-order prefix
  the scan already relies on, so covering spans / page provenance / Evidence offsets of the
  retained relations are unchanged (a stable, deterministic prefix). Documents trimmed and mentions
  dropped are audited into `RelationQuality` (`mention_capped_documents` /
  `mention_dropped_mentions`), bumping `QUALITY_SCHEMA_VERSION` 14 → 15 (additive; a v14 report
  still validates). The Runner threads the input-cap tally alongside the output-cap tally through
  the same `_execute_stage` → `_fanout_total` → `_relation_quality` seam into the persisted report,
  end-to-end regression-tested (`test_mention_cap_audit_reaches_quality_report_end_to_end`).
* **Reduced-scale memory-scaling validation** — a `tracemalloc` test proves the Export's peak is
  bounded (not corpus-linear) when the corpus is fed as a generator; the determinism (byte-stable
  checkpoints), provenance/offset, Evidence, and delta/Purge-read guarantees are pinned by
  regression tests.
* **Streamed executor outputs — doc-level records released before accumulators (#69).** The DAG
  executor no longer holds every Stage's doc-level output in the `outputs` dict for the whole run.
  A doc-level Stage's output is **released** from the dict the moment its last consumer has run
  (a reference-count over `depends_on`), and a doc-level *sink* (an Export, which passes its
  input through unchanged) is released the moment it commits. The corpus-level accumulators
  (Profiling / Disambiguation / Graph Assembly) then **stream** their doc-level input back from
  the parents' checkpoints one record at a time (`_stream_checkpoint` / `_stream_inputs_of_type`),
  and the Quality Report reads any released doc-level set back from its checkpoint per kind
  (`_records_of_kind`), so the whole doc-level output is never materialised into a retained list.
  Corpus-level memory is thereby bounded by the **live DAG fan-in + the accumulators' KG state**
  (canonical entities + nodes/edges + corpus features), not by O(all doc-level records) — the
  ADR's honest boundary, now actually reached. Determinism/resume/delta/Purge are preserved: the
  on-disk checkpoints are byte-identical and re-readable, the streamed read yields the same
  records in the same order as the buffered read, and the release is a pure memory-shaping step
  (a `_release_doc_level_outputs` parity knob keeps the whole `outputs` dict retained for a
  byte-for-byte differential check). A mis-wired accumulator still raises the same typed
  `ContractError` through a streamed presence check. Regression tests
  (`test_runner_executor_streaming.py`) pin the release, the checkpoint-streamed accumulator input
  (the wired path), the peak-retained bound vs. retain-all, and report/KG byte-identity.

**Built (C3 + C6 + audit R2, after the Runner refactor):**
* **Streaming dispatch through the Stage-execution seam.** `_dispatch_stage` returns
  `Iterator[StageOutput]`; `_run` is `resolve inputs -> execute -> persist -> release`, with the
  Provider's stream written batch-granularly into the checkpoint. The `O(corpus-slice)` doc-level
  **output** materialisation is gone, and the Provider teardown moved after consumption so the
  tallies and the VRAM lifecycle stay correct under lazy iteration (see **Honest memory
  boundary**).
* **Streaming the doc-level INPUT (audit R2).** `StageContext.incoming` is an `Iterable`, not a
  list: for `_STREAMED_INPUT_KINDS` the Runner passes a lazy stream over the parents' checkpoints
  and the narrowing/masking stay lazy too, so neither side of those Stages is ever corpus-sized.
  The interface got *smaller* (a consumer may iterate `incoming` at most once) rather than gaining
  a layer; a Runner adapter on a materialised substrate (Airflow XCom) still passes a list. The
  now-unreachable `_require_redaction_inputs` guard folded into the streamed narrowing's message,
  which is the first time its W16 advice can actually reach an operator.
* **The accumulator audit** the "Blast radius / risk" section required — performed, the ADR's
  over-claim corrected, and both the `O(mentions)` bound and the single-pass contract pinned by
  `test_accumulator_memory_bound.py`.

**Remaining (explicitly out of this issue, still flagged):**
* **The `batch_docs` doc-level streaming loop (issue #110) — still NOT BUILT, and still the only
  mechanism that bounds the residue.** C6 + R2 bound both sides of a doc-level Stage *whose
  Provider consumes a stream*, without a scheduling change — so #110 is no longer needed for those
  six Capabilities. It is still needed for the other four: Screening, Relation Extraction, Context
  Enrichment and Export materialise their input by Protocol, and only limiting how many documents
  enter the chain at once bounds them. (An earlier revision of this ADR retired #110 outright on
  the strength of a claim the code did not hold; that was the exact mistake the H-E1 correction
  above warns against.) #110 additionally carries the *latency* argument (a batch reaching Export
  before the whole corpus is parsed), which is separate and needs its own justification.
* The real **100k / 1M peak-RSS + no-OOM-at-ceiling** numbers still need a dedicated perf rig
  (the DoD Hardening "no OOM at the documented ceiling" is validated at reduced scale here; the
  headline numbers stay flagged, per the issue).
## Rejected
- **Record-by-record streaming — of PROVIDER CALLS.** This rejection is about *call granularity*,
  not about how the Runner consumes a Stage's output, and the distinction was never stated
  explicitly enough. A Provider still receives an `Iterable` and is free to batch internally (and
  the batching ones do — that is the amortization this bullet protects). What C6 changed is that
  the **Runner** no longer collapses the Provider's returned iterator with `list(...)`; it writes
  it batch-granularly into the checkpoint. Audit R2 changed the mirror image: the Runner no longer
  collapses the Provider's *input* into a list either, for the Capabilities that consume it once.
  Nothing about the Provider's call shape changed — it still receives one `Iterable` and may batch
  internally — and the on-disk output is byte-identical. So neither reopens this rejection.
- A queue/thread backpressure model (the local Runner is single-process; bounded batches ARE the backpressure — a real distributed backpressure story is the Databricks/Airflow adapter's job, ADR-0003).
- Rewriting accumulators to be incremental-across-runs (that's the delta engine's job, #58/#42) — orthogonal.
