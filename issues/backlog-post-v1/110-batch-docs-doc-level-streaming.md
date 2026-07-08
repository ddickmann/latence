# #110 — `batch_docs` doc-level streaming loop (the real memory bound the ADR-0033 doc-level claim promised)

**Status: backlog / deferred.** Filed by H-E1 Part 4 (audit THEME E) when ADR-0033's "built"
doc-level bounded-batching claim was corrected to honest. This is the **real work** the corrected
ADR now defers.

## Why this exists
ADR-0033 ("Streaming, memory-bounded local Runner") decided the doc-level Stages (Source,
Screening, Parse, Chunk, Entity, Relation, Redaction) should push the corpus through the chain in
**bounded batches** (`batch_docs`, default ~256) so doc-level memory is `O(batch × depth)` instead
of `O(corpus-slice)`. That loop was **never built** — `batch_docs` exists nowhere in the code, and
every doc-level Stage still does `return list(provider…)` in `runner.py::_dispatch_stage`,
materialising its whole output for the corpus slice. The ADR previously read "built" with a
`batch_docs`-default-256 mechanism; H-E1 Part 4 corrected that overclaim (see ADR-0033 "Honest
memory boundary"). What IS built today: incremental Export (`_EXPORT_BATCH_ROWS`), batch-granular
checkpoint writes (`_CHECKPOINT_BATCH_RECORDS`), the #69 release of a doc-level Stage's retained
output once its consumers have run (bounds the *simultaneously-retained* set by DAG fan-in), and
the relation fan-out caps (#63/#68). None of those bound a single doc-level Stage's own
`O(corpus-slice)` materialisation.

## The work
Add a `batch_docs` execution loop to the local Runner so the doc-level chain runs in bounded
batches end to end:

* A configurable `batch_docs` (Pipeline- or Runner-level; default e.g. 256). A batch of documents
  flows the **full** doc-level chain (Source → … → Redaction) as an iterator, its per-Stage outputs
  are appended to the batch-granular checkpoints (the format already exists), then the batch is
  released before the next batch is pulled. Provider signatures are already `Iterable`-in /
  `Iterable`-out (ADR-0033), so provider code barely changes — the change is the Runner's execution
  loop.
* The corpus-level **accumulators** (Profiling / Disambiguation / Graph Assembly) already stream
  their doc-level input back from checkpoints one record at a time (#69), so they are unaffected —
  they consume the streamed batches' checkpoints, not a materialised list.

## Sequencing (important)
Land this **after** the THEME-F Runner refactor, not before — the doc-level execution loop is
exactly the code THEME-F reshapes, so writing `batch_docs` first would write it twice. This issue
is deliberately parked until THEME-F is done.

## Definition of Done (Hardening bar)
- [ ] `batch_docs` config threads through the Pipeline/Runner; a doc-level Stage's peak retained
      output is `O(batch)`, not `O(corpus-slice)` — proven by a `tracemalloc` scaling test that
      feeds a large corpus as a generator and asserts sub-linear peak growth of the **doc-level
      chain** (not just Export, which `test_export_memory_scaling.py` already covers).
- [ ] Determinism preserved: identical input+config ⇒ identical Corpus Version fingerprint
      (batches flow in stable input order; accumulators still sort before emit).
- [ ] Resume stays exact at batch granularity: a crash-mid-batch resume produces a **byte-identical**
      export + checkpoints (the H-E1 Part 1 resume-equality discipline, extended to mid-batch).
- [ ] No regression to Provenance / Classification / Evidence / offset guarantees or the
      delta/Purge checkpoint-read path.
- [ ] ADR-0033 flipped from "doc-level batching DEFERRED (#110)" back to "built", with the "Honest
      memory boundary" section updated to the new real bound.
- [ ] The off-heap-RSS caveat: pair the Python-alloc `tracemalloc` assertion with the perf rig's
      RSS number (#63) if #63 has landed, or restate the caveat if not.

## Related
- ADR-0033 (`docs/adr/0033-streaming-memory-bounded-runner.md`) — the decision + the corrected
  honest boundary.
- #63 — the dedicated perf rig for the real 100k/1M peak-RSS numbers (off-heap RSS, out of the
  in-suite `tracemalloc` scope).
- THEME-F Runner refactor — the prerequisite this is sequenced behind.
