# R4 — C5's headline memory bound does not ship: both Runners hand `build()` a fully materialised whole-corpus sink list

- **Severity:** major
- **Dimension:** memory claims real (ADR-0033), the deepening is real not documentary (C5)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/quality_report_builder.py` —
  `QualityReportBuilder.final_records` (:338–363, `-> list[StageOutput]`, `final.extend(...)`)
- **Production callers:** `packages/latence-core/src/latence_core/runner.py` :1192 (end-of-run
  report) and :1152 (mid-run FAIL report), via the `_final_records` delegator (:2517);
  `packages/latence-runner-airflow/src/latence_runner_airflow/runner.py` :142 (`run`) and :318
  (the DAG's report task)
- **Claim contradicted:** the module docstring's C5 headline — *"Peak memory is `O(aggregators x
  bounded state)` instead of `O(corpus x sections)`"* — restated in `build()`'s own docstring
  (:387).
- **Invariants at stake:** ADR-0033 (streaming / memory-bounded runner: `O(batch)`, no
  whole-corpus materialisation), ADR-0008 (the report's bytes are the contract — the fix must not
  move them)
- **Status:** FIXED (this commit)
- **Not on main:** the C5 fold and the `QualityAggregator` protocol are new in this change set;
  the headline bound they were written to deliver is what did not reach the shipped path.

## What was wrong

C5 rewrote the report's thirteen per-kind gathers as a fold over the lazy `StageRecordReader`
seam, and that half is real. The report's OTHER input is not folded from a Stage subscription: it
is the DAG's **sink** records, which `build()` folds into the contract-completeness check and the
distinct-source-document count (`_fold_final_records`, :433). That fold was written lazy —
`build()`'s own docstring says *"A caller that hands over an iterator therefore never materialises
the sinks either"* — but **no shipped caller hands over an iterator**. Every production call site
gets its argument from `QualityReportBuilder.final_records`, which returned
`list[StageOutput]` built with `final.extend(self._reader.stage_records(...))`.

So the lazy path had zero production callers, and report-time peak allocation was still linear in
the corpus — at the point in the run where every Stage has already been streamed to its checkpoint
and released (#69), i.e. exactly where ADR-0033 promises the executor is holding nothing.

`ADR-0033`'s "NOT bounded today" section listed four Capabilities' *inputs* and never mentioned
report-time sink materialisation, so an operator sizing RAM off that section was not warned.

## Failure scenario (measured)

`tracemalloc` started at the top of `final_records` and read again at the end of `build()`, on a
real `LocalRunner` pipeline (source → parse → chunk; the Chunk Stage is the sink):

| documents | sink records | peak after `final_records` | peak after `build()` |
| --- | --- | --- | --- |
| 10 | 150 | 728 KB | 762 KB |
| 40 | 600 | 2 888 KB | 2 978 KB |
| 160 | 2 400 | 11 527 KB | 11 690 KB |
| 640 | 9 600 | 46 086 KB | 46 278 KB |

Strictly linear, ~4.8 KB per sink record; the fold itself contributes 0.2–0.4%. **~99.6% of
report-time peak was the sink list C5 was supposed to have removed.** Extrapolated to the
1M-document ceiling the ADRs size against, the report phase alone allocates tens of GB, so a run
whose sink is an Export or a Redaction Stage over a large corpus OOMs at report time even though
every Stage stayed `O(checkpoint batch)`.

## What was built

`final_records` yields instead of extending — the consumer of the lazy reader now matches the
producer, which is C5's own principle applied to the one method C5 missed:

```python
for name in sinks:
    yield from self._reader.stage_records(name, outputs, storage=..., pipeline=..., run_id=...)
```

`build()` consumes it exactly once (`_fold_final_records`, :433) and nothing else in the process
reads it, so yielding is sufficient — no seam is added, no interface grows, and no caller changes:
both Runners already pass the value straight through. `LocalRunner._final_records`'s return type
follows (`Iterator[StageOutput]`).

Order is unchanged (sink Stages in topological order, reader order within a Stage), so the
report's bytes are unchanged — ADR-0008.

Deliberately NOT done:

* **Removing the `final_records` parameter from `build()`** (the smaller interface). It looks like
  the stronger fix, but `build()` would then have to stream the sink Stages itself — and a sink
  Stage is usually also a *subscribed* Stage, so the builder would either decode it twice
  (regressing C5's decode-once property, which
  `test_each_stage_stream_is_decoded_exactly_once` pins) or merge the sink fold into the
  subscription pass, a rewrite of the fold's fan-out for no memory gain over yielding.
* **A `list`-rejecting guard on `build()`.** `Iterable` is the honest type for a consumer, tests
  legitimately pass `iter(())`, and a guard is a layer that compensates for the producer instead
  of fixing it.

The honest residual is written into ADR-0033 rather than papered over: the report phase still holds
one document id per distinct source document (that *is* the document count) and the drift
aggregator's `PageIndexResolver` page window — both per-DOCUMENT and bounded, neither per-record.

## Acceptance criteria

1. **AC1 — the seam is lazy.** `final_records` pulls NOTHING from the reader before the report asks
   for a record, and pulling one record costs exactly one record.
2. **AC2 — the report phase does not materialise the sinks.** Peak allocation across
   `final_records` + `build()` is a small fraction of ONE materialised copy of the sink stream.
3. **AC3 — the PRODUCTION call site is fixed, not just the seam.** Measured on
   `LocalRunner._final_records` → `_build_report` (the exact sequence `run()` executes, with the
   empty `outputs` the streaming default leaves), the whole report phase peaks below half of what
   one materialised copy of its sink stream costs. Non-vacuous: the sinks are corpus-sized and the
   completeness check folded every one of them.
4. **AC4 — the report is byte-identical.** The C5 fixture comparison and the H-F1 rebuild-through-
   a-fresh-public-builder byte-equality test stay green, unchanged.
5. **AC5 — no claim is left standing that the code does not hold.** The C5 headline in the module
   docstring names the sink stream, and ADR-0033's memory boundary states the report phase's bound
   and its per-document residual.

## Tests

- `packages/latence-core/tests/test_quality_aggregators.py`
  - `test_final_records_pulls_no_sink_record_until_the_report_asks_for_one` (AC1) — **fails on the
    pre-fix code**: `final_records drained 512 of 512 sink records before the report asked for one
    — the sinks are being materialised, so report-time peak is O(corpus)`
  - `test_the_report_phase_never_materialises_the_sink_records` (AC2) — **fails on the pre-fix
    code**: `the report phase peaked at 13040558 bytes; one materialised copy of the sink stream is
    13019898 bytes — the sinks are being retained`
- `packages/latence-core/tests/test_quality_report_builder.py`
  - `test_the_runners_report_phase_never_materialises_the_sink_corpus` (AC3) — **fails on the
    pre-fix code**: `the whole report phase peaked at 2414296 bytes while ONE materialised copy of
    its sink stream costs 2157728 bytes (360 records): the sinks are being materialised, so
    report-time peak is still O(corpus)` (post-fix: 288 KB vs 2 158 KB, a 7.5x margin)
  - `test_report_is_byte_identical_when_rebuilt_through_a_fresh_public_builder` (AC4, unchanged)
- `packages/latence-core/tests/test_quality_aggregators.py::test_folded_report_is_byte_identical_to_the_pre_c5_gather_fixture`
  (AC4, unchanged)
