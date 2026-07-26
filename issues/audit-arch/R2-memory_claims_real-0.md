# R2 — a CONSUMER-side Stage failure reported `duration_seconds=0.0` and `endpoint_retries=0`

- **Severity:** major
- **Dimension:** memory claims real / NO SILENT ERRORS (ADR-0034)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/stage_execution.py`
  — the `finally` inside `StageOutcome._counted` (the clock) and inside `StageExecution.execute`'s
  `_run` (the Provider tally drain + `del provider` + `empty_cuda_cache`);
  consumed at `packages/latence-core/src/latence_core/runner.py:1046-1069`
  (`outcome.metrics(records_in=..., error=exc)` in the checkpoint-write `except`)
- **Invariants at stake:** ADR-0034 (a failure is REPORTED, typed, and the row that reports it is
  true — a quietly zeroed column is a silent error), ADR-0008 (the Quality Report is the audit
  artifact), ADR-0033 (the streaming shape is what created the asymmetry), plus the cross-Stage
  VRAM lifecycle (`del provider` / `empty_cuda_cache`)
- **Status:** FIXED (this commit)

## What was wrong

`StageOutcome.records` is a lazy generator. A Stage can therefore fail on **either side of the
`yield`**:

* the **Provider** raises → the exception travels out *through* `_run`'s `finally` and
  `_counted`'s `finally`, so by the time the Runner's `except` runs, `_elapsed` is set and the
  Provider's tallies are drained. The FAILED row is honest.
* the **consumer** raises — the checkpoint / WAL / storage write the Runner drives the stream into
  (`runner.py:1046`, `_write_checkpoint` → `_stream_batches`) → the producing generator is still
  **suspended at its `yield`**, kept alive by the traceback. *Neither* `finally` has run. The
  Runner's `except` immediately calls `outcome.metrics(records_in=..., error=exc)`, which reads
  `self._elapsed` (still `None` → `0.0`) and `self._tallies.endpoint_retries` (still `0`), and the
  Provider is never `del`-ed and the CUDA cache never released.

The module documented the opposite, in two places:

> Stops the clock on BOTH the success and the failure path, so a failed Stage's metrics report the
> time it actually burned rather than zero.

> Runs on the failure path too, so a failed Stage still reports the retries it burned.

Both were false for the consumer-side half of the failure space — which is the half that covers a
full disk, an S3 5xx, a WAL append failure, or a serialization error, i.e. the failures a
`storage`-category error taxonomy exists for. The zeros land in three artifacts at once: the
persisted `quality-report.json` FAILED row, the failed `stage_end` span (`duration_ms`) and the
FAIL lineage event (`duration_ms`).

## Failure scenario (reproduced)

A 6-document `parse` Stage whose Provider increments `endpoint_retries` per record and sleeps 5 ms
per record; the checkpoint writer raises `OSError("[Errno 28] No space left on device")` on the 4th
record. The Provider had produced 4 records and burned 4 retries over ~20 ms of real work.

| column | pre-fix (consumer raises) | control: same Provider raising mid-stream | expected |
|---|---|---|---|
| `endpoint_retries` | **0** | 4 | 4 |
| `duration_seconds` | **0.0** | 0.020185 | > 0 |
| Provider stream closed (`del provider`, `empty_cuda_cache`) | **no** | yes | yes |

Pre-fix test output:

```
E  AssertionError: the FAILED row reported the retries as 0 — the Provider burned 4.
   The producing generator was suspended at its yield and its tally drain never ran.
E  assert 0 == 4
```

The pre-existing `test_stage_failures_surface_from_the_stream_with_populated_failure_metrics`
stayed green throughout: it triggers a **provider-side** `ContractError` and asserts only
`error_count` / `error_category`, never the duration or the retries.

## What was built

`StageOutcome` now keeps the handle on the stream it hands out, and `metrics(error=...)` finalizes
an abandoned one before it reads a single counter (`StageOutcome._finalize_stream`). Closing the
generator throws `GeneratorExit` in at the suspended `yield`, which runs both `finally` blocks —
delegated through `yield from` into the Provider's own stream **first**, so the Provider tears down
before the seam drains it, then the clock stops. No new column, no new layer, and no fourth name on
the seam: the outcome already owns the Provider's whole lifetime, so finalizing is its job and not
something a Runner adapter (local, Airflow, or an adopter's) can forget.

A teardown that *itself* raises while closing an already-failing Stage must not replace the Stage's
real exception (that would also skip the Runner's FAIL-report persistence), so it is attached to
that exception with `add_note` — visible, never swallowed silently (ADR-0034), carrying the stable
typed **category** and the exception type name only, never a message that could quote a record.

## Acceptance criteria

- [x] A Stage that fails in the CONSUMER reports the retries it actually burned in its FAILED
      `StageMetrics` row, its `stage_end` span and its FAIL lineage event.
- [x] The same Stage reports a non-zero `duration_seconds` (the time it really spent), matching the
      provider-side control path.
- [x] The Provider's stream is closed on that path, so `del provider` + `empty_cuda_cache` run
      rather than waiting for the garbage collector to reach the traceback frame.
- [x] The primary Stage exception still propagates unchanged and the FAIL Quality Report is still
      persisted; a secondary teardown failure never masks it and is never silent.
- [x] Regression test that FAILS pre-fix:
      `packages/latence-core/tests/test_stage_execution_seam.py::test_a_consumer_side_failure_still_reports_the_time_and_retries_it_burned`
- [x] `make test` green; `ruff check` + `mypy --strict` green on the touched files.
