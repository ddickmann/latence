# H-E1 — Make the resume / S3 / Airflow guarantees real, and ADR-0033 honest

**Source: hostile 360 audit, THEME E. The systemic "assert a boolean, not the property" pattern +
one unbuilt ADR headline.** These are the crash/resume/object-storage paths an enterprise adopter
trusts most, and today they are proven by tests that check a flag or run on the wrong backend.

## Part 1 — Resume is "exact": prove output equality, not a flag
`test_runner_streaming_checkpoint.py::test_resume_exact_after_streamed_write` and
`test_e2e_redaction.py::test_redaction_slice_resumes_from_checkpoints` claim resume is exact but
assert only `all(m.from_checkpoint)` (+ contract validity) — never that the **resumed run's output
equals the first run's**. A resume that produced corrupt/different data passes both (the exact class
of the P3-F1 sticky-resume bug already found).
**Fix:** make each resume test run to completion once, capture the exported artifact bytes (Parquet/
JSONL/TTL as applicable), then resume (same `run_id`) and assert the resumed export is **byte-identical**
(or record-set-identical where byte-order is legitimately nondeterministic — justify which). Audit the
other `from_checkpoint`-only assertions in the suite and upgrade any that claim exactness. Each upgraded
test must FAIL if resume dropped/duplicated/reordered a record (prove it bites).

## Part 2 — S3 is actually exercised (ADR-0009), not simulated on local FS
`test_e2e_s3.py` is named/docstringed as end-to-end S3 but sets `store = file://…` and runs
unconditionally; the CI `storage-s3` MinIO job runs only `test_storage_s3.py` (a 5-byte round-trip).
So "the same pipeline runs on `s3://` with no code change" is untested on object storage.
**Fix:** run a **real** end-to-end pipeline against `s3://` in the MinIO lane. Preferred shape:
parametrize the e2e (or add an e2e variant) on the storage URI, and in the `storage-s3` CI job point
it at the MinIO endpoint (`s3://…` + the test credentials/`endpoint_url` already wired for
`test_storage_s3.py`). It must exercise the real pipeline invariants (intake-quarantine-before-parse,
offset round-trip, sensitivity inheritance, idempotent resume) on object storage. If a full e2e on
MinIO is too heavy, run at least the spine (source→parse→chunk→export) + a resume against `s3://`.
Rename/retire the misleading local-FS `test_e2e_s3.py` label so nothing claims S3 coverage it doesn't have.

## Part 3 — The Airflow DAG wiring is actually tested in CI
`test_airflow_dag_path.py`'s real `build_dag` tests are `@skipif(not _HAS_AIRFLOW)`; CI installs the
package `--no-deps` and never installs `apache-airflow`, so both **always skip** — the package's
raison d'être has zero enforced CI coverage. A revert that mis-wires `report` upstream of a sink or
drops a dependency edge ships green.
**Fix:** add a dedicated CI job (like the MinIO one) that installs `apache-airflow` (the pinned
optional extra) and runs the real `build_dag` DAG-structure tests (one operator per stage, `report`
downstream of sinks, `retries` wiring). Keep it a separate job so the core lanes stay light. Confirm
`_HAS_AIRFLOW` is True there and the DAG tests actually execute (assert collected-count > 0).

## Part 4 — ADR-0033 memory claim: make the doc honest (build streaming is separate)
[ADR-0033](../../docs/adr/0033-streaming-memory-bounded-runner.md) status is "built" and claims doc-level
stages flow in **bounded batches** (`configurable batch_docs`, default 256), but `batch_docs` exists
nowhere and every doc-level stage does `return list(provider…)` — only the Export/checkpoint
serialization transient is bounded. A reader sizing for `O(batch×depth)` doc-level memory will OOM.
**Fix (honesty now, streaming later):** correct ADR-0033 and the memory-scaling docs/tests to state the
**real** boundary — the doc-level chain materializes each stage's output for the corpus slice
(`O(corpus-slice)`); only incremental Export and batch checkpoints are streamed/bounded. Remove the
"built"/`batch_docs`-default-256 claims (or mark that decision **proposed/deferred**). File a **new
backlog issue** for the actual `batch_docs` doc-level streaming loop (real work, deferred — and it
should land *after* the THEME-F Runner refactor so it isn't written twice). Also note the
memory-scaling test's honest scope (it bounds Python allocations via `tracemalloc`, not Arrow/Parquet
off-heap RSS). Do **not** silently leave the "built" claim; do **not** hand-wave a fake streaming loop.

## Definition of Done (Gate bar)
- Resume tests compare **actual output** across resume (byte/record-identical) and FAIL if reverted to
  a corrupt-resume; the `from_checkpoint`-only exactness claims are gone.
- A real `s3://` end-to-end (or spine+resume) runs in the MinIO CI lane and passes; the misleading
  local-FS S3 label is corrected. ADR-0009 "same code on object storage" is now actually exercised.
- An Airflow CI job installs airflow and runs the real `build_dag` tests (collected-count assertion);
  a mis-wired DAG would now fail CI.
- ADR-0033 + memory docs state the true boundary; a new backlog issue tracks the real `batch_docs`
  streaming; no "built"/`batch_docs` overclaim remains. `mypy --strict`/`ruff`/pytest green.

## Reviewer should check
- Each upgraded resume test genuinely bites (would fail on a dropped/duplicated/reordered record) —
  not just a second flag assertion.
- The S3 lane really talks to MinIO over `s3://` (not `file://` in disguise) and runs pipeline stages,
  not a 5-byte round-trip.
- The Airflow job's DAG tests actually execute (`_HAS_AIRFLOW` True, collected > 0) — no silent skip.
- ADR-0033 no longer claims a memory-bounded doc-level Runner that isn't built; the deferral is explicit
  and linked, and the memory-test scope caveat (Python-allocs vs off-heap) is stated.
