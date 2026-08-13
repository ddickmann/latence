# Findings — wayfinder-enterprise-serving (out-of-scope, logged not chased)

Per the campaign pattern: things noticed during ticket work that are real but out of the
ticket's scope. Each entry names where it lives and what would close it. Nothing here blocks
the enterprise-readiness verdict; everything here was deliberately NOT fixed in passing.

## F1 — Graph sources re-read `graph-edges.parquet` per call

`NetworkxGraphSource` / the DuckDB graph sources in `latence-retrieval` open and read the
edges parquet on every `search` call; the S5 harness bypasses them with a run-lifetime source.
Out of ticket 01's profile scope (the harness never pays it), but a production caller using the
shipped sources directly pays a per-query file read. Candidate for the architecture pass —
the same file-identity-keyed cache discipline as `HypergraphIndex` (ADR-0056 amendment) applies.
(Found: ticket 01. Recorded there as a residual; carried here so it survives the ticket close.)

## F2 — `test_goldset_property.py` dense-cluster stress costs ~288 s CPU

`packages/latence-core/tests/test_goldset_property.py` is green but its dense-cluster stress
test alone burns ~5 min of CPU, which is why full-suite wall time is dominated by a single
file. Flagged in ticket 03's CI leg; a complexity fix (smaller cluster budget with the same
adversarial property) would cut suite time materially. Out of scope here — a test-performance
change should not ride a readiness pass.

## F3 — `wayfinder-e2e-sota-run` ticket statuses lag their stages

Tickets s1–s3 still read `Status: claimed` although their campaign stages completed
(S1/S2 approved in STATE.json, S3 completed 2026-08-06 per its metrics.json). s5/s5b are
genuinely in flight (the pod rescue sweep). Left untouched: those statuses belong to that
campaign's approve-flow, and flipping them from a different campaign would falsify who
closed them.

## F4 — local `benchmark/sota-campaign/STATE.json` is a stale snapshot

The tracked STATE.json in this working tree shows s3 `blocked` (a 2026-08-05 pod incident that
was subsequently resumed and completed) and sits modified-uncommitted, touched by parallel
campaign processes. The pod's own state is authoritative while the sweep runs. Left exactly as
found (the parallel-session discipline: never sweep another session's dirty files); reconcile
after the sweep lands.

## F5 — stale branch inventory (for Dennis — nothing deleted)

31 non-archive local branches besides `hypergraph-m1`/`main`, most long-merged or superseded
(`oss-*`, `pr-*`, `lane-*`, `docs*`, `_pod*`, `sota-final`, …), plus `archive/*` tags carrying
the same names. `git branch --merged main` is the safe triage start. Per the operating
contract, no branch was deleted or will be without an explicit go.

## F6 — `chunk-validation/` untracked WIP carries 2 lint errors

`benchmark/sota-campaign/stages/s3-pipeline/chunk-validation/{chunk_stats.py,
induction_sample.py}` are another session's in-flight, untracked scripts (2 ruff errors in
chunk_stats.py). Deliberately neither fixed, ignored, nor committed here — the owning session
decides whether they become evidence or disappear.
