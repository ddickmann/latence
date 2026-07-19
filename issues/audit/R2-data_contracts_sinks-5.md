# R2-data_contracts_sinks-5 — [critical] An overlapping `QdrantSink.load` silently erases the whole collection

**Stream:** feat/release-fleet-integration (#183, Qdrant sink)
**File:** `packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py` — `_sweep` (invoked from `load`)
**Status:** FIXED

## What is wrong

The reconciling sweep deletes by a **collection-global** filter:

```python
stale = models.Filter(must_not=[FieldCondition(key=LOAD_ID_KEY, match=MatchValue(value=load_id))])
```

Nothing scopes that filter to the points *this* load actually processed. The sweep's whole
correctness argument is "every point I touched carries my marker, so `must_not(my marker)` is
exactly the stale set" — and that argument silently stops holding the moment anything else stamps
the collection between this load's restamp phase and its sweep.

This directly falsifies the module docstring's containment guarantee (sink.py:39-41): *"the worst
case is a collection with stale-but-present data, never one with silently-erased data."*

## Failure scenario (reproduced against the shipped `FakeQdrantClient`)

Collection `corpus` holds a 3-record corpus.

1. Loader **A** starts `load(export)`. All three digests match, so A restamps every point with its
   marker `mA` (no upsert — that is the documented delta path).
2. Before A reaches its sweep, loader **B** runs a *complete* load of the **same** Export — a cron
   re-load overlapping a manual one, or a retry whose predecessor is still in flight. B also sees
   every digest matching, restamps all three points with `mB`, and sweeps `must_not(mB)`, which
   matches nothing. B returns `3 record(s) read — upserted 0, unchanged 3, deleted 0` and exits 0.
3. A now sweeps `must_not(mA)`. Every point carries `mB`, so **all three are deleted**. A returns
   `3 record(s) read — upserted 0, unchanged 3, deleted 3 stale point(s)` and exits 0.

Surviving points: **0**. Neither run raises, the CLI exits 0 twice, and the serving index is gone.
The convergence check at sink.py:433-443 passes trivially because the collection is empty.

The report is self-contradictory on its face — it deleted 3 points it had just counted as
`unchanged 3` — which is exactly the cheap invariant that was missing.

## What to build

A pre-delete invariant in `_sweep`, evaluated **before** any deletion, in O(1) memory (ADR-0033 —
no whole-corpus id set):

* count the points that carry this load's marker: `count(must(_latence_load_id == my marker))`;
* it must equal the number of records this load processed (`upserted + unchanged`), because every
  one of those was either upserted with the marker or payload-restamped with it;
* if it does not, **abort with a typed `SinkLoadError` and delete nothing**.

Fail-closed is the whole point: an unexplained mismatch leaves stale-but-present data, which is the
guarantee the module documents, instead of erasing a live index.

## Acceptance criteria

1. Two overlapping loads against the same collection never leave it empty; the points the losing
   load declared current survive.
2. The losing load fails **loudly** with a typed `SinkLoadError` (ADR-0034: no silent errors, no
   raw native traceback) naming the concurrent-load cause, and its sweep issues **no** `delete`.
3. All existing convergence behaviour is unchanged: first load, delta re-load, retraction/purge
   sweep, purge-everything (empty Export), unstamped-legacy-point sweep, and `reconcile: false`.
4. A regression test fails on the pre-fix code (collection emptied, no exception) and passes after.
5. `make test` stays green at or above 2985 passed / 28 skipped.
