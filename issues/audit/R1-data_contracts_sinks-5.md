# R1-data_contracts_sinks-5 — a configured `load_id` silently disables the Qdrant reconciling sweep

**Severity:** critical
**Where:** `packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py:278` (`load_id =
self._configured_load_id or uuid.uuid4().hex`) together with `_sweep` (`must_not` on
`LOAD_ID_KEY`); reachable through `cli.py --load-id` and documented in the package README config
table (`| load_id | random | explicit sweep marker, for an auditable run |`) and the `load_id`
entry of the `QdrantSink` docstring.

## What is wrong

The sink conflates two different things under one config key:

* the **audit label** of a run — what the operator wants when they set `load_id: nightly-load`
  ("make a run auditable/reproducible", per the docstring), and
* the **sweep marker** — the per-run-unique stamp the reconciling sweep needs, because the sweep
  deletes exactly `must_not(_latence_load_id == <this run's marker>)`.

The marker only works as a marker if it differs between runs. Pinning it to a stable configured
value makes the stale filter match the previous run's points as well, i.e. it matches nothing —
the sweep becomes a no-op while still reporting success.

## Failure scenario

1. Config (file or `--load-id`) sets a stable `load_id: nightly-load`.
2. Run 1 loads `[c1, c2, c3]`; every point is stamped `_latence_load_id = "nightly-load"`.
3. `c2` is retracted/purged; the pipeline re-exports the survivors `[c1, c3]` (ADR-0018/0029).
4. Run 2 stamps `c1`/`c3` with the **same** marker `nightly-load`. The stale filter
   `must_not(_latence_load_id == "nightly-load")` matches **zero** points, because `c2` still
   carries `nightly-load` from run 1.
5. The load reports `deleted 0 stale point(s)` as a green result, and the built-in convergence
   check (`after = count(stale)` must be 0) passes **vacuously** — the stale set is empty by
   construction.
6. `QdrantSink(cfg).verify_absent(["c2"]) == ["c2"]`: the purged vector is still present,
   retrievable and searchable. This breaks the erasure guarantee the sweep exists to provide.

The only existing test for the option (`tests/test_qdrant_sink.py:516`) asserts that
`load_id=run-7` appears in the summary string and never performs a second load, so the suite is
green over the defect.

## What to build

Separate the two roles instead of removing the option:

* `load_id` stays the **label** — reported, printed, and carried in the payload stamp so a run is
  still auditable/filterable.
* The value actually stamped and swept on is always **unique per load**: `f"{label}:{nonce}"` when
  a label is configured, and the bare nonce when it is not (so the default path stays
  byte-identical — off-by-default means byte-identical).

## Acceptance criteria

* AC1 — With a stable configured `load_id`, a record dropped from the Export on a re-load is
  deleted, `report.deleted == 1`, and `verify_absent([...]) == []`.
* AC2 — Two loads with the same configured `load_id` stamp two **different** `_latence_load_id`
  values on the points they touch (including restamped *unchanged* points).
* AC3 — The configured label is still recoverable: it prefixes the stamped marker and still shows
  up in `LoadReport.summary()` as `load_id=<label>…`.
* AC4 — With no `load_id` configured, behaviour is unchanged (a bare random hex marker).
* AC5 — README config table and the `load_id` docstring describe the label semantics, not
  "explicit sweep marker".
* AC6 — `make test` stays green at or above 2985 passed / 28 skipped; no test weakened or deleted.
