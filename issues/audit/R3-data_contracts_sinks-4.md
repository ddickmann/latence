# R3 · data_contracts_sinks · 4 — the Qdrant sweep guard is check-then-act, and the window erases live records

**Severity:** major
**Where:** `packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py` — `QdrantSink._sweep`
(the `marked = self._count(mine)` guard → `delete(FilterSelector(must_not == my marker))` sequence)
**Invariants at stake:** ADR-0017 (files are the source of truth), ADR-0034 (no silent errors),
ADR-0033 (streaming / O(batch) memory), plus the absolute guarantee the module docstring asserts at
`sink.py:41` and restates at `sink.py:446`.

## What is wrong

The reconciling sweep decides *what is stale* twice, at two different instants, and acts on the
second one:

1. **The guard** counts the points carrying this load's marker and refuses to sweep unless that
   count equals the number of records the load processed. This is a *pre-check*.
2. **The delete** then removes, by filter, `must_not(_latence_load_id == my marker)` — a condition
   the store re-evaluates *at delete time*, not at guard time.

Between (1) and (2) another load's `_restamp` / `_upsert` can land on a point this load just
declared current. The point loses this load's marker, and the delete filter — evaluated *after* the
restamp — matches it. The **winning** load therefore deletes a live record it had itself counted as
`unchanged`, reports exit 0, and its own post-sweep convergence check (`after = self._count(stale)`
= 0) *confirms the corrupted state as converged*.

The existing overlap test only covers a restamp landing **before** the guard (the guard trips, the
losing load refuses). Nothing covers a restamp landing **inside** the guard→delete window, where
the guard passes.

## Failure scenario (observed)

Collection holds `c0, c1, c2` from load 1. Load 2 reads the post-retraction Export (`c0, c1`):

* Load 2's guard passes — 2 points carry its marker, 2 records processed.
* Concurrent load B restamps `c0` with its own marker (a plain `set_payload` over the wire).
* Load 2's `delete(must_not == my marker)` now matches `c0` **and** the genuinely retracted `c2`.
* Result: `points remaining: ['c1']` — the live record `c0` is erased — while load 2 returns exit 0
  and prints a green summary. `verify_absent(['c0'])` then certifies an unretracted record as
  erased.

## What to build

Stop letting the store re-derive "stale" from a marker that a concurrent writer can move. The
marker stays a cheap *candidate* filter; the **authoritative** stale condition becomes the Export
itself — `record_id` is not in the Export this load just read — which is exactly ADR-0017's "files
are the source of truth" and is a condition no concurrent restamp can flip.

* Enumerate the stale candidates (`scroll`, paged at `batch_size`, so memory stays O(batch) per
  ADR-0033) instead of issuing a blind filter-delete.
* Delete **by explicit point id**, and only for candidates whose `record_id` is absent from the
  Export (a point with no `record_id` at all — hand-loaded/legacy — stays stale, as documented).
* Make the convergence proof Export-aware too: a candidate that survives *because it is still in
  the Export* is live, not unconverged; a candidate genuinely absent from the Export that survives
  the delete is still a loud `SinkLoadError`.
* Report `deleted` as the number of points actually deleted, not a pre-count of the filter.
* Keep the pre-guard (it still catches the cheaper, earlier overlap) and keep the co-tenant
  `id_namespace` scoping.

## Acceptance criteria

1. A concurrent restamp landing **inside** the guard→delete window does not delete any record
   present in the loading Export; the collection keeps every live record.
2. That same load still deletes the genuinely retracted record and reports the true `deleted` count.
3. A point whose `record_id` is absent from the Export — including an unstamped legacy point with no
   `record_id` — is still swept.
4. A stale point that survives the delete is still a loud `SinkLoadError` (no silent under-delete).
5. Co-tenant `id_namespace` partitions remain untouchable; `reconcile: false` still issues no
   `delete`.
6. Sweep memory stays O(batch): candidates are paged, never materialised whole.
7. Regression test fails on the pre-fix code (live record erased, load green) and passes after.
8. `make test` stays green at ≥ 2985 passed / 28 skipped.
