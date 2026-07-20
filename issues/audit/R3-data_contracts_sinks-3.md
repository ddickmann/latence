# R3-data_contracts_sinks-3 — [major] QdrantSink's reconciling sweep is collection-global, so the shared-collection setup its own `id_namespace` docstring recommends silently erases the co-tenant corpus

**Area:** data contracts / sinks (#183, `latence-sink-qdrant`)
**Severity:** major
**Status:** FIXED

## What is wrong

`DEFAULT_ID_NAMESPACE` (sink.py:82-84) tells the operator:

> Override ``id_namespace`` (per corpus) to share one collection between two corpora whose
> ``record_id`` spaces could collide.

But `_sweep` (sink.py:435-482) deletes by a collection-**global** filter — `must_not(_latence_load_id
== <this run's marker>)` — with no notion of which namespace a point belongs to. So in exactly the
setup the docstring proposes, loading corpus B deletes every point of corpus A, both loads exit 0,
and `verify_absent` afterwards reports A's records as "absent", i.e. it hands back **erasure
evidence for records that were destroyed by accident, not by a retraction** (ADR-0034: no silent
errors; the GDPR/Retraction story of ADR-0018/0029 rests on that evidence being true).

The existing sweep guard (`marked != touched`) does not catch it: it compares this load's own marker
count against its own record count, so it detects a *concurrent* load and nothing else. With
corpus B holding 1 record and 1 marked point, the guard passes vacuously.

Nothing warns: `reconcile` defaults to `true`; the `id_namespace` docstring, the README table
(README.md:53,136) and the CLI help never mention that `--no-reconcile` would be mandatory here; and
the only warning in the module docstring is scoped to "loading an *incremental* Export", which does
not describe this case.

## Failure scenario (pre-fix, reproduced with the shipped `FakeQdrantClient`)

1. Export corpus A (`A#chunk-0`, `A#chunk-1`) and corpus B (`B#chunk-0`).
2. `QdrantSink({"collection": "shared", "id_namespace": NS_A}).load(A)` →
   `upserted 2, deleted 0 stale point(s)`; 2 points in the collection.
3. `QdrantSink({"collection": "shared", "id_namespace": NS_B}).load(B)` →
   `upserted 1, deleted 2 stale point(s)`, exit 0; 1 point left, `record_id`s = `["B#chunk-0"]`.
4. `verify_absent(["A#chunk-0", "A#chunk-1"])` → `[]` — certifying erasure of a corpus nobody
   retracted.

## What to build

Scope the sweep to the namespace partition it owns:

- Stamp every point the sink writes with the id namespace it was derived under
  (`_latence_id_namespace`), alongside the existing digest and load marker; add it to
  `RESERVED_PAYLOAD_KEYS` so an Export column of that name is still a loud `SinkExportError`.
- Make the marker count, the stale-set filter and the post-delete convergence count all carry
  `must(_latence_id_namespace == <mine>)`. The sweep then deletes exactly the stale points of *this*
  corpus and leaves a co-tenant corpus untouched.
- Tell the truth in the prose: module docstring, `DEFAULT_ID_NAMESPACE`, `reconcile`/`id_namespace`
  config docs, README, and the `FakeQdrantClient` docstring sentence that claimed the global
  `must_not` was there to collect points "written by an older loader".

## Acceptance criteria

1. Two corpora loaded into one collection under two `id_namespace` values coexist: the second load
   reports `deleted == 0` and both corpora's `record_id`s are still present afterwards.
2. `verify_absent` on the first corpus's records after the second corpus's load returns them as
   **present** (no false erasure evidence).
3. Retraction within one namespace still converges: re-loading a shrunk Export in namespace A
   deletes exactly A's dropped points, and never B's.
4. The concurrent-load guard (`marked != touched` ⇒ `SinkLoadError`, nothing deleted) still fires.
5. The single-corpus/default-namespace behaviour of every existing test is unchanged;
   `make test` stays green at or above 2985 passed / 28 skipped.

## Regression test

`packages/latence-sink-qdrant/tests/test_qdrant_sink.py::test_second_corpus_in_a_shared_collection_does_not_sweep_the_first`
(plus `::test_retraction_sweeps_only_the_loading_namespace`) — fails on the pre-fix sweep, passes
after.
