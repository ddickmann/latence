# R5 — past the scan cap, the KG views call a checkpointed Evidence mention "not checkpointed"

- **Severity:** major
- **Dimension:** console read-only / provenance truth
- **Branch:** feat/release-fleet-integration
- **Tickets:** #188 (the debugging views), ADR-0021/0031 (provenance is true), ADR-0033 (memory-bounded)
- **Status:** fixed

## What is wrong

`packages/latence-console/src/latence_console/artifacts.py` — `RunArtifacts.mention()` and its
three siblings `chunk()` / `document()` / `node()` each began:

```python
if self._scanned is not None:
    return self._scanned.mentions.get(record_id)
```

`self._scanned` is the *capped* list-view scan: `_Kept.put` stops absorbing at
`MAX_SCAN_RECORDS = 2000` and sets `truncated`. A miss in that map therefore has two very
different meanings — the run never checkpointed the record, or the cap dropped it — and the code
collapsed both into `None`, with no fallback to the streaming `_lookup()` that reads what is
actually on disk.

The KG surface makes this reachable on every real run: `graph_edges()` / `graph_nodes()` always
scan (they need the node and edge maps), so by the time `browse._mention_evidence` resolves an
edge's Evidence the truncated map is already installed. `_mention_evidence` then renders

```
(mention not in this run's checkpoints)
```

for a mention that *is* in the checkpoints — a **false provenance claim**, and the exact inverse
of `browse.py`'s documented contract ("a mention id the run did not checkpoint is shown as
unresolved") and of ADR-0021/0031. 2000 records is tiny for a real run, so essentially every real
run's KG Evidence was mislabelled as unresolved. The per-entity page additionally carried no
`Partial view` notice at all, so it did not even disclose that its own view was capped.

## Failure scenario

A run with `checkpoints/entities.jsonl` holding 2500 `EntityMention`s and `checkpoints/graph.jsonl`
holding two nodes and one edge whose Evidence cites `…#mention-2499`:

- `RunArtifacts(storage, run_dir).mention("…#mention-2499")` on a *fresh* object returns the
  record — the targeted streaming lookup finds it.
- `GET /run/<id>/graph` renders `(mention not in this run's checkpoints)` and never shows the
  mention's text, file, pages or offsets.
- `GET /run/<id>/entity?id=node-a` does the same, and shows no `Partial view` notice.
- The same object, after any list view has scanned, answers `mention("…#mention-2499")` with
  `None` — the lookup got *worse* by being called second.

## The fix

1. `RunArtifacts._from_scan()` — the scan answers a single-record lookup only where it is
   **authoritative**: a hit is always an answer; a miss is an answer only when that map is not
   `truncated` and the scan did not stop early (new `_Scan.stopped_early`, set when every map
   filled before the last checkpoint was read). Otherwise the four lookups fall through to the
   streaming `_lookup()` / `_lookup_node()`.
2. `RunArtifacts.mentions_by_id()` + `_lookup_many()` — the KG page resolves *all* its cited
   mentions in ONE streaming pass instead of one pass per Evidence entry, keeping only the
   requested keys (peak memory O(ids asked for), which is what the caller already holds a
   rendered row for). `browse.edge_views` collects the page's citations and passes the resolved
   map down; `_mention_evidence` is now a pure formatter over `EntityMention | None`.
3. `render_entity` takes `truncated=` and emits the same `Partial view` notice the graph list
   does; `server._entity_view` passes `artifacts.truncated()`.

Per-key semantics are unchanged everywhere: most-downstream file wins, later line within a file
wins, so the redaction-safety of the cap (ADR-0042/0044) is untouched.

## Acceptance criteria

- [x] A cited mention beyond `MAX_SCAN_RECORDS` resolves on `GET /run/<id>/graph` — its real
      text, file, pages and offsets are shown, not the "not checkpointed" row.
- [x] The same holds on `GET /run/<id>/entity?id=…`.
- [x] `mention()` returns the record whether or not a list view has already scanned.
- [x] An id the run genuinely never wrote still resolves to `None` / the honest unresolved row —
      the fallback invents nothing.
- [x] The per-entity page states `Partial view` when the scan was capped.
- [x] Resolving a page of Evidence costs one extra checkpoint pass, not one per mention
      (ADR-0033 stays intact; the existing R4 memory budgets stay green).
- [x] `make test` stays at or above 2985 passed / 28 skipped.

## Regression tests

`packages/latence-console/tests/test_console_scale.py`:

- `test_graph_resolves_evidence_beyond_the_scan_cap` (failed pre-fix)
- `test_entity_page_resolves_evidence_beyond_the_scan_cap` (failed pre-fix)
- `test_lookup_after_a_truncated_scan_still_finds_the_record` (failed pre-fix)
- `test_absent_mention_is_still_reported_as_unresolved` (guards against a fallback that invents)
- `test_entity_page_states_a_partial_view` (failed pre-fix)
