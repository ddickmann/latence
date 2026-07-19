# R5 — the entity page tells the reader an entity is isolated when the cap dropped its edges

- **Severity:** major
- **Dimension:** console read-only / truthful reporting
- **Branch:** feat/release-fleet-integration
- **Tickets:** #188 (the debugging views), ADR-0017 (files are the source of truth),
  ADR-0021/0031 (provenance is true), ADR-0034 (no silent errors)
- **Status:** fixed

## What is wrong

The list views' scan keeps at most `MAX_SCAN_RECORDS = 2000` records of each kind
(`artifacts._Kept.put` stops absorbing and sets `truncated`). The KG views always scan —
`browse.edge_views` reads `artifacts.graph_edges()`, which is `self._scan().edges` — so on any run
whose graph checkpoint holds more than 2000 edges, every edge past the cap is simply absent from
what the page renders.

`render.render_entity` then did two things with that capped list:

```python
f"<h2>Relations ({len(edges)})</h2>",
...
parts.append(_empty("No relation in this graph touches this entity."))
```

Both are **claims about the whole run** made from a **prefix of the run**. For an entity whose
every edge sits beyond the cap the page renders `Relations (0)` over the flat sentence *"No
relation in this graph touches this entity."* — an affirmative statement of isolation that the
run's own `checkpoints/graph.jsonl` contradicts. ADR-0017 makes those files the source of truth;
the page states the opposite of what they say, unqualified.

The `Partial view` notice added by R5-console_readonly-8 does appear on this page, but it is a
notice *about the rows below* ("the rows below are a prefix of the run"). It does not retract the
sentence, and a reader who sees a heading counting zero and a card asserting isolation reasonably
reads the notice as pagination boilerplate. A hedge somewhere on the page is not the same as not
making the false claim.

The same shape sits on the sibling empty states, reachable by the identical mechanism:
`render_graph`'s *"This run assembled no graph nodes/edges."* and `render_chunks`'
*"This run checkpointed no chunks."*, plus `render_graph`'s `Entities (N)` / `Relations (N)`
tallies.

## Failure scenario

Proved with a throwaway probe and then pinned as a test. A run whose
`checkpoints/graph.jsonl` holds 2501 `GraphNode`s and 2500 `GraphEdge`s, where `e{i}` runs
`n{i} → n{i+1}` — so **every** node in the run is an endpoint of at least one edge on Storage:

- `GET /run/kghuge/entity?id=n2499` returns 200 containing `<h2>Relations (0)</h2>` and
  `No relation in this graph touches this entity.`
- Reading `checkpoints/graph.jsonl` directly, bypassing the console, finds `e2498` and `e2499`,
  both with `n2499` as an endpoint.
- An operator auditing *why the KG connected two entities* is told, without qualification, that
  the entity is isolated.

## The fix

`packages/latence-console/src/latence_console/render.py`:

1. `_empty(message, *, partial=None, truncated=False)` — a call site now supplies both the
   whole-run claim and the weaker true statement. On a capped read the affirmative sentence is
   withheld and `partial` is shown: the same emptiness, without the claim the files contradict.
2. `_count(total, truncated)` — a bare `(0)` reads as "there are none". On a capped read the
   tally renders `0 read so far`, so the number is never left standing as a total.
3. Applied at every empty state and tally derived from the capped scan: `render_entity`'s
   relations, `render_graph`'s entities and relations, `render_chunks`' chunk list.

Nothing is hedged where the reader actually read the whole run — a page that hedges
unconditionally teaches the reader to ignore the hedge, which would cost the fix its value.

## Acceptance criteria

- [x] An entity whose edges the cap dropped is never described as touched by no relation, and
      the ground truth is asserted from the checkpoint file, not from the reader under test.
- [x] Its relations tally is not presented as a total of the run.
- [x] A run that genuinely assembled no nodes/edges still says so plainly, unqualified.
- [x] A complete read of a graph with one edge still renders `Relations (1)` with no hedge and
      no `Partial view` notice.
- [x] No new scan, pass or materialisation — the fix is presentational only, so ADR-0033's
      memory budgets are untouched.
- [x] `make test` stays at or above 2985 passed / 28 skipped.

## Regression tests

`packages/latence-console/tests/test_console_scale.py`:

- `test_entity_page_never_claims_isolation_on_a_capped_read` (**failed pre-fix**)
- `test_entity_page_speaks_plainly_when_the_read_was_complete` (guards against hedging always)
- `test_complete_graph_with_no_edges_still_states_it_plainly` (guards the true claim's survival)
