# R6-console_readonly-11 — the quarantine view still claims absence of risk markers on a capped read

**Severity (as reported):** major
**Disposition:** CONFIRMED — fixed.

## Site

- `packages/latence-console/src/latence_console/render.py` — `render_quarantine`, the
  "Flagged — kept, with a risk marker in the corpus" heading and its empty card.
- Data path: `server.ConsoleApp._quarantine_view` -> `browse.flag_entries(artifacts)` ->
  `RunArtifacts.chunks()` -> the `MAX_SCAN_RECORDS = 2000` capped `_scan()`.

## What was reported

The R5 fix "never claim absence from a capped scan" (`_count` / `_empty(partial=...)`) was wired
into `render_graph`, `render_entity` and `render_chunks`, but NOT into `render_quarantine`. The
flagged-risk-marker list — the surface a RAG consumer reads to decide what to exclude — therefore
still renders a bare `(0)` tally and the flat affirmative sentence
"No record in this run's corpus carries a risk marker." on a run whose corpus demonstrably does
carry them.

The R5 issue file `R5-console_readonly-9.md` says in its own words "The notice alone was not
enough", yet on this page the top-of-page `Partial view` notice is the ONLY qualification.

## Failure scenario (reproduced)

A run with 2500 real `ChunkRecord`s whose LAST chunk carries a `prompt_injection` `RiskMarker`,
plus the matching `FLAG` `ScreeningFinding` in `screening/content_screen.jsonl`.
`artifacts.chunks()` is the capped scan, so the flagged chunk is dropped before rendering.

`GET /run/<run>/quarantine` rendered exactly:

```html
<h2>Flagged — kept, with a risk marker in the corpus (0)</h2>
<div class="card"><p>No record in this run&#x27;s corpus carries a risk marker.</p></div>
```

— an unqualified claim of absence that the run's own checkpoints contradict (ADR-0017: the files
are the source of truth). Contrast the sibling views the same fix DID reach: `render_chunks`
emits the qualified pager and `render_entity` emits "Relations (0 read so far)".

The green gate could not see it: the only R5 scale regression asserts
`"Relations (0 read so far)"` (`test_console_scale.py:577`); nothing covered the quarantine view.

## Scope of the fix

Only the **flagged** half is affected. `RunArtifacts.quarantined()` reads the
`quarantine/` sidecars in full (it is not part of the capped `_scan()`), so the
"Quarantined — removed from the pipeline" tally and its
"Screening held nothing back in this run." sentence remain TRUE on a capped read and must NOT be
hedged — hedging a complete read teaches the reader to ignore the hedge.

## Acceptance criteria

- [x] On a capped read, the flagged heading renders `(0 read so far)`, never a bare `(0)`.
- [x] On a capped read, the empty card states the weaker, true claim ("no risk marker was among
      the records this view read") instead of asserting absence from the corpus.
- [x] On a COMPLETE read the page still speaks plainly: `(0)` and the flat sentence survive, and
      "read so far" does not appear.
- [x] The quarantined (held-back) half is unchanged in both cases — its read is not capped.
- [x] A regression test builds a run whose flagged chunk the cap demonstrably drops, asserts the
      marker is on Storage (read from the checkpoint file, not from the reader under test), and
      fails on the pre-fix source.
- [x] `make test` stays at or above 3320 passed / 29 skipped.
