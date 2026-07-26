# R1 — Quarantining a document's sole page-map carrier destroys page resolution for the whole document

- **Severity:** major (as reported)
- **Dimension:** memory claims real / PROVENANCE IS TRUE (ADR-0021, ADR-0031), NO SILENT
  ERRORS (ADR-0034)
- **Branch:** feat/arch-deepening
- **Invariants at stake:** ADR-0021/0031 (chunk offsets resolve to real original-document
  pages), ADR-0034 (degradation is counted, never swallowed), ADR-0008 (Quality Report
  byte-identity), ADR-0033 (streaming / memory-bounded runner)
- **Status:** **WONTFIX — confirmed defect, but already fixed on this branch.** Duplicate of
  `issues/audit-arch/R1-depth_actually_gained-3.md`, fixed in commit `0f76a6a` earlier in
  this same audit campaign. No further code change is required or made by this ticket; what
  follows is the independent re-verification, including the residual sub-claim that is
  still literally true but is **not** a defect.

## What was reported

C1 changed the *carriage* of `PageMap` from "on every chunk" to "on a document's **first**
chunk only" (`SCHEMA_VERSION = 18`), with `latence_core.page_index.PageIndexResolver`
absorbing the map from the stream (route 2). Three sub-claims:

1. **Carrier removal.** After v18 a document has exactly ONE carrier in the record stream
   and nothing re-seeds it. Any Stage that removes chunks — Content Screening quarantine is
   the shipped case — can delete that sole carrier, after which the whole document's page
   resolution is lost for every downstream Stage, including the correctly-migrated core
   ones (`stages/screening.py:355-360`, `passed.append(None)` →
   `passed=[c for c in passed if c is not None]`).
2. **Dead counter.** The degradation is counted but the counter is dead code:
   `PageIndexResolver.unresolved` / `.diagnostics` have no production reader anywhere in
   the repo, so the module's ADR-0034 claim that the degradation is "counted rather than
   swallowed … so a Quality Report can surface the gap" is not realised.
3. **Dead route 1.** `PageIndexResolver.from_documents` (documented as route 1, the
   authoritative ADR-0017 copy) has no production caller either.

## Failure scenario (as reported, and reproduced against the *pre-fix* code)

A document whose first chunk contains a prompt-injection pattern is chunked into N chunks
(only chunk 0 carries the map). Content Screening quarantines chunk 0, so it never reaches
Entity / Relation / Redaction / Header-Refs. A `PageIndexResolver` that observes every
surviving chunk then returns `pages_for_span(...) -> None` for the page-straddling chunk,
and `resolver.unresolved > 0`. Every mention, relation, header-ref and PII span for that
entire document falls back to the inherited chunk page range for the rest of the run.

## Verification performed (2026-07-26)

### Sub-claim 1 — carrier removal: FIXED

`page_index.rehome_page_maps(kept, *, dropped_from)` re-stamps each document's map onto the
first chunk of it that SURVIVED, and the Runner applies it in the `CONTENT_SCREENING`
dispatch — the one point chunks leave the stream — so **every** `ContentScreener` Provider
inherits the guarantee rather than only the ones that remember:

```
packages/latence-core/src/latence_core/runner.py:1255
    return iter(rehome_page_maps(content_outcome.passed, dropped_from=chunk_records))
```

`dropped_from` is `self._as_type(stage, incoming, ChunkRecord)`, which returns a **list**
(`runner.py:1886-1905`), so it is safely re-iterable after `screen_content` has consumed it
— the fix is not silently dead on a spent generator.

A standalone probe reproducing the reported scenario end to end (5-page document, 5 chunks,
1 carrier, injection on chunk 0, `quarantine_threshold: 0.5`):

```
chunks: 5  carriers: 1
quarantined: 1  passed: 4
passed carriers (raw, pre-rehome): 0
passed carriers (post-rehome):     1
  RAW (no rehome): mismatched spans=4  resolver.unresolved=4
  RUNNER (rehomed): mismatched spans=0  resolver.unresolved=0
```

The reported defect is real on the raw Screening output (`RAW`: 4 of 4 surviving chunks
resolve to the wrong pages) and fully closed on the path the Runner actually ships
(`RUNNER`: 0 mismatches, 0 unresolved).

**The fix is protected by a non-vacuous regression test.**
`test_quarantining_the_first_chunk_does_not_strip_the_documents_page_map`
(`packages/latence-core/tests/test_page_index_resolver.py:422`) runs a real `LocalRunner`
pipeline (source → parse → chunk → content_screening → export), reads the actual
`chunk.jsonl` and `content.jsonl` checkpoints off Storage, asserts non-vacuously that the
map-carrying chunk really was quarantined out of the stream, and then asserts every
survivor still resolves. Teeth confirmed by reverting `runner.py:1255` to
`return iter(content_outcome.passed)` (the pre-fix state) and re-running:

```
>       assert resolver.documents == 1
E       assert 0 == 1
E        +  where 0 = <latence_core.page_index.PageIndexResolver object>.documents
packages/latence-core/tests/test_page_index_resolver.py:466: AssertionError
FAILED ...::test_quarantining_the_first_chunk_does_not_strip_the_documents_page_map
```

The wiring was then restored and the file verified byte-identical to `HEAD`.

### Sub-claim 2 — dead counter: FIXED

`_DriftAggregator.finish` (`quality_report_builder.py:743-757`) is the production consumer
of `PageIndexResolver.unresolved`. The audit's own grep command now returns it:

```
$ grep -rn '\.unresolved' packages/*/src
page_index.py:61:  ...docstring...
quality_report_builder.py:749:        unresolved = self._resolver.unresolved
```

It emits a counts-only `logger.warning` when chunks went unaudited, so a drift section that
merely *shrank* no longer reads as a confident `clean`. No record id, no document id and no
content appear in the message, and **no report field was added**, so the ADR-0008
byte-identity fixture is untouched. Both directions are tested
(`test_drift_audit_surfaces_the_unresolved_count_instead_of_shrinking_silently` and
`test_drift_audit_is_silent_when_every_chunk_resolves`).

### Sub-claim 3 — `from_documents` has no production caller: STILL TRUE, but NOT a defect

Confirmed literally: all 13 production constructions are the bare `PageIndexResolver()`
(7 ecosystem Providers, 5 core Stages, `_DriftAggregator`), and `from_documents` is
referenced only by tests and by the module docstring.

It is not being removed, for three reasons:

1. **It causes no wrong output and breaches no invariant.** Route 2 plus
   `rehome_page_maps` closes the one way the stream can lose a map. Route 1 being unused is
   an observation about surface area, not a behavioural defect — there is no input for
   which the shipped system produces a wrong page because `from_documents` went uncalled.
2. **Wiring it into a Runner would *violate* ADR-0033, not honour it.**
   `from_documents(documents)` builds a `{record_id: PageMap}` dict over the whole
   Documents checkpoint. The Runner streams (`_stage_records`, O(checkpoint batch)); it
   never holds the corpus, and materialising every document's page map to seed a resolver
   is precisely the whole-corpus materialisation ADR-0033 forbids. The docstring line "This
   is the route a Runner should use when it has the Documents checkpoint in hand" is
   therefore aspirational for `latence-core`'s own `LocalRunner` — which is exactly why no
   Runner calls it.
3. **`latence_core` is a framework and this is exported public API**
   (`latence_core/__init__.py:152`). It is the ergonomic constructor for a *third-party*
   Runner or an offline tool that genuinely does hold `DocumentRecord`s, and it is covered
   by four behavioural tests (seeded resolution, multi-document drift roll-up,
   out-of-range recovery, `PARSE_ERROR` document contributing no map). Deleting it would
   remove real, tested capability and four real tests to fix nothing.

Per the audit's own rule — do not force a change where there is no defect, and do not undo
a deepening — the correct action is to record the observation, not to shrink the seam.

## Residual observations (no action taken; recorded for the audit, not defects)

- `PageIndexResolver._diag` (`page_index.py:154`) is initialised and read (`diagnostics`,
  line 222) but **never written**. Its comment — "Drift for documents whose index has been
  discarded is folded in here as it is produced" — describes an eviction policy that does
  not exist: `_indices` is never evicted. This is harmless today (the aggregate is exact
  precisely *because* nothing is discarded) but it is unused generality, and the retention
  is O(documents seen by the Stage) rather than O(batch). Worth a look if a
  memory-bound ticket is ever raised against the resolver; it is not this ticket's defect
  and is not reachable as a wrong result.
- `rehome_page_maps` is applied at the only shipped chunk-removal point. The other two
  chunk-narrowing sites in the Runner (`_extract_input_chunks`, W18 masked mode) substitute
  `content` and never drop a record, so they cannot break the stream invariant.

## Acceptance criteria (all already met on this branch)

- [x] A document whose map-carrying first chunk is quarantined still resolves pages on every
      surviving chunk, end to end through a real `LocalRunner` run.
- [x] The guarantee is enforced by the Runner for **every** `ContentScreener` Provider, not
      per-Provider.
- [x] A stream nothing was dropped from comes back byte-identical (identical objects) —
      `test_rehoming_is_a_no_op_when_nothing_was_dropped`.
- [x] `PageIndexResolver.unresolved` has a production consumer that surfaces the gap,
      counts-only, with no PII and no Quality-Report field change (ADR-0034 + ADR-0008).
- [x] `make test` green; `test_page_index_resolver.py` 20/20 green.
