# R1 — quarantining a document's first chunk deletes the whole document's page resolution

- **Severity:** major
- **Dimension:** depth actually gained / PROVENANCE IS TRUE (ADR-0021, ADR-0031),
  NO SILENT ERRORS (ADR-0034)
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/src/latence_core/page_index.py:190` (`_index_for`,
  the absorb-from-the-stream route) together with
  `packages/latence-core/src/latence_core/stages/screening.py:354` (chunk quarantine)
  and `packages/latence-core/src/latence_core/runner.py:1241` (the Runner's
  `CONTENT_SCREENING` dispatch, which is what actually hands the filtered stream on)
- **Invariants at stake:** ADR-0021/0031 (a chunk offset resolves to a real
  original-document offset/page), ADR-0034 (a degradation is observable, never a
  confident clean), ADR-0008 (the Quality Report's bytes are the contract — so the fix
  may not add a report field)
- **Status:** resolved (fixed in this commit)
- **Siblings:** `-0.md` (extraction Providers built a per-chunk `PageOffsetIndex`),
  `-1.md` (redaction Providers, same), `-2.md` (the S10 drift audit gated on
  `record.page_map`). Those three fixed *call sites* that read the map off the record.
  This ticket is the class of failure that survives all three: the call sites are now
  correct and the map is still gone, because something upstream **deleted the record it
  rides on**.

## What is wrong

C1 (schema v18) moved a document's `PageMap` off every chunk and onto its **first**
chunk, with `PageIndexResolver` absorbing it from the stream (route 2 in the module
docstring). That is a real deepening — ~92% of a chunk checkpoint line was the duplicated
map — but it silently converted a property of a *record* into an invariant on the
*stream*:

> every document's page map must ride a chunk that SURVIVES to the consumer.

Nothing kept that invariant. Content Screening is a chunk **filter**: a chunk whose risk
score reaches `quarantine_threshold` is removed from the pipeline entirely
(`screening.py:354` → `passed.append(None)`), and the Runner hands `outcome.passed`
straight downstream (`runner.py:1241`, `return iter(content_outcome.passed)`). A
document's opening chunk — title block, header, a pasted instruction — is a *likely*
place for an injection marker, and it is exactly the chunk C1 put the map on. Removing it
removes page resolution for the entire document: every later chunk reaches Entity,
Relation, Redaction and the header/inline-ref Stages with `page_map=None` and no
document that ever taught the resolver, so `pages_for_span` returns `None` and every
mention, relation and PII span degrades to its chunk's inherited page range. Before C1
the map rode all N chunks and filtering could not do this; the regression is introduced
by the deepening, not pre-existing.

The second half is that the degradation is **unobservable**. The producing resolvers live
inside the Provider packages, so no Quality Report field sees them; and
`PageIndexResolver.unresolved` — the counter whose docstring cites ADR-0034 ("counted
rather than swallowed … so a Quality Report can surface the gap") — had **no production
consumer at all**: `grep -rn '\.unresolved' packages/*/src` found only the definition.
The run reports clean.

## Failure scenario

`KeywordContentScreener({"quarantine_threshold": 0.5})` over a 5-chunk document whose
first chunk carries `ignore previous instructions`:

```
chunks: 5   carriers: [0]
passed: 4   quarantined: 1
carriers after screening: []
resolver documents: 0
pages_for_span(chunk-2) -> None
unresolved: 1
```

Through the Runner (source → parse → chunk → content_screening) the same thing lands in
the checkpoint the next Stage reads. On the multi-page fixture, the observable damage is
a wrong page number, not merely a missing one:

```
AssertionError: assert 5 == 4
  where 5 = ...page_end   (inherited from chunk-3's [4,5] range)
  and   4 = page_for_offset(2179)   (the mention's own page)
```

`grep -rn '\.unresolved' packages/*/src` → no reader, so nothing surfaces it.

## What to build

Keep the deepening; keep the invariant it introduced.

1. `latence_core.page_index.rehome_page_maps(kept, *, dropped_from)` — re-stamp each
   document's page map onto the first chunk of it that survived. Owned by the module
   that owns the map-carriage knowledge, not by the screener. A document whose carrier
   survived (or which never had a map) is returned **untouched** (identical objects), so
   an unfiltered stream is byte-identical to its input. `O(len(kept))`, one map reference
   per document — nothing the caller was not already holding (ADR-0033).
2. The Runner applies it at the one point chunks leave the stream — the
   `CONTENT_SCREENING` dispatch — so **every** `ContentScreener` Provider inherits the
   guarantee rather than only the ones that remember. No new layer, no new Stage, no
   contract change: this is the existing seam being made real at its boundary.
3. Give `PageIndexResolver.unresolved` its first production consumer: the report's drift
   audit (`_DriftAggregator.finish`) warns, **counts only**, when chunks went unaudited
   because their document had no map — a section that merely shrinks otherwise renders as
   a confident `clean`. Deliberately a log warning and not a report field: `-2.md`
   established that a fifth `DriftDiagnostics` field bumps `QUALITY_SCHEMA_VERSION` and
   rewrites the C5 byte-identity fixture (ADR-0008), which a provenance fix must not do.
4. Correct the `page_index.py` module docstring: state the stream invariant and where it
   is kept, and stop promising a Quality Report consumer that does not exist.

## Acceptance criteria

1. Through `LocalRunner`, quarantining the map-carrying first chunk leaves every
   surviving chunk of that document resolvable: a `PageIndexResolver` that observes the
   post-screening checkpoint reports `documents == 1`, `unresolved == 0`, and
   `pages_for_span` is non-`None` for every survivor.
2. The surviving pages are **exact**, not inherited: on a multi-page document every
   mention's `page_start`/`page_end` equals the `PageOffsetIndex` ground truth, including
   at least one mention whose page differs from its chunk's inherited range (non-vacuity).
3. Nothing dropped ⇒ nothing changed: `rehome_page_maps` returns the identical record
   objects (byte-identical stream).
4. `PageIndexResolver.unresolved > 0` produces exactly one counts-only warning on
   `latence.quality` — no record id, no document id, no content — and the happy path is
   silent.
5. `tests/data/quality_report_c5_fixture.json` unchanged; the ADR-0008 byte-identity gate
   stays green.
6. `make test` green; `ruff` + `mypy --strict` clean on the touched files.

## Regression tests

`packages/latence-core/tests/test_page_index_resolver.py`

* `test_quarantining_the_first_chunk_does_not_strip_the_documents_page_map` — AC 1,
  through the real Runner. Pre-fix: `assert 0 == 1` (`resolver.documents`).
* `test_pages_stay_exact_when_screening_quarantines_the_map_carrying_chunk` — AC 2.
  Pre-fix: `AssertionError: assert 5 == 4` (a mention inheriting its chunk's page range).
* `test_rehoming_is_a_no_op_when_nothing_was_dropped` — AC 3.
* `test_drift_audit_surfaces_the_unresolved_count_instead_of_shrinking_silently` — AC 4.
  Pre-fix: `assert 0 == 1` (no warning emitted).
* `test_drift_audit_is_silent_when_every_chunk_resolves` — AC 4, the negative control.

## Follow-up (not in this fix)

The same stream invariant would be broken by any *future* chunk-dropping Stage. Today
Content Screening is the only one in core (Intake Screening drops pre-parse
`ParserInput`s, which carry no map), and the Runner is the only place a filtered chunk
stream is handed on — so the single call site is complete. If a second filter ever
appears, it belongs behind the same `rehome_page_maps` call rather than a second copy of
the knowledge.
