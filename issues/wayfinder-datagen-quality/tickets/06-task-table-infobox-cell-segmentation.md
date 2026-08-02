# 06 — Task: table/infobox cell-aware segmentation

Type: task
Status: open
Blocked by: 01

## Problem (measured, two independent sightings)

- Pattern 5: infobox/table text welds mentions across cell boundaries — **`DJay Heath
  Ledger`** (a role fused to the next actor: a person who does not exist), **`Butler
  Detroit`** (two schools), `Pub\n\nThe London Gazette`, `2002 *`, `Sandra` (fragment of
  "Bullock, Sandra" becoming a second node beside `Sandra Bullock`).
- Parent ticket 11's recorded refinement: the RuleSentenceSplitter's newline rule makes each
  markdown table row one segment — reasonable but unmeasured on real tables.

## Task

Give segmentation a **cell notion**: markdown table rows split at `|` boundaries; infobox-ish
key-value lines (short line, colon/tab) treated as cell pairs; never emit a mention spanning
a cell boundary. The repo already parses structured markdown with tables — reuse that
structure signal instead of re-detecting in the splitter where possible.

Measured before/after: cross-cell weld count on the judged sample + a targeted table-heavy
probe set; retrieval gates re-run.

## Outcome

Splitter/builder change + trap tests (the five sighted welds become fixtures) + measured
delta recorded here.
