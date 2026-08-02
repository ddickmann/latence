# 03 — Task: title-only / no-predicate span gating

Type: task
Status: open
Blocked by: 02

## Problem (measured)

Pattern 2: a span that is just a page title gives the extractor co-occurring names and no
predicate, and it invents one anyway — `Walker, Texas Ranger` → four fabricated triples at
once; a disambiguation parenthetical becomes an ORGANIZATION with the band's own name as a
PERSON "member of" it.

## Task

Implement the degenerate-case gate ticket 02 rules on: a cheap structural detector for
predicate-free spans (no finite verb cue, short, title-position/heading-derived) that
suppresses **relation** extraction on them while keeping entity mentions (the entities are
real; the predicates are not). DE/EN. Measured before/after on the judged sample per the
frozen instruments; retrieval gates re-run.

## Outcome

Builder/extraction change + tests + the measured delta recorded here.
