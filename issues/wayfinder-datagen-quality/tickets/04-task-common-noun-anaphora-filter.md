# 04 — Task: common-noun + anaphora promotion filter

Type: task
Status: open
Blocked by: 01

## Problem (measured)

Pattern 3, ~30/250 cases: mid-prose common nouns become permanent nodes (`study plans`,
`carnivals`, `census`, `invading`, `forthcoming year`) and — worst — the pronoun **`He`**
becomes a PERSON node that will silently conflate every unresolved male referent merged into
it.

## Task

A promotion filter at mention→node time: pronoun stoplist (DE/EN), lowercase-common-noun
shape check, minimum-specificity rule — while PRESERVING the judged sample's not-charged
rule: kind-terms that are the span's own topic (`Aircraft carrier`, `Sicilian pizza`) stay,
because the corpus asks about them. That asymmetry (span-topic vs mid-prose) is the design
core; get it from span position/heading structure, not a vocabulary list alone.

Measured before/after on the judged sample; retrieval gates re-run (entity-only Hyperedges
shrink — the piecewise policy leans on weak Hyperedges, so this one CAN move retrieval and
must be watched).

## Outcome

Filter + tests + measured delta (promotions removed, retrieval gate status) recorded here.
