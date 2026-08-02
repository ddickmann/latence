# 07 — Task: surface-form duplicate merge (coref-lite)

Type: task
Status: open
Blocked by: 01

## Problem (measured)

Pattern 6: title form and body form of one entity become two nodes (`Fadel Brahami` /
`Mohamed Fadel Brahami`; `Geraldine Ferraro` / `Geraldine Anne Gerry ''Ferraro`), two
formats of one date (`October 27, 1986` / `1986 - 10 - 27`), and truncated work titles
beside their full form (`Symphony No.` / `Symphony No. 1 in C major`).

## Task

Deterministic surface-normalization merges at disambiguation time — zero-LLM, rule-scoped to
the sighted shapes: (a) name-subsequence containment with shared surname/token order
(person shapes), (b) date canonicalization (already have DE/EN date handling in the repo —
reuse), (c) truncation-prefix merge for work titles ending in structural fragments
(`No.`, `Vol.`). Each rule ships with a counter-example test (two genuinely different
entities that must NOT merge) — over-merging is worse than the duplicates.

Relevant scale fact from the parent campaign: both shipped resolvers wall past ~10⁴ unique
surfaces; the harness's linear `ExactDisambiguator` is the productization candidate these
rules would extend. Do not couple this ticket to that productization — rules first, provider
packaging is production-direction work.

## Outcome

Rules + trap/counter-example tests + measured duplicate-count delta on the judged sample.
