# 02 — Grilling: relation-fabrication policy (the dominant error)

Type: grilling
Status: open
Blocked by: 01

## Question

The dominant measured error (pattern 1, `M6-CALIBRATION-OFFICIAL.md`): **58% of
relation-bearing Hyperedges carry ≥1 triple the span does not assert** — relations asserted
from co-occurrence rather than predication. Damage is label-concentrated: `capital of` 1/11
genuine, `founded by` 3/15 (one pointing at a *date*), `spouse of` 1/4, while `located in`
(41 triples) is largely reliable. The high-γ tail is the LEAST safe (Q5 has the most
`incorrect`; the single highest-γ Hyperedge in the sample, γ 0.9325, is a fabrication).

Decide the lever mix, each candidate arbitrated by the judged sample + a threshold/ablation
sweep re-scored with the frozen instruments:

- **Per-label γ floors** (fog: are GLiNER2 confidences comparable across labels?).
- **Schema pruning** — drop or rephrase near-pure-noise labels; GLiNER2's extraction is
  schema-driven, so label *definitions* are a first-class lever.
- **Predication evidence gate** — zero-LLM structural check that the span contains a
  predicate cue for the label (verb/pattern lexicon per label, DE/EN); pattern-2 title-only
  spans (ticket 03) are the degenerate case.
- **Type-constrained slots** — `founded by → DATE` and `spouse of` between two countries are
  type-illegal; a slot-type table kills a class of fabrications for free (interacts with
  ticket 05).
- Threshold sweep: the 0.5 default was never tuned; sweep against fabrication-vs-yield.

## Constraints

Retrieval gates must not regress (weak/entities-only Hyperedges are UNAFFECTED by all levers
above — they carry no relations; the piecewise policy already leans on them). Sacrifice rule
applies: a label is only pruned with its measured precision cited.

## Outcome

An ADR ruling the lever mix, with the measured fabrication-rate delta and yield cost per
lever; spawns/scopes ticket 03 and the fog items (per-label thresholds, learned verifier).
