# 13 — Task: min-support floor on the induced type vocabulary

Type: task
Status: open
Blocked by: 01

## Problem (measured, production S3 2026-08-04)

Per-chunk LLM label induction produces a **long tail of one-off entity types**. Measured on
the sanitized ohr_bench run (1,261 docs, 154,545 mentions): **15,584 distinct entity types**
and **1,343 distinct relation types**, the bulk of them singletons.

Ticket-12-adjacent but distinct: the [value-shape guard](../../../packages/latence-schema-inducer/src/latence_schema_inducer/sanitize.py)
(shipped 2026-08-04, commit `0809260`) removes SHAPE-junk — pure numerics (`10.3`),
fiscal/quarter forms (`1q22`), dates, addresses with a street-suffix token, `.net`-style
fragments. It cannot catch the remaining class, which is **well-formed but instance-specific**:

- `2001 indenture`, `2008 farm bill`, `12 cfr part 5`, `2.125% convertible senior notes due 2026`
- `1301 north dearborn parkway` (address-shaped but `parkway` is absent from the suffix list —
  a list that can never be complete; this is the evidence that shape rules bottom out)

These are *instances* the model named as *types*. No per-chunk rule can tell them from a
legitimate rare type, because the distinguishing signal is **corpus-level frequency**, which a
per-chunk inducer by construction cannot see.

## Task

A corpus-level **min-support floor** over the induced type vocabulary, applied as a pass
AFTER induction and BEFORE (or inside) type consolidation:

1. Count type support across the whole corpus (distinct documents, not mentions — a type
   evidenced by one document 40 times is still one-document evidence).
2. Drop induced types below `min_support` (default candidate: 2 documents); mentions carrying
   only dropped types fall back to the configured floor labels, exactly as an induction
   fail-open does today — no mention is silently lost, the fallback is counted.
3. Hand-authored/config label sets are EXEMPT (same scope rule the value-shape guard uses:
   only the untrusted LLM-induced vocabulary is filtered).
4. Wizard/config knob (`min_type_support`, 1 = off = today's behavior) so the default stays
   byte-identical for existing users; the campaign's `s3:` block carries it.

Design note: this is a **two-pass** shape (induce all → count → filter), so it must either
ride the existing type-vocabulary artifact the runner already writes
(`_latence/runs/<id>/type-vocabulary/`, observed in production) or be positioned as a
consolidation-stage input. Prefer the former — the artifact exists and is already durable.

## Gate

- Type-count delta on a re-run of the S3 ohr_bench corpus: distinct entity types before/after
  at `min_support` ∈ {1, 2, 3}; report retained-mention coverage at each (how many mentions
  keep an induced type vs fall back to floor).
- No regression on the frozen M6 retrieval gates.
- Judged-sample check that dropped types were genuinely instance-shaped, not rare-but-real.

## Outcome

Config knob + pass + tests, and a measured recommendation for the default `min_support` — with
the honest possibility that the answer is "1 (off), the tail is harmless" if the coverage cost
outweighs the vocabulary cleanliness.
