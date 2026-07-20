# R6-invariants-1 — `_merge_floor` greedy first-wins defeats the ADR-0044 financial floor

**Status:** WONTFIX — DUPLICATE, already fixed by R6-invariants-0 (commit `0718105`).
**Severity as reported:** major
**Invariant:** ADR-0044 (universal financial-PII redaction floor)
**Site:** `packages/latence-core/src/latence_core/redaction_policy.py` — `_merge_floor`,
invoked from `finalize_redacted_chunk`.

## What was reported

`_merge_floor` unions the Provider's accepted spans with `floor_candidates` using a greedy
first-wins rule (accept iff `span[0] >= last_end`, else DROP WHOLE). A single learned
Provider span that merely overlaps the leading characters of a card/IBAN/SSN therefore
discards the floor span entirely, re-exposing its uncovered tail — on the plain
SINGLE-Redactor path that `deploy/env/gpu-learned/stack.yaml` and
`deploy/env/full-stack/stack.yaml` both run with `redaction.gliner2`.

Reported failure scenario: `chunk.content = 'Contact Jane Roe Card 4111111111111111 end.'`;
`floor_candidates` returns `(22, 38, 'credit_card', ...)`; the Provider emits
`(8, 26, 'person')`. Claimed output: `masked_content == 'Contact [PERSON]111111111111 end.'`
with `pii_spans == ['person']` — 12 card digits shipped, and absent from the audit record.

## Why this is WONTFIX

The finding is real in substance but **stale**: it describes the code as it stood before
`0718105`, which landed earlier in this same audit round. R6-invariants-0 covered exactly
this hole — its ticket names it **scenario (C), "SAME HOLE IN `_merge_floor`, single
Redactor"**, with the identical fixture string and the identical span pair. That fix
replaced the greedy drop-on-overlap rule in BOTH call sites with the shared
coverage-preserving `_resolve_clipping` helper (clip to the uncovered tail, never drop),
plus mask-outranks-TAG ordering in `_compose_prior_spans`.

Re-verified against the current tree by driving the real code with the reported inputs:

```
floor:  [(22, 38, 'credit_card', '4111111111111111', 'floor:credit_card', 1.0)]
merged: [(8, 26, 'person', 'Jane Roe Card 4111', 'gliner2', 0.9),
         (26, 38, 'credit_card', '4111111111111111', 'floor:credit_card', 1.0)]
```

The floor span is CLIPPED to `[26, 38)` and kept, not discarded. End-to-end through
`finalize_redacted_chunk`, `masked_content == 'Contact [PERSON][CREDIT_CARD] end.'` and the
`credit_card` span is present in `pii_spans`, so the Quality Report and console see it. The
reported output `'Contact [PERSON]111111111111 end.'` is not reproducible on this tree.

The secondary claim — "the R5 composition fix replicates the identical resolution rule
rather than fixing it" — is likewise addressed: `_compose_prior_spans` now resolves masking
spans first via `_resolve_clipping` and subtracts TAG spans against the accepted masking
coverage (`_subtract_covered`).

## Regression coverage already in place

`packages/latence-core/tests/test_redaction_composition.py`:

- `test_provider_span_overlapping_the_floor_does_not_truncate_it` — the exact single-Redactor
  scenario reported here (asserts `'111111111111' not in masked_content`).
- `test_partially_overlapping_span_does_not_re_expose_a_masked_card` — the chained variant.
- `test_upstream_tag_span_does_not_swallow_the_financial_floor` — the action-swallow variant.
- A parametrised coverage-monotonicity property over overlap geometries (tail, head, nested
  both ways, abutting, identical, disjoint).

No further code change is warranted; adding a second copy of an existing passing test would
be vacuous.

## Acceptance criteria (all met by `0718105`, re-verified here)

1. A Provider span overlapping the floor's leading characters leaves the financial value
   fully masked. — MET.
2. The `credit_card` span survives into `pii_spans` (audit record / Quality Report honest). — MET.
3. Output remains ordered and non-overlapping for `apply_masking`. — MET.
4. Disjoint/coincident span sets stay byte-identical (`hybrid_rule` stacks unchanged). — MET.
5. `make test` green at or above 3320 passed / 29 skipped. — MET.
