# R6-security_secrets-6 — `_compose_prior_spans` drops an upstream span wholesale (chained Redactors un-redact) — **WONTFIX (already fixed)**

**Severity as reported:** critical
**Status:** WONTFIX — the reported defect is real in history but was ALREADY FIXED earlier in this
same audit round by ticket `R6-invariants-0` (commit `0718105`). No further code change is warranted.

## What was reported

`packages/latence-core/src/latence_core/redaction_policy.py::_compose_prior_spans` (the R5-12 audit
fix for chained Redactors), reached from `finalize_redacted_chunk`, resolves the union of THIS
Stage's spans and an UPSTREAM Stage's spans with a greedy `start >= last_end` rule that DROPS an
overlapping span wholesale. A downstream Redactor whose span starts earlier and ends inside a prior
span therefore discards that prior span entirely, so its uncovered tail — text the upstream Stage had
already masked — is emitted in the clear in the exported corpus, while both Stages report success.
This is the exact defect the fix claims to close and the opposite of its own docstring guarantee
("a prior span survives only where this Stage left a gap").

## Failure scenario (as reported)

Chunk content `"Kontakt: Erika Mustermann, Musterstrasse 12, Berlin. Ende."`

1. Upstream `redact_pii` accepts `postal_address` `[26,50)` → correctly masks `"Musterstrasse 12, Berlin."`.
2. Downstream `redact_financial` accepts `person_name` `[9,40)` (model boundary starting earlier, ending inside the prior span).
3. The tagged list sorts `[9,40)` first (`char_start 9 < 26`); it is appended, `last_end` becomes 40;
   the prior `[26,50)` has `char_start 26 < 40` and is DROPPED entirely.
4. Characters `[40,50)` — masked upstream — are re-exposed verbatim.

The path is live: commit `bce9b44` rewired both GPU deploy stacks so `redact_pii` → `redact_financial`
are chained, i.e. two learned Redactors with independently-chosen boundaries funnel through this seam.

## Why WONTFIX

The report accurately describes the code **as of commit `bce9b44`**, which is no longer HEAD. The
verifier read a stale revision.

**Evidence the defect was real at `bce9b44`.** Replaying that revision's greedy-drop algorithm on the
reported scenario:

```
OLD composed:   [(9, 40, 'person_name')]
OLD re-exposed: ' 12, Berli'
```

**Evidence it is fixed at HEAD.** Commit `0718105` ("fix(audit-R6): The R5-12 audit fix
`_compose_prior_spans` unions chained-Redactor spans with a greedy first-wins rule…") replaced the
greedy drop with `_resolve_clipping`, a COVERAGE-PRESERVING resolution that CLIPS an overlapping span
to its uncovered tail instead of discarding it, and additionally resolves `TAG` spans second so a
non-mutating tag can never displace a mask. Probing HEAD with the reported scenario:

```
composed:                [(9, 40, 'person_name'), (40, 50, 'postal_address')]
prior chars NOT covered: []
re-exposed text:         ''
```

**Evidence the regression coverage is non-vacuous.** The report claims the shipped test file only
uses DISJOINT detector sets (email vs IBAN) so no test produces an overlapping span pair. That was
true at `bce9b44` (3 tests); `0718105` grew the file to 8 tests, including the parametrized
`test_composition_never_removes_a_masked_character`, which drives every overlap topology — among them
`prior=(10,20), this=(5,15)`, exactly the reported "this Stage starts earlier, ends inside prior"
shape. Temporarily reverting `_compose_prior_spans` to the `bce9b44` greedy-drop body and re-running
the file fails loudly:

```
AssertionError: composition UNMASKED offsets [15, 16, 17, 18, 19] (prior=(10, 20), this=(5, 15))
FAILED test_composition_never_removes_a_masked_character[prior_span1-this_span1]
FAILED test_composition_never_removes_a_masked_character[prior_span0-this_span0]
FAILED test_chained_redactors_compose_masks_into_the_exported_corpus
FAILED test_upstream_tag_span_cannot_swallow_the_financial_floor
FAILED test_overlapping_same_type_spans_from_chained_learned_redactors_compose
FAILED test_single_redactor_stack_is_byte_identical
```

The working tree was restored to HEAD afterwards; `git diff` is empty.

## Acceptance criteria (all already satisfied at HEAD)

- [x] Composing a downstream Stage's spans with an upstream Stage's never REMOVES a masked character
      (coverage monotonicity), for every overlap topology: tail-overlap, head-overlap, strictly-inside
      (both directions), abutting, identical, disjoint.
- [x] A partially-overlapping span CLIPS rather than drops, so the uncovered tail of a prior mask
      stays masked.
- [x] A non-mutating `TAG` span can never displace a `MASK`/`REPLACE`/`HASH` span, including the
      ADR-0044 financial floor's.
- [x] Output spans remain ordered and non-overlapping (`apply_masking`'s precondition).
- [x] The single-Redactor path (`prior` empty) is byte-identical — OFF-BY-DEFAULT invariant.
- [x] A regression test exists that FAILS on the pre-fix greedy body and PASSES at HEAD.

## Related

- `R6-invariants-0` — the ticket whose fix (`0718105`) closed this, introducing `_resolve_clipping`.
- `R6-invariants-1` — the sibling `_merge_floor` fix (`3815daa`) for the same greedy-drop class on the
  single-Redactor ADR-0044 floor path.
