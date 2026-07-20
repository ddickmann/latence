# R6-invariants-0 — Span composition SWALLOWS masks: the R5-12 fix re-exposes card/IBAN/SSN text

**Severity:** critical
**Invariant regressed:** ADR-0044 (universal financial-PII redaction floor) / ADR-0042 (chunk-level redaction)
**Site:** `packages/latence-core/src/latence_core/redaction_policy.py`
 — `_compose_prior_spans` (L399-427, called unconditionally at L478) and its sibling `_merge_floor` (L372-396, called at L460), both inside `finalize_redacted_chunk` — the single seam ALL four shipped Redactors funnel through (core `hybrid_rule`, `pii-gliner`, `pii-gliner2`, `pii-presidio`).
**Reachable via:** `deploy/env/gpu-learned/stack.yaml` and `deploy/env/full-stack/stack.yaml` (`redact_pii` -> `redact_financial` -> `export_corpus`), and — for the `_merge_floor` half — ANY single-Redactor stack.

## What went wrong

Both functions union two span sets with a **greedy first-wins** rule: sort by
`(char_start, -length, source_rank, pii_type)`, then accept a span only if
`span.char_start >= last_end`, **dropping the whole span otherwise**. Two consequences,
both of which UN-REDACT text the pipeline had already masked:

1. **Action swallow.** The sort ignores the span's `RedactionAction`. A non-mutating
   `TAG` span (documented, audit-only, text left readable) that merely *starts earlier*
   outranks a `MASK` span — including the ADR-0044 financial floor's. `apply_masking`
   then `continue`s past the TAG span, and nothing masks the region at all.
2. **Partial-overlap truncation.** A span that overlaps an accepted one by a single
   character is discarded **wholesale** rather than clipped to its uncovered tail, so the
   part of it that nothing else covers is emitted verbatim.

The R5-12 fix was written to stop a downstream Redactor clobbering an upstream one's
masks. It regressed the very invariant it was written to protect.

## Failure scenario (all three verified by driving the real `finalize_redacted_chunk`)

**(A) ACTION SWALLOW — total leak.** `content = 'Note: Jane Roe Card 4111111111111111 filed.'`
Upstream Stage runs `action=TAG`, emits person span `[6,36)`. Downstream `redact_financial`
runs `action=MASK`; the ADR-0044 floor correctly detects `credit_card` at `[20,36)`.
The prior TAG span at 6 sorts first, the floor's MASK span is dropped, and the TAG span
is skipped by `apply_masking`:
`masked_content == 'Note: Jane Roe Card 4111111111111111 filed.'` — the full 16-digit
card exported in the clear. Reverting the R5 fix (identity compose) yields
`'Note: Jane Roe Card [CREDIT_CARD] filed.'`.

**(B) PARTIAL-OVERLAP TRUNCATION, chained.** Reachable with the shipped `gpu-learned`
stack (both Stages mask, both carry a credit-card label).
`content = 'Contact Jane Roe Card 4111111111111111 end.'`; upstream `gliner_pii` masks the
card at `[22,38)`; downstream `gliner2` (a learned model — its boundaries need not
coincide) emits `[8,26)`. The prior span is dropped whole rather than clipped:
`masked_content == 'Contact [PERSON]111111111111 end.'` — 12 digits of an
already-fully-masked card re-exposed.

**(C) SAME HOLE IN `_merge_floor`, single Redactor.** No chaining needed: one Provider
person span `[8,26)` partially overlapping the floor's `credit_card` `[22,38)` drops the
floor span entirely →
`masked_content == 'Contact [PERSON]111111111111 end.'`. This breaks ADR-0044 on the
plain single-Redactor path.

Uncovered: `packages/latence-core/tests/test_redaction_composition.py` has three tests
(disjoint-span composition, single-redactor byte-identity, stack wiring) — none exercises
an overlapping prior span, so the suite is green over all three leaks.

## What to build

Replace greedy drop-on-overlap with a **coverage-preserving** resolution shared by both
call sites:

- **Masking spans take absolute priority.** Resolve `MASK`/`REPLACE`/`HASH` spans among
  themselves first; a `TAG` span may never displace one.
- **Clip, never drop.** An overlapping span is trimmed to its uncovered tail and kept,
  so no character masked by either input can become unmasked.
- `TAG` spans are then subtracted against the accepted masking coverage (fragments kept
  for audit), preserving the ordered, non-overlapping contract `apply_masking` needs.

## Acceptance criteria

1. Composing an upstream `TAG` span with a downstream `MASK`/floor span leaves the
   financial value masked (scenario A).
2. A partially-overlapping span never un-masks a character the other set masked, in both
   `_compose_prior_spans` (B) and `_merge_floor` (C).
3. **Coverage monotonicity:** for any two span sets, every character masked by either
   input is masked in the composed output.
4. **OFF-BY-DEFAULT byte-identity:** empty `prior` and disjoint/coincident span sets
   produce byte-identical output and span order to today.
5. Output stays ordered and non-overlapping (`apply_masking`'s precondition).
6. Regression tests fail on the pre-fix code and pass after; `make test` stays green at
   or above 3320 passed / 29 skipped.
