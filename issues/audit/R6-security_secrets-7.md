# R6-security_secrets-7 — WONTFIX (already fixed by R6-invariants-0) + coverage hardened

**Severity as reported:** critical
**Verdict:** NOT A LIVE DEFECT on `feat/release-fleet-integration` @ HEAD. The reported hole was
closed earlier in this same audit round by commit `0718105` (ticket `R6-invariants-0`). Filed here
for traceability, with one genuinely additive regression test landed.

**Site:** `packages/latence-core/src/latence_core/redaction_policy.py` — `_merge_floor`
(reached from `finalize_redacted_chunk`).

## What was reported

The ADR-0044 universal financial-PII floor is bypassable: `_merge_floor` discards a floor candidate
whenever a Provider span merely STARTS BEFORE it, even if the Provider span ENDS INSIDE the floor
match — so part of a credit card / IBAN / SSN reaches the exported corpus despite the floor claiming
to be non-bypassable by model choice.

Failure scenario: chunk content `"Karte 5555 5555 5555 4444 wurde belastet."`. `floor_candidates`
finds `credit_card` at `[6,25)`. A learned Redactor (gliner / gliner2 / presidio — all funnel through
this seam) returns `[0,20)` `"Karte 5555 5555 5555"`: it swallowed the leading label token and
stopped one digit group short, an entirely ordinary NER boundary disagreement. The pre-fix greedy
earliest-start-wins DROP sorted the Provider span first, set `last_end = 20`, and rejected the floor
span because `6 < 20`, yielding `masked_content == '[CREDIT_CARD] 4444 wurde belastet.'` — the card's
last four digits exported in the clear.

## Why it is WONTFIX

The analysis of the *pre-fix* code is correct, but it describes code that no longer exists. Commit
`0718105` (`fix(audit-R6)`, ticket `R6-invariants-0`) replaced the greedy DROP in BOTH `_merge_floor`
and `_compose_prior_spans` with `_resolve_clipping`, a coverage-preserving resolution that CLIPS an
overlapping span to its uncovered tail instead of discarding it. The verifier evidently reproduced
against a tree predating that commit.

Re-run of the exact reported scenario against HEAD:

    _merge_floor(
        accepted=[(0, 20, "person", "Karte 5555 5555 5555", "gliner", 0.9)],
        floor=[(6, 25, "credit_card", "5555 5555 5555 4444", "floor:credit_card", 1.0)],
    )
    -> [(0, 20, "person", ...), (20, 25, "credit_card", "5555 5555 5555 4444", "floor:credit_card", 1.0)]

The floor span survives as its clipped remainder `[20,25)`. End-to-end through
`finalize_redacted_chunk`, `masked_content == '[PERSON][CREDIT_CARD] wurde belastet.'` — no digit of
the card reaches the exported corpus. The reported string `'[CREDIT_CARD] 4444 wurde belastet.'` does
not reproduce.

## What was nonetheless built

The claim that "no test constructs a provider span that starts before and ends inside a floor match"
was ALSO already false at HEAD: `test_misboundaried_provider_span_cannot_suppress_the_floor` (added
by the same commit) parametrizes exactly that shape. But its coverage is `iban`-only. The reported
scenario is the `credit_card` detector with a label-token prefix, which nothing pinned.

Landed: `test_label_prefixed_provider_span_cannot_leak_a_card_tail` in
`packages/latence-core/tests/test_redaction_floor.py` — the credit_card twin of the IBAN cases,
driving the reported chunk end-to-end through `finalize_redacted_chunk`.

## Acceptance criteria

- [x] The reported scenario is proven non-reproducing at HEAD, end-to-end through the real seam
      (not by reading the source).
- [x] A regression test pins the `credit_card` partial-overlap case (Provider span starts before,
      ends inside the floor match) at `finalize_redacted_chunk`, asserting no tail of the card
      survives in `masked_content` AND that every floor offset stays covered by a mutating span.
- [x] The test is NON-VACUOUS: it fails when `_merge_floor`'s clipping is reverted to the pre-fix
      greedy drop, with the exact reported leak
      (`AssertionError: ... leaked card tail '4444': '[PERSON] 4444 wurde belastet.'`), and passes
      on HEAD.
- [x] The test guards itself against vacuity by first asserting the floor sees the whole card at
      `[6,25)`.
- [x] Fixture data is synthetic only (`5555 5555 5555 4444` is a published Mastercard test number) —
      ADR-0044: no real PII in the repo.
- [x] No production source change; no test weakened or deleted; suite does not regress.
