# R6-wiring_dead_code-3 — ADR-0044 financial floor vs. a mis-boundaried Provider span

**Status: WONTFIX (already fixed) — closed by hardening the test coverage the finding correctly
flagged as missing.**

## The reported defect

> The ADR-0044 UNIVERSAL FINANCIAL-PII FLOOR is bypassable by a Provider's own mis-boundaried span.
> `_merge_floor` (`packages/latence-core/src/latence_core/redaction_policy.py`) resolves
> Provider-vs-floor overlap with a greedy earliest-start-wins rule, so a Provider span that starts
> even ONE character before a floor match and ends inside it suppresses the ENTIRE floor span — the
> IBAN/credit-card/SSN tail is emitted in the clear. The docstring asserts "a floor span survives
> only where the Provider left a gap, which is exactly the guarantee"; the code does not honour that.

Severity as filed: **critical**. Reached from `finalize_redacted_chunk`, on the plain
single-Redactor path every stack runs. This stream (#197/#199) is what makes it live in the blessed
path: it wires the learned `redaction.gliner2` / `redaction.gliner_pii` Providers — whose span
boundaries are model-derived and routinely disagree with the regex floor — into the enterprise-sota
wizard profile and both GPU deploy stacks.

## Failure scenario (as filed)

```
content  = 'Bankverbindung IBAN DE89370400440532013000 bei der Sparkasse.'
floor    = [(20, 42, 'iban', ...)]                       # correct, full IBAN
provider = [(19, 21, 'bank_account_number', ...)]        # 1 char early, ends inside

# greedy earliest-start-wins: provider selected first, last_end=21;
# floor span start 20 < 21 -> discarded WHOLE
masked_content = 'Bankverbindung IBAN[BANK_ACCOUNT_NUMBER]E89370400440532013000 bei der Sparkasse.'
#                                                        ^^^ 21 of 22 IBAN chars in the clear
```

## Why WONTFIX: the defect was already fixed, one round earlier

The verifier read a **pre-`0718105` snapshot**. Commit `0718105`
(`fix(audit-R6)`, audit ticket `R6-invariants-0`) replaced the greedy DROP with a
COVERAGE-PRESERVING **clip** in the shared `_resolve_clipping` helper, which both `_merge_floor` and
`_compose_prior_spans` now route through. Pre-fix body at `0718105^`:

```python
    for span, _source in tagged:
        if span[0] >= last_end:      # <- overlap => DROP the span whole
            result.append(span)
```

Current body:

```python
    for start, end, payload in intervals:
        clipped = max(start, last_end)   # <- overlap => CLIP to the uncovered tail
        if clipped < end:
            result.append((clipped, end, payload))
            last_end = end
```

Probe of the filed scenario against the code as it stands on this branch:

```
floor:      [(20, 42, 'iban', 'DE89370400440532013000', 'floor:iban', 1.0)]
(19, 21) -> [(19, 21, 'bank_account_number', ...), (21, 42, 'iban', 'DE89370400440532013000', ...)]
(15, 39) -> [(15, 39, 'bank_account_number', ...), (39, 42, 'iban', 'DE89370400440532013000', ...)]
```

Every offset the floor claimed stays covered in both variants. The floor is not bypassable.

## What the finding got right, and what was built

The finding's supporting claim is accurate and was worth acting on:

> No test covers a partially-overlapping floor span — `test_redaction_floor.py:143` only covers the
> case where "the floor overlaps exactly".

True **of that file**. The partial-overlap regression coverage added by `0718105`
(`test_provider_span_overlapping_the_floor_does_not_truncate_it`) lives in
`test_redaction_composition.py`, i.e. nowhere near where the floor guarantee is documented — which is
precisely why a careful reader re-derived the defect from scratch. That is a real discoverability
gap in a critical, security-relevant invariant.

Added `test_misboundaried_provider_span_cannot_suppress_the_floor` to
`packages/latence-core/tests/test_redaction_floor.py`, parametrised over three overlap shapes:

| Provider span | shape | fails pre-fix? |
| --- | --- | --- |
| `(19, 21)` | starts ONE char before the IBAN, ends one char inside it (the filed scenario) | yes |
| `(15, 39)` | swallows the `IBAN ` prefix and all but the trailing `000` | yes |
| `(20, 30)` | shares the floor's start but stops short | no — already safe via the length tie-break |

The third case is retained deliberately: it pins the boundary of the guarantee rather than only its
broken half.

## Acceptance criteria

- [x] The filed scenario is asserted end-to-end through `finalize_redacted_chunk` — the seam every
      Provider funnels through — not against `_merge_floor` in isolation.
- [x] The assertion is leak-shaped, not shape-shaped: no suffix of the IBAN of length > 3 may appear
      in `masked_content`, **and** every offset `[20, 42)` the floor claimed must remain covered by
      some non-`TAG` span (coverage-preserving, per ADR-0044).
- [x] Non-vacuous: observed FAILING against the pre-fix greedy-drop resolution, reproducing the
      filed leak string byte-for-byte.
- [x] Fixtures are synthetic only — `DE89370400440532013000` is a published example IBAN, no real
      account data enters the repo (ADR-0044/0042).
- [x] No production code changed; no test weakened or deleted.
- [x] `make test` green at or above the 3320/29 floor.
