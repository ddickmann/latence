# R3-console_readonly-11 — The chunk inspector calls the redaction control "OFF" on a chunk the ADR-0044 floor actually masked

**Severity:** major
**Area:** `packages/latence-console/src/latence_console/browse.py` (`_redaction_disposition`)
**Invariants touched:** ADR-0044 (universal redaction floor), ADR-0034 (no silent/false reporting)

## What is wrong

`_redaction_disposition` picks the disposition summary with an `if/elif` chain whose FIRST
branch is `if chunk.redaction_disabled_for_sensitive:`. That branch is therefore taken
unconditionally whenever the H-C1 §4 no-op flag is set — before the `elif spans:` branch that
reports the PII spans that were actually detected and handled.

Since ADR-0044 the two are no longer mutually exclusive. The universal financial-PII floor
(`ssn` / `credit_card` / `iban`) runs on every chunk where redaction runs, regardless of what
the configured policy allow-list enables. So a confidential chunk whose policy allow-list
intersects no enabled detector gets BOTH `redaction_disabled_for_sensitive=True` AND a
non-empty `pii_spans` list, with `masked_content` set.

The console then renders, on its PII-audit surface:

> This chunk is classified confidential but the redaction policy ran no detectors. An empty
> PII list here means the control was OFF — not that the chunk is clean.

directly above a table listing the IBAN it masked. Both sentences are false for that chunk:
detectors did run (the floor), and the PII list is not empty. An auditor asking "was this
confidential chunk redacted?" is told the control was off when the framework's strongest
redaction guarantee had in fact fired.

## Failure scenario (reproduced end to end)

A real `latence run` with provider `redaction.hybrid_rule`, no fixtures, a `confidential`
document containing an IBAN, and `policies: {confidential: {action: mask, types:
[passport_number]}}` yields a `ChunkRecord` with:

- `redaction_disabled_for_sensitive=True`
- `masked_content='... IBAN [IBAN] before the deadline.'`
- `pii_spans=[iban 53–75, detector `floor:iban`, action mask]`

The chunk page for that record renders the "control was OFF" summary above the span table
showing `iban / floor:iban / mask / [IBAN] / 53–75 / 55–77`.

No test exercised `redaction_disabled_for_sensitive=True` together with non-empty `pii_spans`.

## Acceptance criteria

1. When the no-op flag is set AND spans exist, the summary states the truth: the CONFIGURED
   policy ran no detectors, the listed span(s) were caught by the universal financial-PII
   floor and handled, and anything outside the floor's scope was not looked for.
2. The summary in that case must NOT claim "an empty PII list", and must not assert the
   control was OFF full stop.
3. The pure no-op case (flag set, no spans) keeps its existing loud "control was OFF" wording
   verbatim — no regression of the H-C1 §4 signal.
4. A regression test drives `browse.inspect_chunk` (and the rendered page) with a chunk in the
   floor-fired-under-a-disabled-policy state and fails on the old code.
5. `make test` stays green at or above 2985 passed / 28 skipped.
