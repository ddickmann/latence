# R6-wiring_dead_code-2 — `_compose_prior_spans` drops a partially-overlapping upstream span

**Severity (as reported):** critical
**Disposition:** RESOLVED — already fixed in-round by `0718105` (ticket `R6-invariants-0`).
No new production change required. One regression test added to close the remaining
same-`pii_type` topology.

## Site

- `packages/latence-core/src/latence_core/redaction_policy.py` — `_compose_prior_spans`,
  called from `finalize_redacted_chunk`.
- Consumed by `deploy/env/full-stack/stack.yaml` and `deploy/env/gpu-learned/stack.yaml`
  (`redact_pii` -> `redact_financial`).
- Test blind spot alleged at `packages/latence-core/tests/test_redaction_composition.py`.

## What was reported

The R5-12 chained-redaction fix (`_compose_prior_spans`) unioned this Stage's spans with an
upstream Stage's using a **greedy first-wins by earliest start** rule. A prior span that merely
PARTIALLY overlapped a downstream span was discarded **whole**, so adding a second redaction Stage
UN-MASKED PII the first Stage had already masked. The docstring claimed "a prior span survives only
where this Stage left a gap"; the code did not clip to the gap.

Both committed GPU deploy stacks chain two learned Redactors whose `pii_labels` overlap heavily
(`person`, `email`, `phone number`, `address`, `credit card number` appear in BOTH `redact_pii` and
`redact_financial`), so two models scoring the same entity with different boundaries is the expected
case, not an edge case. The R5-12 regression tests were built on DISJOINT detector sets — exactly
the configuration where the greedy never bites.

## Failure scenario (reproduced)

Driving the real shared seam (`plan_chunk_redaction` + `finalize_redacted_chunk`):

```
content = "Kontakt: Herr Max Mustermann arbeitet hier."
Stage 1 (redact_pii)       accepts person (14,28) = "Max Mustermann"
  -> masked_content = "Kontakt: Herr [PERSON] arbeitet hier."   (correct)
Stage 2 (redact_financial) accepts person (9,17)  = "Herr Max"
  -> sorted by (start, -length): (9,17) first, last_end=17;
     prior (14,28) has char_start 14 < 17 and is DROPPED ENTIRELY
  -> masked_content = "Kontakt: [PERSON] Mustermann arbeitet hier."
```

The surname `Mustermann`, masked before the second Stage existed, lands in the clear in
`records.parquet` / `records.jsonl`. Adding a redaction Stage strictly REDUCES redaction coverage,
and every gate stays green.

Verified against the pre-fix source (`git show 0718105^`): the probe fails with exactly
`'Kontakt: [PERSON] Mustermann arbeitet hier.'`.

## Why this is closed without a production change

Commit `0718105` (`R6-invariants-0`), landed earlier in this same audit round, replaced the greedy
first-wins union in BOTH `_compose_prior_spans` and `_merge_floor` with `_resolve_clipping`: an
overlapping span is CLIPPED to its uncovered tail instead of being dropped, making the resolution
COVERAGE-PRESERVING — every character covered by any input interval stays covered. It additionally
resolves `TAG` spans second (subtracted against accepted masking coverage) so a non-mutating TAG
span can never displace a mask.

That fix already covers this report's topology. Running the reported scenario against HEAD yields
`'Kontakt: [PERSON][PERSON] arbeitet hier.'` — no leak. The existing parametrized test
`test_composition_never_removes_a_masked_character` asserts coverage monotonicity over every overlap
topology, including `prior=(10,20), this=(5,15)` — the reported shape.

The one thing the shipped tests did not assert directly is the report's specific configuration:
BOTH spans carrying the SAME `pii_type` (`person` from two learned Redactors, which is precisely
what the overlapping `pii_labels` of `redact_pii`/`redact_financial` produce). `pii_type` is a
tie-breaker in the composition sort, so that combination is worth pinning explicitly.

## Acceptance criteria

- [x] Reported scenario reproduces on the pre-fix source and does NOT reproduce at HEAD.
- [x] `_compose_prior_spans` clips partially-overlapping prior spans instead of dropping them.
- [x] Composition is coverage-preserving over every overlap topology.
- [x] A regression test pins the same-`pii_type` two-learned-Redactor case from the shipped stacks,
      fails on the pre-fix source, and passes at HEAD.
- [x] `make test` stays at or above 3320 passed / 29 skipped.
