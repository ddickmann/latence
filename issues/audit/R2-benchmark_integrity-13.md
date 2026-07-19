# R2-benchmark_integrity-13 — The redaction headline cannot detect over-redaction: `coverage` mixes measurement bases

**Severity:** major
**Stream:** feat/release-fleet-integration (#187 benchmark suites)
**Status:** FIXED

## What is wrong

`coverage` in `packages/latence-benchmark/src/latence_benchmark/redaction_suite.py:179-201`
scored its two halves on **different bases**:

```python
covered      = [g for g in gold     if any(_covers(d, g) for d in detected)]   # GOLD items
false_alarms = [d for d in detected if not any(_covers(d, g) for g in gold)]   # DETECTED spans
```

`true_positives` counted gold values, `false_positives` counted detected spans, and nothing tied
one detection to one value. A single detected span could therefore claim an unbounded number of
gold values for free, so **widening a mask was never charged**. Over-redaction — the single
largest failure mode of a redactor, and the one that destroys corpus utility — was invisible to
the metric.

This is the number the framework's central privacy claim rests on: it is the headline of
`docs/BENCHMARKS.md` ("precision 100.0% · recall 77.1% · F1 87.1%") and is repeated in
`CHANGELOG.md` as "100.0% precision / 77.1% coverage recall".

The module that defines the harness's other confusion metric already states the correct rule —
`latence_benchmark/metrics.py`: *"Matching is multiset matching … so a Provider that emits the
same span twice is charged one false positive instead of being silently forgiven"*. `coverage`
was the one scorer that broke it.

`docs/running-the-benchmarks.md:169` discloses the mirror-image hazard ("a system that predicts
nothing would score a perfect 'no false positives'") while leaving the predict-everything hazard
undisclosed.

## Failure scenario

Substitute a degenerate detector that emits one span `(doc_id, "person_name", 0, len(text))` per
document — i.e. it masks every document in full and destroys 100% of corpus utility. Every such
span contains at least one gold value in the 14 non-clean documents, so `false_alarms` is empty
for them; only the 6 clean control documents produce a false positive:

| detector | TP | FP | FN | precision | recall | **F1** |
| --- | --- | --- | --- | --- | --- | --- |
| pinned `redaction.hybrid_rule` | 27 | 0 | 8 | 100.0% | 77.1% | **87.1%** |
| mask-the-whole-document | 35 | 6 | 0 | 85.4% | 100.0% | **92.1%** |

The redactor that masks everything **outranks the real one on the published headline.**

No test covered this: `test_coverage_credits_a_wider_mask_but_not_a_partial_one` and
`test_a_boundary_disagreement_is_not_also_charged_as_a_false_alarm` deliberately assert only
that a wider mask is *credited*; neither asserts that it is *bounded*.

## What to build

1. Put both halves of `coverage` on one basis by **aligning** detections to gold values: a
   maximum one-to-one matching under the existing `_covers` containment predicate. TP = matched
   pairs, FN = gold values left unmatched, FP = detections left unmatched. This is exactly the
   rule `metrics.confusion` already applies, with containment substituted for equality, and it
   introduces no threshold or tuning constant.
2. Keep the deliberate properties intact: a wider mask still fully credits the value it covers,
   a partial mask still credits nothing, and a boundary disagreement is still not charged twice.
3. Disclose the over-redaction hazard in `docs/running-the-benchmarks.md` and in the suite's own
   run notes, alongside the predict-nothing hazard already documented.
4. Regression test that a mask-the-whole-document detector cannot outrank the pinned Provider.

## Acceptance criteria

- [x] `coverage` counts true positives, false positives and false negatives in one unit
      (aligned pairs / unmatched golds / unmatched detections); no detected span is credited for
      more than one gold value.
- [x] A mask-everything detector scores strictly WORSE than the pinned `redaction.hybrid_rule`
      on the headline F1; a regression test asserts it and FAILS on the pre-fix source.
- [x] The three pre-existing scoring-rule tests still pass **unmodified** (wider mask credited,
      partial mask not, clean-text detection is a false positive, boundary slop not double-charged).
- [x] The published pinned numbers are unchanged (27 TP / 0 FP / 8 FN — 100.0% / 77.1% / 87.1%),
      so `docs/BENCHMARKS.md` and `docs/benchmarks/results.json` need no renumbering; the fix
      removes a loophole rather than moving the measurement.
- [x] `docs/running-the-benchmarks.md` discloses the over-redaction hazard.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.
