# R3-benchmark_integrity-13 — The "By PII type" table drops every false positive whose predicted type is not a gold type

**Severity:** major
**Area:** benchmark integrity / published numbers
**Location:** `packages/latence-benchmark/src/latence_benchmark/redaction_suite.py` (`RedactionSuite.run`, the `per_type` construction)
**Rendered by:** `packages/latence-benchmark/src/latence_benchmark/results.py` `_metric_tables` (bold `**MICRO**` row, because `is_partition("pii_type")` is True)

## What is wrong

`RedactionSuite.run` builds the per-type slices by iterating the **gold** type inventory only:

```python
gold_types = sorted({item[1] for item in gold})
per_type = [
    coverage("pii_type", t, gold=[g for g in gold if g[1] == t],
             detected=[d for d in detected if d[1] == t])
    for t in gold_types
]
```

A detection whose `pii_type` never appears in the gold set therefore belongs to no slice at all.
It is not scored as a false positive anywhere in the `pii_type` dimension — it is simply not
counted. Because `pii_type` is a partition dimension, `results._metric_tables` renders a bold
`**MICRO**` aggregate over exactly those slices, so the published aggregate **understates false
positives and overstates precision** by however much the detector over-redacted under an
out-of-inventory label.

Two published claims are false as a consequence:

* the `**MICRO**` row of the "By PII type" table (inflated precision / F1), and
* the run note *"Per-type slices count a detected span as a false positive for its own type when
  it covers no gold value"* — which does not hold for a type absent from the gold set.

## Failure scenario (observed)

A one-document corpus whose gold set contains only an `email` span, in text that also contains
`192.0.2.44`. `redaction.hybrid_rule` at the pinned default config emits 2 spans (`email`, `ip`);
`ip` is not a gold type. Observed on the pre-fix code:

```
language: [('en', TP 1, FP 1, precision 0.5)]
pii_type: [('email', TP 1, FP 0, precision 1.0)]
pii_type MICRO: TP 1  FP 0  precision 1.0
```

The `language` dimension correctly charges the over-redaction; the `pii_type` dimension erases it
and publishes 100% precision.

The shipped 2026-07-19 run is unaffected only by luck: the detected types on the shipped corpus
(`api_key, aws_key, credit_card, email, iban, ip, jwt, phone, secret, ssn`) happen to be a strict
subset of the 14 gold types. The re-pin path documented in the same run notes
(`--provider redaction.gliner_pii`), or any detector emitting a label outside the gold inventory
(`URL`, `DATE_TIME`, `LOCATION`, `NRP`, …), publishes an inflated per-type precision.

No test in `packages/latence-benchmark/tests` covered a detected type absent from the gold set.

## Acceptance criteria

1. The `pii_type` slices are built over the union of gold types **and** detected types, so every
   emitted detection is charged to exactly one slice of the dimension.
2. A type present only in the detections renders as its own slice with TP 0 / FP n / FN 0
   (precision 0%, recall `n/a` — an undefined ratio, never 0%).
3. The `pii_type` `**MICRO**` false-positive count equals the total number of unmatched
   detections, matching what the `language` dimension already reports for the same run.
4. A regression test exercises a corpus where the pinned Provider emits a type outside the gold
   inventory, and fails on the pre-fix code.
5. `make test` stays green at or above 2985 passed / 28 skipped; no published number on the
   shipped corpus changes (the shipped detected types are a subset of the gold types, so the
   fix is byte-identical there).
