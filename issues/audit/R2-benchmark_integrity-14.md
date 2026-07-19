# R2 · benchmark integrity · 14 — the published `match` MICRO row aggregates non-disjoint criteria

**Severity:** major
**Site:** `packages/latence-benchmark/src/latence_benchmark/results.py:243` (`total = micro(slices, dimension=dimension)`)
**Published symptom:** `docs/BENCHMARKS.md`, section `#### By match criterion`

## What is wrong

`_metric_tables` renders one precision/recall/F1 table per metric dimension and then, for
*every* dimension unconditionally, appends a bold `**MICRO**` row computed by
`metrics.micro`, which sums the slices' confusion counts.

Micro-averaging is only defined when the slices **partition** one gold set — disjoint buckets
whose gold counts add up to the corpus (`language`, `pii_type`, `predicate`). The `match`
dimension is not that. It holds *alternative scoring criteria* applied to the **same** gold
items: how strictly a detection has to line up with a gold span to be credited. Summing them
counts every gold item once per criterion.

## Failure scenario

The redaction corpus has **35** gold PII spans. The `match` dimension carries three overlapping
criteria over those same 35 spans (`coverage (value fully masked)`, `exact span`,
`exact span+type`). The renderer sums their confusion counts and publishes:

```
| **MICRO** | **91.4%** | **70.5%** | **79.6%** | 74 | 7 | 31 | 105 |
```

A Gold column of **105** for a 35-span corpus, and a "91.4% precision" that is an arithmetic
blend of three mutually redundant measurements of the same detections. It is the only bold row
in the section, so it reads as the section's authoritative summary — while corresponding to no
measurable quantity at all.

The same latent bug is queued for the not-yet-published entity suite: `redfm_suites.py`
`EntitySuite` emits `span (boundaries only)` and `span+type` both under dimension `match`, so
publishing that suite would double-count every RED^FM gold span in its MICRO row.

`test_published_artifacts_are_fresh.py` cannot catch this — it only asserts the committed bytes
equal a re-render, so it locks the wrong row in rather than flagging it.

## What to build

1. Make "is this dimension a partition of one gold set?" an explicit, checkable property of the
   dimension rather than an unstated assumption of the renderer. `match` is a **criteria**
   dimension; `language` / `pii_type` / `predicate` are partitions.
2. `results._metric_tables` emits the aggregate row only for partition dimensions. For a
   criteria dimension it emits, in the row's place, a sentence stating why no aggregate exists
   (the slices measure the same gold items under different rules).
3. Asking for an aggregate over a criteria dimension is a typed, loud error — never a silently
   wrong number (ADR-0034). `SuiteResult.headline` goes through the same guard, so a suite that
   ever names a criteria dimension as its headline fails instead of publishing a fiction.
4. Regenerate `docs/BENCHMARKS.md` from the committed sidecar so the published page carries the
   corrected section.

## Acceptance criteria

- [ ] No `**MICRO**` row appears under `#### By match criterion` in `docs/BENCHMARKS.md`, and the
      number `105` no longer appears as a Gold total for the 35-span corpus.
- [ ] The per-criterion rows themselves are unchanged — the fix removes a fabricated aggregate,
      it does not alter any measurement.
- [ ] Partition dimensions (`language`, `pii_type`, `predicate`) still get their MICRO row, with
      identical numbers.
- [ ] Requesting an aggregate over a criteria dimension raises a typed error.
- [ ] A regression test fails on the pre-fix renderer and passes on the fixed one.
- [ ] `make test` stays green at or above 2985 passed / 28 skipped.
