# R1-acceptance_criteria-18 — #187's suites publish no results, and nothing structural can see it

**Severity:** major
**Stream:** feat/release-fleet-integration (#187 benchmark suites, #189 public flip)
**Status:** FIXED (reported symptom already remediated; residual structural hole closed here)

## What was reported

> #187 shipped four new benchmark suites but published ZERO results for them: `docs/BENCHMARKS.md`
> and `docs/benchmarks/results.json` still contain only the Parse slice, while `CHANGELOG.md` and
> `docs/running-the-benchmarks.md` both claim the new suites publish to that page.

## Verification against the tree — the reported symptom is already fixed

Every factual assertion in the report is **false against `feat/release-fleet-integration` as it
stands now**. It was true when the finding was raised; three earlier audit fixes landed in between.

| Reported | Actual, verified |
| --- | --- |
| `docs/BENCHMARKS.md` has exactly one `## … suite` section | **two** — `## Parse — \`parse\` suite` and `## Redaction — \`redaction\` suite` |
| `results.json` `measurements` has a single entry | **two** — the olmOCR parse run and the redaction run |
| No NER/RE/ER/redaction row anywhere | redaction is published headline + per-language + per-PII-type + per-match-criterion |
| No "not yet measured" disclosure | `docs/running-the-benchmarks.md:84` `## Not yet measured` declares the three RED^FM suites, with the repro command |
| CHANGELOG claims the numbers are "rendered on the same page" / "one of the published rows" | corrected: "Their numbers are **not published yet** (RED^FM has to be downloaded first)" |

Remediated by, in order:

- `0795f6d` — ran the redaction suite, published both artifacts, rewrote the guide + CHANGELOG,
  added `test_benchmarks_page_truthfulness.py` (claim-set vs published-set vs parked-set).
- `bc9a816` — corrected the wrong learned-re-pin Provider names baked into `SuiteResult.notes`
  and regenerated both artifacts.
- `d75e981` — added `test_published_artifacts_are_fresh.py`, which rebuilds both artifacts from
  the committed sidecar and demands a byte-identical render.

The three RED^FM suites remain unpublished, which is **legitimate and disclosed**: RED^FM is not
in the repository and must be downloaded. The report itself concedes this, singling out redaction
as the one obtainable offline — and redaction is now published.

## The residual defect this ticket fixes

The guards added above are anchored to **prose**, not to the registry. `_claimed_suites()` in
`test_benchmarks_page_truthfulness.py` scrapes the suite table in `docs/running-the-benchmarks.md`
and treats those rows as the set of suites that exist. `latence_benchmark.suite.suite_names()` —
the actual registry, and the source of truth for what `--suite all` runs — is read by no docs gate
at all (`grep -rn registered_suites packages/*/tests` finds one unrelated CLI test).

So a registered suite that is simply **absent from the guide** is invisible to every gate. That is
precisely the mechanism that let #187 ship four suites with zero published results while the build
stayed green: the docs were never the source of truth, so consistency-with-the-docs could never
detect a suite the docs had never heard of.

## Failure scenario

Delete the `| \`disambiguation\` | …` row from the guide's suite table and leave everything else
untouched. The suite is still registered and still runs under `--suite all`; it is now published
nowhere, claimed nowhere and parked nowhere. Observed:

```
PYTHONHASHSEED=0 uv run pytest \
  packages/latence-core/tests/test_benchmarks_page_truthfulness.py \
  packages/latence-benchmark/tests/test_published_artifacts_are_fresh.py -q
..........                                                               [100%]
```

Ten green. `_claimed_suites()` returns the four surviving rows, `undeclared` is empty, and the
freshness gate only compares committed bytes to a re-render of those same bytes. A sixth suite
added tomorrow and never documented is equally invisible; so is deleting a row to turn the
existing guard green rather than publishing the number.

## What to build

A registry-anchored guard: every name in `suite_names()` must be accounted for in the published
documentation — published on the results page, or explicitly parked under *Not yet measured*.
The registry is the claim set; prose may not shrink it.

## Acceptance criteria

- [x] A test derives the suite set from `latence_benchmark.suite.suite_names()`, not from docs prose.
- [x] Every registered suite is either published (section on `docs/BENCHMARKS.md` **and** a
      measurement in `results.json`) or declared under *Not yet measured* with a repro command.
- [x] The published set is a subset of the registry — the page cannot carry a suite that no
      longer exists.
- [x] The guard fails on the mutation above (a registered suite dropped from the guide) and on an
      unregistered suite appearing on the page.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.
