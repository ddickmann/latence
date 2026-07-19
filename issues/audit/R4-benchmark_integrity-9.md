# R4 · benchmark-integrity · 9 — the published redaction number is never re-measured

**Severity:** major
**Stream:** feat/release-fleet-integration (tickets #178–#199)
**Files:**
- `packages/latence-benchmark/src/latence_benchmark/results.py` (`_carried_scorer_pin`, `carry_forward`)
- `packages/latence-benchmark/tests/test_published_artifacts_are_fresh.py`
- `packages/latence-core/tests/test_benchmarks_page_truthfulness.py`
- `packages/latence-benchmark/tests/test_every_registered_suite_is_accounted_for.py`

## What is wrong

`docs/BENCHMARKS.md` publishes the redaction headline — **precision 100.0% · recall 77.1%
(27 TP / 0 FP / 8 FN)** — as the framework's central privacy evidence (ADR-0044). That number is
the *only* published benchmark number that reproduces offline in under a second from bytes already
committed: the synthetic PII corpus ships in-package, the pinned Provider is `redaction.hybrid_rule`
at its default config, and nothing is downloaded.

No gate ever re-runs it.

Every gate that reads the published artifacts checks **self-consistency only**:

- `test_published_artifacts_are_fresh` rebuilds `SuiteResults` *from the committed sidecar* via
  `carry_forward([])` and asserts the committed markdown equals a re-render of those same numbers.
  Its own docstring: "it never asserts a particular pass rate, precision or recall."
- `test_benchmarks_page_truthfulness` asserts only that a redaction headline exists and is non-None.
- `test_every_registered_suite_is_accounted_for` compares suite-name sets.

The one anti-restamping mechanism, `ScorerPin`, is scoped to the **unit-test** scorer, and
`_carried_scorer_pin` short-circuits to `None` for any measurement without `splits` — i.e. for
every labelled-set suite. The labelled-set scorer (`metrics.confusion/micro/total/is_partition`
and `redaction_suite.coverage/_alignment`) carries no pin at all. So `carry_forward` republishes
the redaction measurement indefinitely with no validation of the rules that produced it and no
comparison against a live run.

The asymmetry is the tell: the number nobody can re-run without a download (`parse`) is pinned
against scorer drift; the number anyone can re-run in 0.3 s is not re-run by anything.

## Failure scenario

A future commit narrows `redaction.hybrid_rule`'s default recogniser set — exactly the regression
the published run note claims "would show up here as a recall drop" — so real recall falls from
77.1% to ~0.72 with precision still 1.0.

- `test_redaction_suite.py` asserts only loose bounds (`recall > 0.7`, `recall < 1.0`,
  `precision == 1.0`) and a handful of per-type recalls. It never asserts TP/FP/FN counts and never
  reads the page.
- The freshness gate re-renders the **stale sidecar** byte-identically.
- `verify-local.sh` is green.
- `docs/BENCHMARKS.md` keeps publishing "recall 77.1%".

Proven with a probe: editing the sidecar's redaction `language` slices to 35 TP / 0 FN produces a
page reading `**Headline (micro over language):** precision 100.0% · recall 100.0% · F1 100.0%
(35 TP / 0 FP / 0 FN)` that re-renders byte-identically through `render_markdown`/`render_json` and
satisfies every assertion the three gate modules make — while the harness, run live on the same
committed corpus, measures 27 TP / 8 FN.

## What to build

1. Make "this published number can be re-measured here, with nothing to download" a **stated
   property of the suite**, not a name hardcoded in a test: a dataset pinned by `content-digest`
   ships in-package (this is already the criterion `cli._source` uses to pick
   `ShippedCorpusSource`). Expose it as `latence_benchmark.results.reproduces_offline(suite)`.
2. Add a gate that, for **every** published measurement whose suite reproduces offline, re-runs the
   suite live against the shipped corpus and asserts the published record equals what the live run
   measures — every metric slice, the headline, the counts, the stack pin and the notes, not a
   loose bound. This closes the scorer-drift hole for labelled-set suites structurally: any change
   to the scoring rules, the Provider defaults or the corpus moves the live number and turns the
   gate red, which a per-measurement pin could only approximate.
3. The gate must be self-guarding: if no published suite reproduces offline, it fails rather than
   passing vacuously.

## Acceptance criteria

- [ ] `reproduces_offline(suite)` is production code and derives from the registry's `DatasetSpec`,
      so a future in-package suite is gated automatically with no test edit.
- [ ] A gate re-runs every offline-reproducible published suite and compares the **full** published
      record (metrics, headline, documents_scored, document_failures, splits, stack pin, notes)
      against the live measurement, exclusive only of `measured_on`.
- [ ] Falsifying the published redaction numbers in the sidecar turns the new gate RED while the
      three existing gate modules stay green (demonstrated).
- [ ] The gate needs no network and no download, and adds well under a second to `make test`.
- [ ] No existing test is weakened or deleted; `make test` stays at or above 2985 passed, 28 skipped.
