# R3-benchmark_integrity-12 — carried-forward measurements are re-labelled by today's scorer

**Severity:** major
**Area:** `packages/latence-benchmark/src/latence_benchmark/results.py`, `scoring.py`
**Status:** FIXED

## What is wrong

`render_markdown` reads the module-level `SUPPORTED_TYPES` (imported from
`latence_benchmark.scoring`) **at render time** and prints it as
`**Scored test classes:** …` next to numbers that may have been produced by a *different*
scorer. `carry_forward` / `_carried` validate only `measurement["dataset"]["revision"]` against
the registry — the sidecar records nothing at all about the scorer that produced the numbers.

So a change to the scoring code silently **re-labels** stale numbers instead of invalidating
them. The dataset-revision guard exists precisely to stop "attributing an old number to new
data"; the same hole is wide open for "attributing an old number to new scoring rules", which
changes the *meaning* of the number just as completely.

## Failure scenario (observed)

Rebuild the committed measurements with `carry_forward([], json_path=docs/benchmarks/results.json)`
and render twice, the second time with `SUPPORTED_TYPES` extended by `UnitTestType.TABLE`
(simulating a future commit that implements table scoring):

* The Parse `OVERALL` row is byte-identical — still `20.0%`, 524 / 2090 / 2614, 4405
  unsupported, i.e. produced by the scorer that treated `table` as **unsupported**.
* The page now reads `**Scored test classes:** absent, baseline, order, present, table.`
  immediately followed by `` `table` and `math` assertions are counted in the Unsupported
  column`` — a self-contradiction, published as reproducible evidence.

`test_published_artifacts_are_fresh.py` cannot see this: it re-renders the same sidecar with the
same new code, so it would actively *demand* the self-contradictory page be committed. Every
gate stays green and no re-run is required.

The same hole applies to any change to `fuzzy_find`, `normalise`, `BASELINE_MIN_CHARS` or
`_score_baseline`: the published numbers keep their old values while the page keeps asserting
they are reproducible from the committed config.

## What to build

1. A **scorer pin** recorded in every unit-test measurement, alongside the existing `StackPin`:
   the supported test classes, `BASELINE_MIN_CHARS`, and a digest over the *semantics* of the
   scoring functions (docstring- and formatting-insensitive, via `ast.unparse`), so an edit to
   `fuzzy_find`/`normalise`/`_score_baseline`/`score` changes the fingerprint automatically and
   a comment rewrite does not.
2. `_carried` must validate the recorded scorer pin against the scorer of today, exactly as it
   validates the dataset revision — refusing with a typed `PublishedResultsError` that names the
   re-run command. A sidecar measurement that reports unit-test splits but carries **no** scorer
   pin is refused too: it predates the pin and cannot be vouched for.
3. `render_markdown` must describe each measurement in terms of **that measurement's own**
   scorer pin, never a module-level constant read at render time.

## Acceptance criteria

* `SuiteResult` carries `scorer_pin: ScorerPin | None`; unit-test suites populate it, labelled-set
  suites (RED^FM, redaction) leave it `None`. `as_dict` publishes it.
* Extending/shrinking `SUPPORTED_TYPES`, changing `BASELINE_MIN_CHARS`, or changing the body of
  `normalise` / `fuzzy_find` / `fuzzy_contains` / `_score_baseline` / `score` changes
  `scorer_pin().fingerprint`; editing only a docstring, comment or formatting does not.
* `carry_forward` over a sidecar whose scorer pin no longer matches raises
  `PublishedResultsError` naming the suite and the `--suite <name>` re-run.
* A sidecar measurement with `splits` and no `scorer_pin` is refused.
* The page's "Scored test classes" / unsupported-classes sentence is derived from the carried
  measurement's own pin, so a future scorer change cannot re-label a stale number.
* `docs/BENCHMARKS.md` stays byte-identical for the current pin; `docs/benchmarks/results.json`
  gains the `scorer_pin` record.
* Regression test fails on the old code and passes on the fix; `make test` stays green.
