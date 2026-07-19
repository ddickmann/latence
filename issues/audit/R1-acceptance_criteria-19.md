# R1 / acceptance_criteria #19 — the benchmark publisher overwrites the results page with only the suites of one run

- **Severity:** major (as reported)
- **Dimension:** acceptance-criteria / published-artifact integrity
- **Branch:** `feat/release-fleet-integration`
- **Tickets:** #181 (one results page), #187 (Stage benchmark suites)
- **Surface:** `packages/latence-benchmark/src/latence_benchmark/results.py`,
  `packages/latence-benchmark/src/latence_benchmark/cli.py`
- **Status:** **DUPLICATE — already fixed on this branch by commit `0795f6d`
  (ticket `R1-docs_truth-9`). No further code change made.**

## Failure scenario (as reported by the audit)

`write_results` called `markdown_path.write_text(render_markdown(results, ...))` and
`json_path.write_text(render_json(results, ...))` with only the current invocation's `results`,
with no read-merge against the existing page. `DEFAULT_MARKDOWN = Path("docs/BENCHMARKS.md")` and
`DEFAULT_JSON = Path("docs/benchmarks/results.json")` point straight at the published artifacts,
so any partial run silently deleted every other Stage's published measurements.

Concretely: the page's own `### Reproduce` block tells a reader to run
`latence-benchmark run --suite parse --provider parser.pdfplumber`. Following that documented
command wiped the Redaction / Entity Extraction / Relation Extraction / Disambiguation sections
out of both artifacts, with no warning and no error. `--suite all` was the only invocation that
produced a correct page, and nothing enforced it — making the "one results page, every Stage"
contract of #181/#187 structurally unachievable.

## What to build

A partial run must **republish** the previously published measurements for the Stages it did not
score, rather than deleting them, with each measurement carrying the date it was actually
produced so an assembled page never implies a carried number was measured today.

## Acceptance criteria

1. A run covering a subset of the suites leaves the other suites' published measurements present
   in both `docs/BENCHMARKS.md` and `docs/benchmarks/results.json`.
2. A carried-forward measurement is republished **byte-for-byte**; it must not drift as a side
   effect of another suite's run.
3. Each suite section states the date its own numbers were measured, so a carried number is never
   attributed to today's run.
4. A carried number is validated against the suite registry before republication: if its dataset
   has since been re-pinned, the run is **refused** with a typed error (ADR-0034: no silent
   errors), and the sidecar is left untouched rather than half-rewritten.
5. Deliberately dropping the other Stages remains possible, but only via an explicit opt-out flag.
6. Output ordering is stable so two equivalent runs write byte-identical artifacts (DETERMINISM).

## Resolution — why no change was made

Every criterion above is already implemented and covered on this branch:

- `latence_benchmark.results.carry_forward` (`results.py:172-191`) merges the current run's
  results with the sidecar's previously published measurements, current results winning, output
  sorted by suite name (AC 1, 2, 6).
- `results._carried` (`results.py:101-169`) rebuilds each carried measurement from the **registry**
  `DatasetSpec`, not the sidecar, and raises the typed `PublishedResultsError` on an unregistered
  suite, a re-pinned dataset revision, or a malformed record (AC 4).
- `cli.main` applies `carry_forward` **by default**; the opt-out is `--no-carry-forward`
  (`cli.py:97-106`, `212-218`), and the run reports which suites were republished (AC 5).
- `render_markdown` emits a per-section `_Measured <date>._` line from `SuiteResult.measured_on`
  (`results.py:292-293`), and the CLI stamps `measured_on` only on freshly-run suites (AC 3).

Regression coverage that pins the behaviour:

- `test_cli_all_suites.py::test_a_partial_run_republishes_the_other_stages_instead_of_deleting_them`
- `test_cli_all_suites.py::test_no_carry_forward_writes_a_single_suite_page`
- `test_cli_all_suites.py::test_republishing_a_number_against_a_re_pinned_dataset_is_refused`

The first of these was verified non-vacuous for **this** ticket's exact failure scenario: with the
`if args.carry_forward:` guard in `cli.py` forced to `if False:` (restoring the pre-fix
overwrite), it fails with

```
>       assert set(published) == {"parse", "redaction"}
E       AssertionError: assert {'redaction'} == {'parse', 'redaction'}
E         Extra items in the right set:
E         'parse'
```

and passes with the guard restored. Writing a second test asserting the same invariant would be
duplicate coverage, so this ticket is closed as a duplicate rather than re-fixed.
