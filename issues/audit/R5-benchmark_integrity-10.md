# R5-benchmark_integrity-10 — A parse failure moves `table`/`math` tests into the pass-rate denominator

**Severity:** major
**Stream:** feat/release-fleet-integration (#187 benchmark suites)
**Status:** FIXED

## What is wrong

`ParseSuite.run` (`packages/latence-benchmark/src/latence_benchmark/parse_suite.py:166-182`)
has two document-level failure branches — the referenced-PDF-missing branch and the
`Disposition.PARSE_ERROR` branch. Both do:

```python
results.extend(
    TestResult(t.id, split, t.type, Outcome.FAILED, ...) for t in split_tests
)
```

over **all** of that document's tests, bypassing `scoring.score()` and therefore bypassing the
single place the harness implements its published rule:

```python
if test.type not in SUPPORTED_TYPES:   # scoring.py:175
    return TestResult(..., Outcome.UNSUPPORTED, ...)
```

So a `table` or `math` test — a class this harness slice cannot score — is UNSUPPORTED
(outside the pass-rate denominator) when its PDF parses, and FAILED (inside the denominator)
when the same PDF does not. The Parse suite's denominator therefore depends on which documents
the pinned Provider happened to parse.

This contradicts, in code, the guarantee the repo publishes in three places:

- `docs/running-the-benchmarks.md:71` — unsupported classes are "excluded from the pass-rate
  denominator ... never counted as passes";
- `CHANGELOG.md:191-192`;
- the page text rendered by `results.py:441-446`.

## Failure scenario

A split with one `math`, one `table` and one `present` test on a single PDF:

- (a) PDF absent → `SplitScore(passed=0, failed=3, unsupported=0)`
- (b) PDF present but unparseable → `SplitScore(passed=0, failed=3, unsupported=0)`,
  `document_failures=1`

Expected in both cases: `failed=1, unsupported=2`.

Real magnitude at the pinned revision `54a96a6f…`: 4,405 of 7,019 tests are `math`/`table`,
and 746 of the 1,403 PDFs carry almost nothing else (522 `arxiv_math` + 36 `old_scans_math`
carry ONLY math tests; 188 `table_tests`). `docs/BENCHMARKS.md` and
`docs/running-the-benchmarks.md` explicitly recommend re-pinning to `--provider parser.lighton`
to compare an OCR Provider. If that GPU Provider raises on even 10 `arxiv_math` PDFs, ~56
unscoreable math tests are injected into that run's denominator as failures — while the exact
same tests are excluded from the committed `parser.pdfplumber` denominator (2,614). The two
OVERALL pass rates are then computed over different denominators and published in the same
column of the same page, making the cross-Provider comparison the docs recommend invalid.

The committed 2026-07-18 pdfplumber number is **not** contaminated (its 7 parse failures all
landed on PDFs carrying scoreable tests — `arxiv_math` and `old_scans_math` both show
`failed=0` in `docs/benchmarks/results.json`), so this is a latent defect no green gate can
see: the only re-measurement gate,
`test_published_offline_numbers_are_remeasured.py`, covers content-digest suites only and never
runs Parse. No published artifact changes as a result of this fix.

## What to build

1. Put the supported/unsupported rule in ONE place. Add
   `scoring.score_document_failure(test, split=…, reason=…)`: FAILED for a scoreable class
   (a parser must never be able to raise its way to a better number — that property stays), and
   for a class this slice cannot score, **delegate to `score()`** so the UNSUPPORTED verdict and
   its reason string have exactly one implementation.
2. Use it from both document-level failure branches in `ParseSuite.run`.
3. Regression test that fails on the pre-fix code.

`score()`'s body is left byte-identical on purpose. `ScorerPin.algorithm_digest` is a sha256
over the docstring-free ASTs of `_SCORING_BEHAVIOUR` (which includes `score`), and
`results.carry_forward` refuses to republish a measurement whose scorer pin has moved. An
early draft that merely factored `score()`'s UNSUPPORTED branch into a helper flipped the digest
to `37be24761a363d84` and broke `test_published_offline_numbers_are_remeasured.py` at
collection — forcing a full 1,403-PDF re-run of a measurement this fix provably does not change.
`score_document_failure` is deliberately NOT added to `_SCORING_BEHAVIOUR`: it introduces no
verdict of its own (its UNSUPPORTED branch *is* `score`, its FAILED branch is the pre-existing
"a parse failure is a failure" rule that `ScorerPin.supported_types` already describes). Digest
after the fix: `b2221b17e5b69df4`, unchanged from `docs/benchmarks/results.json`.

Proof the committed pdfplumber measurement is numerically identical under the fix: its per-split
`unsupported` sums to 2927 + 458 + 1020 = **4405**, which is the full count of `math`/`table`
tests at the pinned revision. Not one of them was converted into a failure, so no published
number moves.

## Acceptance criteria

- [x] Neither document-level failure branch constructs `Outcome.FAILED` directly; both route
      through the scoring module so the `SUPPORTED_TYPES` rule cannot be bypassed.
- [x] A missing PDF whose tests include `math`/`table` yields `unsupported > 0` and those tests
      stay out of `SplitScore.scored`.
- [x] A `PARSE_ERROR` document behaves identically.
- [x] The anti-gaming property is preserved: every scoreable test of a failed document is still
      FAILED, and `document_failures` still counts the document.
- [x] Regression test `test_a_failed_document_cannot_move_unsupported_tests_into_the_denominator`
      FAILS on the pre-fix source and PASSES after.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.

## Residual (not in scope, logged)

The `PARSE_ERROR` branch emits no `notes` entry at all (the missing-PDF branch does). A reader
of the results page sees `document_failures: 7` but no prose. Adding a note changes published
artifact text, so it is left for a separate ticket.
