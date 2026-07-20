# R7 — benchmark_integrity-9 — the ScorerPin fingerprint is blind to half the scoring rules

**Severity:** major
**Area:** `packages/latence-benchmark/src/latence_benchmark/scoring.py`,
`packages/latence-benchmark/src/latence_benchmark/results.py`
**Status:** FIXED

## What is wrong

`ScorerPin` exists for exactly one purpose: to refuse republishing a published pass rate whose
scoring rules have changed since the number was measured
(`results._carried_scorer_pin` → `carry_forward`). Its `algorithm_digest` is a sha256 over the
docstring-free ASTs of `_SCORING_BEHAVIOUR`, a hand-maintained tuple of five functions.

That tuple is missing rules that *are* the metric:

1. **`score_document_failure`** (`scoring.py:237`) — excluded by an explicit comment claiming it
   "introduces no verdict of its own". That claim is false, and was made false by the R5 audit fix
   (`fa35f06`) which moved the document-failure rule *into* this module and changed it: a
   `table`/`math` test of a `PARSE_ERROR` / missing-PDF document used to be counted `FAILED`
   (inside the pass-rate denominator) and now stays `UNSUPPORTED` (outside it). That is a
   different denominator, i.e. a different published number — and the fingerprint does not move.
2. **`_WHITESPACE` (`:45`) and `_REPLACEMENT_CHAR` (`:59`)** — module constants read *inside*
   fingerprinted functions but defined outside them, so `ast.unparse` of the function bodies is
   identical whatever their values are. `_WHITESPACE` decides what `normalise` collapses;
   `_REPLACEMENT_CHAR` decides what a `baseline` row rejects.
3. The same hole, structurally: `_required` and `_window` are behaviour-bearing helpers called by
   `score` (the `first_n`/`last_n` window is a scoring rule) and were never in the tuple either.

The root cause is that `_SCORING_BEHAVIOUR` is a **manually curated list**, so every future helper
or constant is opted *out* of the fingerprint by default — the wrong default for a mechanism whose
job is to notice change.

## Failure scenario

Against a synthetic olmOCR-bench root with one missing-PDF document carrying one `present`, one
`table` and one `math` test:

* post-R5 rule → `table_tests` SplitScore = passed 0 / failed 1 / unsupported 2, **scored = 1**
* pre-`fa35f06` rule → passed 0 / failed 3 / unsupported 0, **scored = 3**

Two different pass-rate denominators, one identical scorer fingerprint `7e2cca8a2a0f7591` — which
is the fingerprint recorded for the Parse measurement in `docs/benchmarks/results.json`. Mutating
`scoring._WHITESPACE` / `scoring._REPLACEMENT_CHAR` likewise leaves the fingerprint unmoved.

Consequence: `carry_forward` republishes the Parse measurement and `render_markdown` describes it
with today's rules, because the pin validates. The published counts happen to coincide only
because none of the 7 document failures fell in a split containing `table`/`math` tests
(`arxiv_math` and `old_scans_math` both show `failed = 0`, `table_tests` keeps `unsupported =
1020`) — luck, not a guard.

## What to build

Stop curating the fingerprint by hand. `_SCORING_BEHAVIOUR` becomes the *entry points* of the
metric; the digest is taken over the **transitive closure** reachable from them:

* every module-level function they call (so `_required`, `_window`, and anything added later are
  covered automatically), and
* every module-level constant they read, canonicalised deterministically (regex → pattern+flags,
  set/frozenset → sorted, everything else → `repr`).

`score_document_failure` joins the entry points, and the false comment is deleted.

Source parsing and closure discovery stay cached (source cannot change in-process); the **constant
values are read on every call**, so `scorer_pin()`'s documented promise — "the constants are read
on every call rather than frozen at import, so a slice of the scorer that is swapped out is
described by what it actually is" — becomes true of the digest as well, not just of the two
explicit pin fields.

## Published-artifact migration

Fixing the fingerprint necessarily changes its value, so the `algorithm_digest` recorded for the
Parse measurement in `docs/benchmarks/results.json` is updated in the same commit. This is not a
silent restamp:

* the pin the sidecar carried was computed **before** `fa35f06`, when the document-failure rule
  still lived in `parse_suite.py` — outside the fingerprinted module entirely — so no fingerprint
  definition, correct or not, could have covered it at measurement time;
* the published per-split counts are unchanged by the R5 rule: it can only convert `table`/`math`
  verdicts, and the two all-math splits report `failed = 0` / `passed = 0` while `table_tests`
  keeps 1020 unsupported and only 2 scored rows;
* `docs/BENCHMARKS.md` is byte-identical either way — the page renders the *stack* pin
  fingerprint, never the scorer digest.

Re-running the Parse suite against olmOCR-bench (network required, not available in the merge
gate) remains the authoritative confirmation and is noted here for traceability.

## Acceptance criteria

1. Reverting `score_document_failure` to its pre-`fa35f06` body (all tests `FAILED`) **moves** the
   scorer fingerprint.
2. Mutating `scoring._WHITESPACE` or `scoring._REPLACEMENT_CHAR` **moves** the fingerprint.
3. Mutating a helper the entry points call (`_window`) **moves** the fingerprint.
4. A docstring-only edit still does **not** move the fingerprint.
5. `carry_forward` refuses a sidecar carrying the old digest with the existing
   `PublishedResultsError` message, i.e. the refusal path is unchanged.
6. `docs/BENCHMARKS.md` is untouched; `docs/benchmarks/results.json` differs only in
   `algorithm_digest`.
7. No test is weakened or deleted; the fast suite stays green at or above 3320 passed / 29
   skipped.
