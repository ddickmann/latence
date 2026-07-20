# R1 / benchmark_integrity #14 — the published benchmark artifacts can silently rot behind their renderer

**Severity:** major
**Stream:** `feat/release-fleet-integration` (#178–#199)
**Surface:** `docs/BENCHMARKS.md`, `docs/benchmarks/results.json`,
`packages/latence-benchmark/src/latence_benchmark/results.py`

## Failure scenario (as confirmed by the audit)

`docs/BENCHMARKS.md` and `docs/benchmarks/results.json` are generated files that are **committed**
to the repository and pointed at by `README.md` as the project's public benchmark evidence. Nothing
regenerates them, and nothing checks them. They therefore drifted away from the renderer that is
supposed to have produced them:

* the committed page carried `| Pinned revision |` while `results.py:207` emits
  `| Pinned revision (commit) |`;
* it carried `'#187 implements them through the same suite seam'` while the renderer emits
  `'Scoring them remains UNIMPLEMENTED (issue #187 … did not implement the `table`/`math`
  assertion classes)'` (`results.py:255-257`);
* it carried the pre-#187 determinism paragraph instead of the current
  `'splits/language slices in a fixed order … or a content digest for a corpus shipped in this
  repository'` text (`results.py:269-278`);
* `results.json` lacked the `metrics` / `headline` / `headline_dimension` keys that the current
  `SuiteResult.as_dict` emits, so a machine consumer diffing against the published sidecar gets a
  schema mismatch;
* `grep -n '^## '` on the page returned only `## Parse`, though `__init__.py:97-101` registers five
  suites and `docs/running-the-benchmarks.md` promises "one page, every Stage".

Every one of those is a *proof that the committed bytes were produced by an older renderer*. The
coverage half of this (only Parse published; the offline `redaction` number never published) was
fixed under `R1-benchmark_integrity` by commits `0795f6d` and `bc9a816`. **The rot itself was not
fixed**: `grep -rn 'BENCHMARKS.md'` across the test suites, `scripts/verify-local.sh` and the CI
workflows shows no freshness check whatsoever, so a fully green gate cannot see the drift, and the
identical divergence will reappear the next time anyone edits `render_markdown` / `render_json`
or `SuiteResult.as_dict` without re-running the harness.

## What to build

A machine gate that makes stale published artifacts a **test failure**, not a reviewer's job.

The sidecar is already a complete, self-describing record of every published measurement, and
`results.carry_forward([], json_path=…)` already rebuilds `SuiteResult`s from it (validating each
against the registry's current dataset pin). So the artifacts are reproducible from what is
committed, with no dataset, no network and no Provider: rebuild them and require the render to be
**byte-identical** to the committed files.

Scope note: this is a *freshness* gate, not a re-measurement. It never asserts a particular
number — it asserts that the committed bytes are what today's renderer emits from today's
registry for the numbers that were measured.

## Acceptance criteria

1. A test rebuilds the published results from `docs/benchmarks/results.json` and asserts
   `render_markdown(...)` is byte-identical to `docs/BENCHMARKS.md`.
2. The same test asserts `render_json(...)` is byte-identical to `docs/benchmarks/results.json`
   (round-trip through `as_dict`), so schema keys added to `SuiteResult` cannot go unpublished.
3. Both use the sidecar's own `generated_on` — no `now()`, no ambient date, deterministic under a
   pinned `PYTHONHASHSEED` (ADR determinism invariant).
4. The gate fails loudly with a unified diff naming the exact command that fixes it
   (`latence-benchmark run --suite all`), so the failure is self-servicing.
5. The gate is blind to nothing it claims to cover: it asserts the rebuilt set is non-empty and
   covers every suite the page sections, so an emptied sidecar cannot make it vacuously pass.
6. Regression evidence: the test FAILS against the pre-fix artifacts committed at `799790b` and
   PASSES on `HEAD`.
7. `make test` stays green at or above 2985 passed / 28 skipped. No test weakened or deleted.
