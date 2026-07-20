# R3 — the "PyPI auto-publish is disabled" guard is anchored to comment prose, not the gate

- **Severity:** major
- **Dimension:** docs-truth / anti-false-green (buyer-facing claims)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), issue #179 (the continuity statement)
- **Status:** fixed

## What is wrong

`packages/latence-core/tests/test_continuity_statement.py::test_pypi_publish_disabled_claim_matches_the_workflow`
guards the buyer-facing claim that automatic PyPI publishing is deliberately disabled. Its
entire assertion is:

```python
workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
assert "if: false" in workflow, ...
```

The literal string `if: false` **does not exist as a gate anywhere in the workflow**.
`grep -n 'if: false' .github/workflows/release.yml` returns only lines 15, 216, 220 and 228 —
every one of them inside a `#` comment block. The executable condition on the `pypi-publish`
job (line 235) is:

```yaml
if: ${{ false && github.event.inputs.dry_run != 'true' }}
```

So the guard is satisfied by *prose describing the gate*, not by the gate. The two are free
to diverge, and the workflow's own enable-checklist instructs exactly the divergence that
breaks it: line 228 says *"Remove `if: false` below"* and line 233 says *"remove the
`false &&` to enable"* — following either one removes the real short-circuit while leaving
the comment prose (and therefore the literal `if: false`) untouched.

This is the precise anti-false-green failure the file's own docstring claims to prevent, and
§7 of the statement advertises that it cannot happen: *"If a row's artifact is missing, the
regression test … fails and the merge gate goes red — so this page cannot quietly become
aspirational."*

Secondary, and true today: the cited artifact is already imprecise. `docs/CONTINUITY.md:111`,
`:191`, `:283`, `docs/GO-LIVE-RUNBOOK.md:13` and `docs/RELEASING.md:197/208/221` all name an
`if: false` gate that no longer literally exists — a buyer who greps for it finds nothing.

## Failure scenario

A maintainer follows the ENABLE CHECKLIST in `release.yml` to turn on Trusted Publishing:

```console
$ sed -i '' "s/if: \${{ false && github.event.inputs.dry_run/if: \${{ github.event.inputs.dry_run/" \
    .github/workflows/release.yml
```

The `pypi-publish` job now fires on **every** `v*` tag push, auto-uploading `dist/*` to PyPI.
Yet `"if: false" in workflow` is still `True` (the comments still say it), so
`test_pypi_publish_disabled_claim_matches_the_workflow` passes and the suite stays at
**2985 passed, 28 skipped**. The merge gate is green while:

- `docs/CONTINUITY.md:111` — "automatic PyPI publishing is **deliberately disabled**"
- `docs/CONTINUITY.md:283` — the §7 verification row asserting the gate as verified fact
- `docs/GO-LIVE-RUNBOOK.md:13` — "scaffolded behind `if: false`"

are all false statements to a buyer verifying the maintainer-manual-publish policy. The one
control that is supposed to catch a silent change of release posture is inert.

## What to build

1. **Anchor the guard to the executable condition.** Parse `release.yml` with `yaml.safe_load`
   (comments are dropped by the parser, so prose can no longer satisfy the check), read
   `jobs["pypi-publish"]["if"]`, and require that the condition **unconditionally
   short-circuits to false** — a bare `false`, or a `false &&` prefix inside the `${{ … }}`
   expression. Any other condition (including one that merely mentions `false` further in)
   must be rejected.
2. **Prove the guard is non-vacuous.** Add a regression test that takes the *real* workflow
   text, applies the exact mutation the ENABLE CHECKLIST prescribes (drop `false && `), and
   asserts the checker **rejects** it. This test fails against the old string-matching
   implementation and passes against the parsed-condition one.
3. **Self-guard the subject.** Fail loudly if the `pypi-publish` job disappears or loses its
   `if:` entirely, rather than passing vacuously.
4. **Correct the cited artifact.** Update `docs/CONTINUITY.md`, `docs/GO-LIVE-RUNBOOK.md`,
   `docs/RELEASING.md` and the workflow's own comments to name the real gate
   (`if: ${{ false && … }}` / "remove the `false &&`") so the enable checklist stops
   instructing a change that defeats the guard.

## Acceptance criteria

- [x] The guard reads the parsed `jobs["pypi-publish"]["if"]` condition; a workflow whose only
      `if: false` occurrences are in comments **cannot** satisfy it.
- [x] A regression test applies the documented enable-mutation to the real workflow and asserts
      the checker rejects it; it **fails on the pre-fix code** and passes on the fix.
- [x] The guard fails loudly if the `pypi-publish` job or its `if:` key is removed
      (no vacuous green).
- [x] No doc claims an `if: false` literal that does not exist in `release.yml`; the workflow's
      ENABLE CHECKLIST names `false &&` consistently.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped.
