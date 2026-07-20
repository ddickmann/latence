# R3-docs_truth-6 — `docs/CONTINUITY.md` §5.5 claims all GitHub Actions are SHA-pinned; 12 of 27 external refs are tag-pinned

**Severity:** major
**Where:** `docs/CONTINUITY.md:204` (the §5.5 supply-chain controls table)
**Evidence:** `.github/workflows/codeql.yml:33,36,45,48`; `.github/workflows/scorecard.yml:33,38,47,54`;
`.github/workflows/docs.yml:43,46,61,79`

## What is wrong

§5.5 asserts, as a buyer-facing supply-chain control:

> | GitHub Actions pinned to commit SHAs | `.github/workflows/*.yml` (e.g. `actions/checkout@93cb6ef…`) — a moved tag cannot change what runs |

The claim is scoped to `.github/workflows/*.yml` — *all* workflows. It is false for three of
the six workflows. Only `ci.yml`, `ci-heavy.yml` and `release.yml` are SHA-pinned.

```
$ grep -rnE '^\s+uses:' .github/workflows/*.yml | grep -vc '@[0-9a-f]{40}'
12
```

Of those 12, three are local composite actions (`./.github/actions/setup-workspace`, which has
no SHA to pin), so **12 of 27 external action references are pinned to mutable tags**:

| Workflow | Tag-pinned `uses:` |
| --- | --- |
| `codeql.yml` | `actions/checkout@v4`, `github/codeql-action/init@v3`, `.../autobuild@v3`, `.../analyze@v3` |
| `scorecard.yml` | `actions/checkout@v4`, `ossf/scorecard-action@v2.4.0`, `actions/upload-artifact@v4`, `github/codeql-action/upload-sarif@v3` |
| `docs.yml` | `actions/checkout@v4`, `astral-sh/setup-uv@v5`, `actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4` |

## Failure scenario

The two workflows with *zero* SHA pins are `codeql.yml` and `scorecard.yml` — the exact two
files the same §5.5 table cites as the repo's **SAST** and **supply-chain posture scoring**
controls. An attacker (or a compromised upstream maintainer) who force-moves the `v3`/`v4`
tag changes what executes inside `codeql.yml`, which holds `security-events: write`, and
inside `docs.yml`, which holds the GitHub Pages deploy (`pages: write`, `id-token: write`).
The document's own parenthetical cites `actions/checkout@93cb6ef…` from `release.yml` as if
it were representative of all workflows.

The claim is unguarded in **both** directions:

* No test in the repo opens `codeql.yml` or `scorecard.yml`
  (`grep -rn 'scorecard\|codeql' --include=*.py packages/` → no matches).
* `test_continuity_statement.py::_claimed_repo_paths` explicitly skips any backticked token
  containing `*` (lines 84–85, `if "*" in cand: continue`), so the `.github/workflows/*.yml`
  citation in that very row is never verified at all.

A buyer running the repository's own OpenSSF Scorecard would see **Pinned-Dependencies fail**
against a document telling them it passes.

## What to build

1. **Make the claim true** (the real fix, not a doc downgrade): SHA-pin every external
   `uses:` in `codeql.yml`, `scorecard.yml` and `docs.yml`, each with a trailing
   `# <version>` comment, matching the convention already used in `ci.yml`/`release.yml`.
   Where the repo already pins a given action elsewhere, reuse that exact SHA so the repo
   carries one version of each action, not two.
2. **Guard it** with a non-vacuous regression test that walks *every* workflow and composite
   action and fails on any external `uses:` that is not a 40-hex SHA — so the §5.5 row can
   never silently go false again when a workflow is added.

## Acceptance criteria

* `grep -rnE '^\s+uses:' .github/workflows/*.yml | grep -v '@[0-9a-f]{40}'` returns **only**
  `./.github/actions/...` local composite-action references.
* Every SHA-pinned `uses:` carries a `# vX.Y.Z` version comment.
* A regression test asserts, for every `.github/workflows/*.yml` **and**
  `.github/actions/*/action.yml`, that each non-local `uses:` is `owner/repo[/path]@<40-hex>`;
  it must fail on the pre-fix tree and pass after.
* The test is self-guarding: it fails if it discovers no workflows or no external `uses:`
  (so deleting the workflows cannot make it vacuously green).
* `docs/CONTINUITY.md` §5.5 stays accurate and now points at the enforcing test.
* No regression: `make test` ≥ 2985 passed, 28 skipped.
