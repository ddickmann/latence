# Go-live runbook (flip public)

Issue #189. Everything an automated agent can prepare and verify is already done and
gated by tests; this page is the short list of **maintainer-manual** acts that remain,
in the order they must happen, with the check that proves each one landed.

Three of the four steps are irreversible in practice — a public repository, a PyPI
distribution name, and a published announcement are all very hard to take back. Read the
[pre-flight](#pre-flight) section first and do not skip the two open decisions in
[Blockers](#blockers-decide-before-step-1).

!!! warning "Nothing here is automated on purpose"
    The Release workflow's `pypi-publish` job is scaffolded behind a `false &&` short-circuit and a
    protected `pypi` Environment; repository visibility is a GitHub setting. Neither
    changes without a human doing it deliberately. See
    [`RELEASING.md`](RELEASING.md) → *Publishing to PyPI*.

---

## Pre-flight

Run from a clean checkout of `main`. Every one of these is green as of the #189 sweep.

```bash
# 1. The full local merge gate: secret-scan · ruff · mypy --strict · pytest ·
#    conformance · stack-validate · bake-off.
bash scripts/verify-local.sh

# 2. The docs site builds with --strict (the same command the docs workflow runs).
uv sync --extra docs
uv run mkdocs build --strict --site-dir site

# 3. Build every distribution and validate its PyPI metadata.
rm -rf dist
for pkg in packages/*/; do uv build --package "$(basename "$pkg")" --out-dir dist; done
uvx twine check dist/*

# 4. Rehearse the published install in a CLEAN environment — local wheels for the
#    latence-* distributions, real PyPI for their third-party dependencies. This is
#    exactly what an outsider gets on day one.
uv venv --python 3.11 --seed /tmp/latence-clean
/tmp/latence-clean/bin/pip install --find-links dist latence-demo latence-console
/tmp/latence-clean/bin/latence-demo --storage-uri file:///tmp/latence-demo-rehearsal
open /tmp/latence-demo-rehearsal/_latence/runs/demo/export/index.html
```

Step 4 must print `contracts: complete`, four `[ok]` screening checks, and a non-empty
graph. If it does not, **stop** — the front door is broken and nothing below should run.

### What the front door promises, and what enforces it

`packages/latence-core/tests/test_front_door_truthfulness.py` fails the build if the
README claims a Runner adapter with no package behind it, quotes an ADR count that does
not match `docs/adr/`, links at a file that does not exist, or tells a reader to install a
distribution that is not in `packages/`. `test_publish_readiness.py` fails if any package
loses its SPDX `License-Expression`, its `license-files` glob, or its release metadata.
Those tests are the reason this runbook can be short.

---

## Blockers (decide before step 1)

- [ ] **Git history is published too.** Flipping visibility exposes every commit, not just
      the current tree. The candid competitor/gap review — which used to live at
      docs/COMPETITIVE-GAP-ANALYSIS.md (deliberately written here *without* backticks: this
      doc set reserves a backticked `docs/….md` for a document that is actually in the
      tree, and the front-door gate enforces exactly that) — was untracked in #189 and is
      now `.gitignore`d, but it **remains reachable in history at commit `fd0a1c1`**.
      Decide one of: (a) accept it as public, (b) rewrite
      history to purge that path before the flip (`git filter-repo --path
      docs/COMPETITIVE-GAP-ANALYSIS.md --invert-paths`), or (c) publish from a fresh,
      squashed history. Do the same audit for any other maintainer-local note.
      *Back the file up outside the repo before merging #189 — the commit deletes it from
      your working tree.*
- [ ] **The `latence-*` names must be available on PyPI.** All 32 distributions are named
      `latence-core`, `latence-parser-*`, … If any name is taken, the publish 403s
      mid-way and you end up with a partially released set. Check every name first:
      `for p in packages/*/; do n=$(basename "$p"); curl -s -o /dev/null -w "%{http_code} $n\n" "https://pypi.org/pypi/$n/json"; done`
      — a `404` means available. See `RELEASING.md` → *Open decisions*.
- [ ] **Intra-package dependencies are unpinned.** Each provider package depends on a bare
      `latence-core` with no version floor. That is correct inside the uv workspace but
      means a published `latence-ner-gliner` will accept *any* `latence-core`. Decide
      whether to add `>=0.1.0` floors before the first publish or after; either is
      defensible, but decide deliberately rather than by omission.

---

## Step 1 — flip the repository to public

1. Confirm the pre-flight above is green on the exact commit you are about to expose.
2. Resolve the history question in [Blockers](#blockers-decide-before-step-1).
3. GitHub → **Settings → General → Danger Zone → Change repository visibility → Public**.
4. Immediately after:
   - **Settings → Code security**: enable Dependabot alerts + security updates, secret
     scanning, and push protection (these are public-repo features and are the reason to
     do this step *before* the release).
   - **Settings → Security → Private vulnerability reporting**: enable it — `SECURITY.md`
     directs reporters to GitHub Security Advisories, so that path must exist.
   - Confirm the OpenSSF Scorecard workflow (`.github/workflows/scorecard.yml`) and CodeQL
     both run; they are no-ops or restricted on private repos.

**Check:** open the repository in a logged-out browser. The README renders, the install
block names real distributions, and `SECURITY.md` shows a *Report a vulnerability* button.

## Step 2 — deploy the docs site

The `docs` workflow builds with `--strict` on every PR and deploys to GitHub Pages on push
to `main`. It has never deployed while the repo was private.

1. **Settings → Pages → Source: GitHub Actions.**
2. Re-run the `docs` workflow on `main` (`workflow_dispatch`).
3. **Check:** the site is live and its navigation reaches the pages an evaluator needs —
   [Continuity & security process](CONTINUITY.md), [Throughput envelopes
   (measured)](perf-results.md), [Public benchmark results](BENCHMARKS.md), the
   [Decision Log](adr/index.md) (all 52 ADRs), [Deploy](deployment.md),
   [Observability](observability.md) and [Secrets](secrets.md).
4. If a custom domain is used, set it in **Settings → Pages** and add the `CNAME`; then
   re-run the workflow so the deployed artifact carries it.

## Step 3 — cut the first release (signed + SBOM)

This is the flow from #178. It is already automated on a tag — you only push the tag.

```bash
# Rehearse first: build + SBOM + Sigstore signing, artifact-only, creates NO release.
gh workflow run release.yml -f tag=v0.1.0 -f dry_run=true
gh run watch          # download the `release-assets` artifact and read VERIFY.md

# Then, for real:
git tag -a v0.1.0 -m "latence-framework 0.1.0"
git push origin v0.1.0
```

The tag push runs `build` → `release-integrity` (CycloneDX SBOM + `SHA256SUMS` + a keyless
Sigstore bundle) → `github-release`. The release job refuses to publish if the Sigstore
bundle is missing.

**Check:** run the adopter-side verification from [`RELEASING.md`](RELEASING.md) →
*How an adopter verifies a release* against the published assets, from a directory that
contains nothing but the downloads. If `sigstore verify` fails, the release is not
trustworthy — delete it and investigate before going further.

## Step 4 — publish to PyPI

Only after step 3's assets verify. The distributions you publish must be **the ones the
release signed**, not a fresh local build.

1. Create every project on PyPI and add a Trusted Publisher for each:
   owner `ddickmann`, repo `latence-framework`, workflow `release.yml`, environment `pypi`.
   No API token is ever stored.
2. **Settings → Environments → `pypi`**: add yourself as a Required reviewer, so each
   publish waits for an explicit approval.
3. *(Recommended)* rehearse against TestPyPI once by setting
   `repository-url: https://test.pypi.org/legacy/` on the publish step, then install from
   TestPyPI in a clean venv exactly as in pre-flight step 4.
4. In `.github/workflows/release.yml`, remove the `false &&` from the `pypi-publish` job's
   `if:` condition. Commit that change on `main`.
5. Re-run the release workflow for the tag and approve the `pypi` environment gate.

**Check:** in a clean environment with **no** `--find-links`, the documented install works
and the demo runs:

```bash
uv venv --python 3.11 --seed /tmp/latence-pypi
/tmp/latence-pypi/bin/pip install latence-demo
/tmp/latence-pypi/bin/latence-demo --storage-uri file:///tmp/latence-demo-pypi
```

If that fails, **yank** the affected release on PyPI (do not delete — yanking preserves the
name and keeps existing pins working) and fix forward.

## Step 5 — announce

Do this last, once steps 1–4 all check out, so that every link in the announcement already
resolves.

- Repository **About**: description, the docs-site URL as Website, and topics
  (`ai-ready-data`, `knowledge-graph`, `rag`, `nlp`, `data-pipeline`, `model-agnostic`).
- Pin the release. Enable Discussions if you want a support channel that is not the issue
  tracker.
- Announcement copy must survive a hostile reader. Say what is measured, not what is
  hoped: the framework is **built and G1-passed, not 1.0**; performance-at-scale and
  deeper structural file validation are open Phase-2 items; two Runners ship (local +
  Airflow) and further substrates are seam-compatible follow-ons, not shipped adapters.
  Link [`PERF-RESULTS.md`](perf-results.md) and [`CONTINUITY.md`](CONTINUITY.md) rather
  than paraphrasing them.

---

## Rollback

| Step | Reversible? | How |
|---|---|---|
| 1 — visibility | Yes, but forks/clones/caches persist | Settings → Change visibility → Private |
| 2 — docs site | Yes | Settings → Pages → Source: None |
| 3 — GitHub Release | Yes | Delete the release + tag; re-tag after the fix |
| 4 — PyPI | **No** | `twine`/web **yank** the version; the name and files stay forever |
| 5 — announcement | No | Post a correction |
