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
for pkg in packages/*/; do uv build "$pkg" --out-dir dist; done
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
- [x] **PyPI is not part of go-live.** [ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
      closed it under the PolyForm licence; [ADR-0063](adr/0063-relicence-to-apache-2-0.md)
      superseded that rationale and reopened the decision, but nothing is enabled and the
      `pypi-publish` job stays short-circuited — so distribution at go-live is still the signed
      GitHub Release and only that, no `latence-*` name has to be available, and nothing here
      blocks step 1. If the decision is ever taken the other way (new ADR first), the name check is
      `for p in packages/*/; do n=$(basename "$p"); curl -s -o /dev/null -w "%{http_code} $n\n" "https://pypi.org/pypi/$n/json"; done`
      — a `404` means available — and the rest of the mechanics are in step 4.
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
   - **Settings → General → Features → Discussions**: enable it. This is not optional
     housekeeping — Discussions is the decided front door for questions, so
     `.github/ISSUE_TEMPLATE/config.yml` lists it as contact link #1 and `SUPPORT.md` routes
     every question to it. Leaving the setting off ships two files pointing at a `/discussions`
     tab that 404s — the exact defect those files were fixed for once already. No test can
     catch this; it is a repository setting, and only the owner can flip it.
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
   [Decision Log](adr/index.md) (all 63 ADRs), [Deploy](deployment.md),
   [Observability](observability.md) and [Secrets](secrets.md).
4. If a custom domain is used, set it in **Settings → Pages** and add the `CNAME`; then
   re-run the workflow so the deployed artifact carries it.

## Step 3 — cut the first release (signed + SBOM)

This is the flow from #178. It is already automated on a tag — you only push the tag.

!!! note "PyPI is not enabled (and the decision is open)"
    [ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
    closed this under the noncommercial licence; that rationale is superseded by
    [ADR-0063](adr/0063-relicence-to-apache-2-0.md) and the decision is reopened. Either way
    the signed GitHub Release is the distribution channel today, `pypi-publish` stays
    short-circuited, and the tag flow below is unaffected. The ordering trap is only live if
    the decision is ever taken the other way: the tag you push freezes the workflow definition
    every run for that tag will use, so the enable would have to be merged **before** the tag —
    see [step 4](#step-4--publish-to-pypi-closed-not-part-of-go-live).

```bash
# 1. Rehearse LOCALLY — build + SBOM + SHA256SUMS + VERIFY.md, everything except the
#    signature. This is the only pre-flight that exists before the tag does: a Sigstore
#    signature is minted from the workflow run's OIDC identity, a workflow_dispatch checks
#    out its `tag` input, and that tag must exist on origin — so nothing can sign v0.1.0
#    until v0.1.0 is pushed, and pushing it is the real release (step 2).
for pkg in packages/*/; do uv build "$pkg" --out-dir dist; done
bash scripts/release-integrity.sh --tag v0.1.0 --dist dist --out release-assets --no-sign
(cd release-assets && shasum -a 256 -c SHA256SUMS)   # then read release-assets/VERIFY.md

# 2. Then, for real. The tag push runs on refs/tags/v0.1.0, which is the identity the
#    published VERIFY.md pins.
git tag -a v0.1.0 -m "latence-framework 0.1.0"
git push origin v0.1.0

# 3. Optional, AFTER the tag exists: re-run the signed flow artifact-only. `--ref` is what
#    Sigstore certifies — omit it and the run dispatches on the default branch and signs
#    under `…/release.yml@refs/heads/main`, an identity no release ever pins.
gh workflow run release.yml --ref v0.1.0 -f tag=v0.1.0 -f dry_run=true
gh run watch          # download the `release-assets` artifact and read VERIFY.md
```

The tag push runs `build` → `release-integrity` (CycloneDX SBOM + `SHA256SUMS` + a keyless
Sigstore bundle) → `github-release`. The release job refuses to publish if the Sigstore
bundle is missing.

**Check:** run the adopter-side verification from [`RELEASING.md`](RELEASING.md) →
*How an adopter verifies a release* against the published assets, from a directory that
contains nothing but the downloads. If `sigstore verify` fails, the release is not
trustworthy — delete it and investigate before going further.

## Step 4 — publish to PyPI (CLOSED, not part of go-live)

!!! danger "Do not do this step"
    [ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
    closed it, on the ground that `pip install` is overwhelmingly a corporate action and
    PolyForm Noncommercial did not licence commercial use. That ground is gone —
    [ADR-0063](adr/0063-relicence-to-apache-2-0.md) relicensed the project to Apache-2.0 — so
    the decision is **reopened, not reversed**: it has not been retaken, the job is still
    disabled, and go-live runs step 1 → 2 → 3 → 5 and skips this one. Everything below is kept
    as the **enable procedure** — record the decision in a new ADR first, then follow it.

The distributions you publish must be **the ones the release signed**, not a fresh local
build.

**This step straddles step 3.** Items 1–3 are PyPI-side setup and can be done at any time.
Item 4 — the workflow enable — must be merged **before** the tag in step 3 is cut, because a
run only ever executes the workflow definition present at the ref it runs on. So decide about
PyPI *first*: if you want v0.1.0 on PyPI, do items 1–4 before pushing the tag, then push it
once and let the single tag run carry the release all the way through. Item 5 is the recovery
path if you have already tagged.

1. Create every project on PyPI and add a Trusted Publisher for each:
   owner `ddickmann`, repo `latence-framework`, workflow `release.yml`, environment `pypi`.
   No API token is ever stored.
2. **Settings → Environments → `pypi`**: add yourself as a Required reviewer, so each
   publish waits for an explicit approval.
3. *(Recommended)* rehearse against TestPyPI once by setting
   `repository-url: https://test.pypi.org/legacy/` on the publish step, then install from
   TestPyPI in a clean venv exactly as in pre-flight step 4.
4. **Enable the publish job *before* the tag is cut — this ordering is the whole step.**
   In `.github/workflows/release.yml`, remove the `false &&` from the `pypi-publish` job's
   `if:` condition (keep the `dry_run` clause), merge that to `main`, and **only then** cut
   the tag in step 3.

    !!! danger "Enabling after the tag is cut publishes nothing, silently"
        A run executes the workflow **definition from the ref it is on**. A run on
        `refs/tags/v0.1.0` therefore uses `release.yml` *as it exists at that tag* — which
        still contains `false &&` if you enabled on `main` afterwards. `pypi-publish`
        evaluates to false and is **skipped**: exit 0, green run, nothing on PyPI, and a
        skipped job leaves no annotation and no summary line to notice.

        Dispatching from `main` instead does not help: the *Refuse to publish under a
        non-tag signing identity* step rejects any non-dry run whose signing identity is not
        `…@refs/tags/<tag>` (step 3). The tag is the only ref allowed to publish, so the tag
        must be the ref that carries the enable.

    Every release run — dry or real — prints its own gate state to the job summary as
    **PyPI publish gate: DISABLED / ENABLED at this ref**. Read that line before concluding
    a publish happened.

5. **If you already cut and signed the tag without the enable**, do not re-run the workflow
   and expect an upload. Choose one:

    - **Preferred — ship it in the next tag.** Merge the enable to `main` and cut `v0.1.1`.
      The signed `v0.1.0` release stays valid and untouched, and nothing an adopter may have
      already downloaded or pinned is invalidated.
    - **Re-cut `v0.1.0`.** Delete the GitHub Release *and* the tag, re-tag the post-enable
      commit, and redo step 3 in full. This **destroys the already-published `v0.1.0`
      assets**: their Sigstore bundle and `SHA256SUMS` certify the old commit, so any copy
      an adopter already holds stops matching what the name now resolves to. Only do this if
      you are certain nothing has been downloaded.

6. Push the tag and approve the `pypi` environment gate when the run pauses on it.

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
- Pin the release. Confirm the Discussions tab enabled in step 1 is actually there before the
  announcement sends readers at `SUPPORT.md`'s first row.
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
