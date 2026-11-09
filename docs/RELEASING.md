# Releasing

How a version of the Latence Framework is cut. All publishable `latence-*`
distributions in this workspace share **one** version number and are released
together.

## TL;DR

```bash
# 1. Land everything on main, green (see "Before you tag").
# 2. Bump the version across all packages (see "Versioning").
# 3. Update CHANGELOG.md: move [Unreleased] → [X.Y.Z] with today's date.
# 4. Tag and push — the Release workflow does the rest.
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

1. **build** — builds a wheel + sdist for every package under `packages/` into `dist/`.
2. **release-integrity** — runs `scripts/release-integrity.sh`: generates the per-release
   CycloneDX SBOM, checksums every asset into `SHA256SUMS`, and signs `SHA256SUMS` with
   keyless Sigstore. See "Release integrity" below.
3. **github-release** — creates the GitHub Release for the tag, auto-generates notes from
   the merged PRs since the previous tag, appends the verification instructions and the
   coordinated-vulnerability-disclosure link, and attaches the **signed asset set**.

PyPI publishing is **not** part of this flow at all, automatic or otherwise: the project is
distributed as signed GitHub Releases only
([ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md) — whose
rationale is superseded and whose decision is reopened by
[ADR-0063](adr/0063-relicence-to-apache-2-0.md), without being reversed).
See "Publishing to PyPI" below for the current state and what enabling it would take.

## Release integrity (signing + SBOM)

Every release carries its own verification evidence, produced automatically. Nothing here
is a manual step at release time — this section documents what the workflow already does.

### What a release contains

| Asset | What it is |
| --- | --- |
| `latence_*.whl`, `latence_*.tar.gz` | the built distributions |
| `sbom.cdx.json` | **CycloneDX** SBOM for this exact release, generated from `uv.lock` |
| `SHA256SUMS` | SHA-256 digest of *every* other asset, bare flat filenames |
| `SHA256SUMS.sigstore.json` | keyless **Sigstore** bundle signing `SHA256SUMS` |
| `VERIFY.md` | the copy-pasteable verification instructions (also appended to the notes) |

Only `SHA256SUMS` is signed, because it covers every other asset by digest — one signature
transitively authenticates the whole release, instead of one per file. `VERIFY.md` is written
*before* the checksums so it is itself covered: the instructions telling you how to verify are
not the one file an attacker can rewrite freely.

What that signature does *not* say by itself is that the release carries nothing else.
`sha256sum -c` checks the files the manifest lists and never enumerates the directory, so an
attacker with release-write but no Sigstore identity could leave `SHA256SUMS` and its bundle
untouched and simply **add** a wheel. That is why the procedure has a third step that diffs the
directory against the manifest — it is the step that makes "every asset is covered" checkable
rather than assumed.

### How an adopter verifies a release

Download all the assets of a release into one directory, then:

```bash
# 1. Verify the signature over SHA256SUMS (uvx = uv tool run; `pipx run sigstore` also works)
uvx sigstore verify identity \
  --bundle SHA256SUMS.sigstore.json \
  --cert-identity 'https://github.com/ddickmann/latence/.github/workflows/release.yml@refs/tags/v0.1.0' \
  --cert-oidc-issuer 'https://token.actions.githubusercontent.com' \
  SHA256SUMS

# 2. Verify every artifact against the now-trusted SHA256SUMS
sha256sum -c SHA256SUMS        # macOS: shasum -a 256 -c SHA256SUMS

# 3. Reject any asset SHA256SUMS does NOT list. Step 2 only checks the files the manifest
#    names, so on its own it cannot see an EXTRA distribution added to the release.
diff \
  <(find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -exec basename {} \; | sort) \
  <(cut -c 67- SHA256SUMS | sort)
```

Substitute the tag you downloaded in `--cert-identity`. All three steps must succeed, and step
3 must print nothing; any of them failing means the directory does not hold exactly what this
repository's release workflow produced.

### Why keyless

The signing identity is the **release workflow itself** (`release.yml` at the tag ref),
certified by Sigstore's Fulcio CA from the job's GitHub OIDC token and recorded in the
Rekor transparency log. There is no long-lived private key to store, rotate, or leak, and
nothing for a maintainer to do by hand. The trade-off is that a signature can only be
produced where a Sigstore-trusted OIDC identity exists — in practice, inside the workflow —
so a **local** run of the script uses `--no-sign` and produces everything except the bundle.

`scripts/release-integrity.sh` deliberately has **no default** for signing: you pass
`--sign` or `--no-sign` explicitly. A missing `id-token: write` permission therefore fails
the job loudly instead of quietly shipping an unsigned release, and `github-release` re-checks
for the bundle before publishing.

### Rehearsing a release (dry run)

A rehearsal with a **real** signature needs an **existing** tag: the run is dispatched with
that tag as its input and every job checks it out, and the signature itself is minted from
the run's OIDC identity. So this rehearses a tag that is already on origin (a previous
release, or the one you just pushed) without creating or touching a release. *Before* a tag
exists, rehearse [locally](#rehearsing-locally) instead — that is the first command of
[Cutting the tag](#cutting-the-tag), and it covers everything except the signature.

```
Actions → Release → Run workflow
  Use workflow from:  Tag: v0.1.0   ← select the TAG, not a branch
  tag:                v0.1.0        (the same existing tag)
  dry_run:            true          (default)
```

**Select the tag as the ref.** Keyless Sigstore certifies the ref the *run* is on
(`github.workflow_ref`), not the `tag` input — so only a run on the tag signs under
`…/release.yml@refs/tags/v0.1.0`, the identity a published release's `VERIFY.md` pins. A
rehearsal dispatched from a branch is still signed and still valid, but its identity is
`…/release.yml@refs/heads/<branch>`; `release-integrity.sh` then quotes *that* identity in
`VERIFY.md` and stamps the set with a "Rehearsal — do not publish" banner, so the verification
commands still pass and nothing can be mistaken for a release.

`build` and `release-integrity` run exactly as they would for a real release; the signed
asset set is uploaded as the `release-assets` workflow artifact and the `github-release` and
`pypi-publish` jobs are skipped. Download the artifact and run the three verification commands
from its `VERIFY.md` against it.

To publish for real, push the tag (or re-dispatch with the tag selected as the ref and
`dry_run: false`). A non-dry run on a non-tag ref is **refused** by the `release-integrity`
job: it would create a Release whose own signed instructions could not verify its signature.

### Rehearsing locally

This is the rehearsal available *before* the tag exists — everything except the signature,
offline:

```bash
for pkg in packages/*/; do uv build "$pkg" --out-dir dist; done   # by PATH — see the note below
bash scripts/release-integrity.sh --tag v0.1.0 --dist dist --out release-assets --no-sign
(cd release-assets && shasum -a 256 -c SHA256SUMS)
```

**Last verified local dry run** (tag `v0.1.0`). Each count below is *derived from the tree*,
and `test_releasing_dry_run_evidence_matches_the_tree` recomputes all three from `packages/`
on every test run — so this block cannot silently rot as the workspace grows (it once
claimed 25 packages / 56 distributions / 58 assets against a tree that had none of those):

| measured | value | derivation |
| --- | --- | --- |
| packages built | 34 | one per `packages/*/pyproject.toml` — the `for pkg in packages/*/` loop the workflow also uses |
| distributions | 68 | one sdist + one wheel per package (every package builds with `hatchling`) |
| assets checksummed | 70 | the 68 distributions + `sbom.cdx.json` + `VERIFY.md` |

`shasum -a 256 -c SHA256SUMS` verified all 70. The CycloneDX 1.6 SBOM was generated from
`uv.lock`, stamped as `latence-framework 0.1.0`, and carried 207 components — 173 pinned
third-party requirements plus 34 workspace path entries (see below). Running
`scripts/generate-sbom.sh` twice produced a byte-identical `sbom.cdx.json`
(`--output-reproducible`), so the release is reproducible from the same lockfile.

> **Re-measured 2026-08-25** on the commit that added `packages/latence-gliner25` (ADR-0064).
> The previous block recorded 33 / 66 / 68 assets and 180 components (147 + 33) against a
> 33-package tree; the numbers above are a fresh local rehearsal of the three commands under
> [Rehearsing locally](#rehearsing-locally), not an extrapolation. Re-measured again after
> `latence-gliner25` gained H-G1 caps on `protobuf` / `sentencepiece`: the component counts
> moved (205→207, 171→173 pinned, 24→26 folded-in pins) because those two became DECLARED
> requirements of the excluded package's own lock. The three table rows did not move.
>
> Those 34 components are emitted as `unknown`: they are the `./packages/...` path
> requirements that `uv export --no-emit-project` does not strip and that `cyclonedx-py`
> cannot resolve to a name or version. They are inert for CVE matching — the third-party
> surface an adopter scans is the 173 pinned components — but they do pad the component
> count. Recorded during the pre-release documentation-truth audit.
>
> There is one such entry per package built, and that is the point: `generate-sbom.sh`
> exports with `--all-packages` and **fails closed** if the export covers fewer packages
> than `packages/` contains. Without that flag `uv export` resolves only the root
> `latence-framework-dev` meta-project, which by design (ADR-0016, thin core) depends on
> none of the heavy Providers — the SBOM then covered 12 of the 32 distributions and
> omitted `torch`, `transformers`, `gliner`, `openai`, `presidio-analyzer`,
> `sentence-transformers` and the rest, while still being stamped, checksummed and
> Sigstore-signed as evidence. Recorded during the pre-release supply-chain audit.
>
> **One of the 34 is not a workspace member** (ADR-0064). `latence-gliner25` pins
> `gliner2[local]>=2.0,<3` while `latence-extract-gliner2` / `latence-pii-gliner2` pin
> `gliner2[local]>=1.3,<2`; a uv workspace is ONE resolved environment, so `uv lock`
> refuses the set — correctly — and the root `pyproject.toml` excludes it under
> `[tool.uv.workspace] exclude`. It is still built and published, so it carries its **own
> committed `uv.lock`** and `generate-sbom.sh` folds that lock's export into the
> requirements union (26 pins the workspace lock does not already carry, deduplicated by
> name+version) before the coverage check runs. Leaving it out would have been the exact
> silent false negative that check exists to prevent, and `generate-sbom.sh` fails closed
> if an excluded package has no lock of its own. The release builds by PATH
> (`uv build "$pkg"`) rather than by `--package <name>` for the same reason: the name form
> resolves through the workspace and cannot see an excluded package at all.

> The SBOM generator is version-pinned (`cyclonedx-bom>=7,<8`) on purpose. It had silently
> stopped working: `cyclonedx-py` 7 removed the `--outfile` flag the script passed, so the
> "existing SBOM machinery" produced nothing until this was fixed. Release tooling that is
> only exercised at release time must pin its tools.

### Changing any of this

`scripts/release-integrity.sh` is the single source of truth for the asset set, the SBOM
stamping, and the verification text — the workflow only calls it, and the release notes are
generated from its `VERIFY.md`, so the published instructions cannot drift from the published
assets. `packages/latence-core/tests/test_release_integrity.py` runs the script for real and
fails if the wiring, the fail-closed checks, or the front-page CVD link regress.

## Before you tag

The tag build has no test gate of its own by design (it builds an already-proven commit),
so the commit you tag must already be green:

```bash
bash scripts/verify-local.sh   # mirrors the CI merge gates locally
```

The heavy lanes not covered by `verify-local.sh` (S3-emulator `s3://` e2e, the real-Airflow DAG
tests) run in `ci-heavy.yml` on the PR and on push to `main`; confirm those are green on
the commit you are about to tag.

## Cutting the tag

The tag push **is** the release: everything before it is rehearsal and everything after it
is automated. Run these three in order.

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

**Check:** run the [adopter-side verification](#how-an-adopter-verifies-a-release) against
the published assets, from a directory that contains nothing but the downloads. If
`sigstore verify` fails, the release is not trustworthy — delete it and investigate before
going further.

## Versioning

- Scheme: [SemVer](https://semver.org/). `0.x` — minor bumps may still break APIs.
- **Every** publishable package is kept at the **same** version. Today that version is a
  literal `version = "X.Y.Z"` in each `packages/*/pyproject.toml`.
- To bump, update all of them at once, e.g.:

  ```bash
  # from the repo root — review the diff before committing
  grep -rl '^version = ' packages/*/pyproject.toml \
    | xargs sed -i '' -e 's/^version = "0.1.0"/version = "0.2.0"/'
  ```

  (drop the `''` after `-i` on GNU/Linux `sed`.)

- The workspace root `pyproject.toml` is the dev meta-project `latence-framework-dev`
  at `0.0.0`; it is **not** published and its version is intentionally never bumped.

> **Single source of truth (open option).** Versions are currently duplicated per package.
> A future improvement is a shared source (a `_version.py` read via Hatchling `dynamic`,
> or a release script). It was left out of 0.1.0 to avoid perturbing the already-green
> per-package build backends; adopt it when the release cadence justifies the machinery.

## Publishing to PyPI — not enabled, and the decision is open

**Nothing is uploaded to PyPI today, and `release.yml`'s `pypi-publish` job stays
short-circuited.** The distribution channel is the signed GitHub Release.

[ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
decided this, and its reasoning was entirely about the licence: `pip install` is
overwhelmingly a corporate action, PolyForm Noncommercial did not licence commercial use, and
corporate SCA scanners hard-block the `PolyForm-Noncommercial-1.0.0` SPDX identifier — so
publishing would have put the artifact in a channel whose majority consumer was, by the
licence's own terms, unlicensed.

[ADR-0063](adr/0063-relicence-to-apache-2-0.md) relicensed the project to **Apache-2.0**, and
every one of those premises is now false: `pip install` consumers are licensed, no scanner
blocks `Apache-2.0`, and there is no retained-subset invisibility problem because there is no
retained subset. ADR-0062's own revisit condition ("reopen if the `pip install` audience
changes") is met. So its **rationale is superseded and the decision is reopened** — but it has
**not** been reversed here: whether to publish is a distribution question (name claims, release
cadence, support expectations, an irreversible first upload) that the licence change did not
answer. It stays disabled pending the maintainer's decision, which needs its own ADR.

The install path an adopter uses today is therefore the Release URL or a clone — see the
README's `## Install` section, which states it.

### If the decision is taken the other way

The `pypi-publish` job is kept, fully formed, precisely so this stays cheap. It uses **PyPI
Trusted Publishing (OIDC)** — no API token is ever stored — disabled with a leading `false &&`
in its `if:` condition and gated behind a protected `pypi` GitHub Environment. To enable it
(one-time setup, then per-release approval):

1. **Claim the names on PyPI.** Create each distribution project (`latence-core`,
   `latence-parser-*`, …) and, on each, add a Trusted Publisher:
   owner `ddickmann`, repo `latence-framework`, workflow `release.yml`, environment `pypi`.
   Verified when the decision was taken: `latence-core` is free, while `latence` itself is
   taken (0.2.0, MIT) under the maintainer's separate `latenceainew` account and is unrelated
   to this workspace.
2. **Create the `pypi` Environment** in repo Settings → Environments and add the
   maintainer(s) as **Required reviewers**, so every publish pauses for a manual approval.
3. **Settle the namespace** — every `latence-*` name must be available or owned on PyPI, or
   the upload 403s mid-way and leaves a partially released set.
4. **Turn it on, then tag — in that order.** Remove the `false &&` from the `if:` condition
   on the `pypi-publish` job (keep the `dry_run` clause) and merge it to `main` **before**
   cutting the tag you want published. From then on, a `v*` tag builds + creates the GitHub
   Release automatically, and the PyPI upload waits in the `pypi` environment for a reviewer
   to approve.

    A run executes the workflow definition from the **ref it is on**, so a tag cut before the
    enable landed will never publish — `pypi-publish` short-circuits and is silently skipped,
    and re-running the workflow for that tag re-reads the same disabled definition.
    Dispatching from a branch instead is refused by the non-tag signing-identity guard. Every
    run states its own gate status in the job summary (**PyPI publish gate: DISABLED /
    ENABLED at this ref**) — read that line before concluding a publish happened.

    **If you already cut and signed the tag without the enable**, do not re-run the workflow
    and expect an upload. Choose one:

    - **Preferred — ship it in the next tag.** Merge the enable to `main` and cut `v0.1.1`.
      The signed `v0.1.0` release stays valid and untouched, and nothing an adopter may have
      already downloaded or pinned is invalidated.
    - **Re-cut `v0.1.0`.** Delete the GitHub Release *and* the tag, re-tag the post-enable
      commit, and redo [Cutting the tag](#cutting-the-tag) in full. This **destroys the
      already-published `v0.1.0` assets**: their Sigstore bundle and `SHA256SUMS` certify the
      old commit, so any copy an adopter already holds stops matching what the name now
      resolves to. Only do this if you are certain nothing has been downloaded.

Optional dry run: point the publish step at TestPyPI first by setting
`repository-url: https://test.pypi.org/legacy/` (a commented hint is in the workflow).

## Decided, not open (maintainer)

Both of the questions this section used to leave open were settled by
[ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md),
which also supplied the one they never raised — the licence. That licence has since changed
([ADR-0063](adr/0063-relicence-to-apache-2-0.md)), so the *namespace* question below is live
again in the sense that publishing is no longer ruled out; nothing is claimed or published
until the decision is retaken:

- **Brand / PyPI namespace.** Moot for release purposes. Nothing is published, so no
  `latence-*` name has to be claimed and no upload can 403. The accepted consequence is that
  the names stay unclaimed and could be taken by someone else; reserving them would mean
  uploading a placeholder distribution, which is the act the ADR declines.
- **Auto-publish.** No. The `false &&` short-circuit stays, and manual `uv publish` from a
  maintainer machine is not a substitute route — the decision is about the channel, not about
  the automation. The scaffolding is kept only so a future reversal is a one-line edit.

Reopening either means amending ADR-0062, not editing this page.
