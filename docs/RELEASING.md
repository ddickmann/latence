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

PyPI publishing is **not** part of this automatic flow — see "Publishing to PyPI" below.

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
  --cert-identity 'https://github.com/ddickmann/latence-framework/.github/workflows/release.yml@refs/tags/v0.1.0' \
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

Before tagging, rehearse the whole flow — including a **real** signature — without creating
a release:

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

Locally, everything except the signature can be rehearsed offline:

```bash
for pkg in packages/*/; do uv build --package "$(basename "$pkg")" --out-dir dist; done
bash scripts/release-integrity.sh --tag v0.1.0 --dist dist --out release-assets --no-sign
(cd release-assets && shasum -a 256 -c SHA256SUMS)
```

**Last verified local dry run** (tag `v0.1.0`). Each count below is *derived from the tree*,
and `test_releasing_dry_run_evidence_matches_the_tree` recomputes all three from `packages/`
on every test run — so this block cannot silently rot as the workspace grows (it once
claimed 25 packages / 56 distributions / 58 assets against a tree that had none of those):

| measured | value | derivation |
| --- | --- | --- |
| packages built | 32 | one per `packages/*/pyproject.toml` — the `for pkg in packages/*/` loop the workflow also uses |
| distributions | 64 | one sdist + one wheel per package (every package builds with `hatchling`) |
| assets checksummed | 66 | the 64 distributions + `sbom.cdx.json` + `VERIFY.md` |

`shasum -a 256 -c SHA256SUMS` verified all 66. The CycloneDX 1.6 SBOM was generated from
`uv.lock`, stamped as `latence-framework 0.1.0`, and carried 179 components — 147 pinned
third-party requirements plus 32 workspace path entries (see below). Running
`scripts/generate-sbom.sh` twice produced a byte-identical `sbom.cdx.json`
(`--output-reproducible`), so the release is reproducible from the same lockfile.

> Those 32 components are emitted as `unknown`: they are the `./packages/...` path
> requirements that `uv export --no-emit-project` does not strip and that `cyclonedx-py`
> cannot resolve to a name or version. They are inert for CVE matching — the third-party
> surface an adopter scans is the 147 pinned components — but they do pad the component
> count. Recorded in `issues/audit/R2-docs_truth-7.md`.
>
> There is one such entry per package built, and that is the point: `generate-sbom.sh`
> exports with `--all-packages` and **fails closed** if the export covers fewer packages
> than `packages/` contains. Without that flag `uv export` resolves only the root
> `latence-framework-dev` meta-project, which by design (ADR-0016, thin core) depends on
> none of the heavy Providers — the SBOM then covered 12 of the 32 distributions and
> omitted `torch`, `transformers`, `gliner`, `openai`, `presidio-analyzer`,
> `sentence-transformers` and the rest, while still being stamped, checksummed and
> Sigstore-signed as evidence. Recorded in `issues/audit/R4-security_secrets-3.md`.

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

The heavy lanes not covered by `verify-local.sh` (MinIO `s3://` e2e, the real-Airflow DAG
tests) run in `ci-heavy.yml` on the PR and on push to `main`; confirm those are green on
the commit you are about to tag.

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

## Publishing to PyPI (manual, gated — currently OFF)

Per the maintainer-manual-publish policy, releasing to PyPI is a deliberate human act and
is **not** wired to fire on a tag. `release.yml` contains a fully-formed `pypi-publish`
job that uses **PyPI Trusted Publishing (OIDC)** — no API token is ever stored — but it is
disabled with a leading `false &&` in its `if:` condition and gated behind a protected
`pypi` GitHub Environment.

To enable it (one-time setup, then per-release approval):

1. **Claim the names on PyPI.** Create each distribution project (`latence-core`,
   `latence-parser-*`, …) and, on each, add a Trusted Publisher:
   owner `ddickmann`, repo `latence-framework`, workflow `release.yml`, environment `pypi`.
2. **Create the `pypi` Environment** in repo Settings → Environments and add the
   maintainer(s) as **Required reviewers**, so every publish pauses for a manual approval.
3. **Settle the namespace** (see "Open decisions") — the `latence-*` names must be
   available or owned on PyPI, or the upload 403s.
4. **Turn it on:** remove the `false &&` from the `if:` condition on the `pypi-publish` job
   (keep the `dry_run` clause). From then on, a
   `v*` tag builds + creates the GitHub Release automatically, and the PyPI upload waits in
   the `pypi` environment for a reviewer to approve.

Optional dry run: point the publish step at TestPyPI first by setting
`repository-url: https://test.pypi.org/legacy/` (a commented hint is in the workflow).

## Open decisions (maintainer)

- **Brand / PyPI namespace.** The distribution names are `latence-*`. Confirm these are
  available (or owned) on PyPI before first publish, or pick/claim a namespace. `latence`
  as a lead name may already be taken — verify. Consider reserving the namespace even
  before enabling auto-publish.
- **Auto-publish.** Whether to ever remove the `false &&` short-circuit and let approved tags flow to PyPI,
  or keep publishing fully manual (`uv publish` from a maintainer machine). Both are
  supported; the gated workflow is the recommended path.
