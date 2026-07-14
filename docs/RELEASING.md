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
2. **github-release** — creates the GitHub Release for the tag, auto-generates notes
   from the merged PRs since the previous tag, and attaches every artifact in `dist/`.

PyPI publishing is **not** part of this automatic flow — see "Publishing to PyPI" below.

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
> or a release script). It was left out of 0.1.0 to avoid perturbing the 25 already-green
> build backends; adopt it when the release cadence justifies the machinery.

## Publishing to PyPI (manual, gated — currently OFF)

Per the maintainer-manual-publish policy, releasing to PyPI is a deliberate human act and
is **not** wired to fire on a tag. `release.yml` contains a fully-formed `pypi-publish`
job that uses **PyPI Trusted Publishing (OIDC)** — no API token is ever stored — but it is
disabled with `if: false` and gated behind a protected `pypi` GitHub Environment.

To enable it (one-time setup, then per-release approval):

1. **Claim the names on PyPI.** Create each distribution project (`latence-core`,
   `latence-parser-*`, …) and, on each, add a Trusted Publisher:
   owner `ddickmann`, repo `latence-framework`, workflow `release.yml`, environment `pypi`.
2. **Create the `pypi` Environment** in repo Settings → Environments and add the
   maintainer(s) as **Required reviewers**, so every publish pauses for a manual approval.
3. **Settle the namespace** (see "Open decisions") — the `latence-*` names must be
   available or owned on PyPI, or the upload 403s.
4. **Turn it on:** delete the `if: false` line on the `pypi-publish` job. From then on, a
   `v*` tag builds + creates the GitHub Release automatically, and the PyPI upload waits in
   the `pypi` environment for a reviewer to approve.

Optional dry run: point the publish step at TestPyPI first by setting
`repository-url: https://test.pypi.org/legacy/` (a commented hint is in the workflow).

## Open decisions (maintainer)

- **Brand / PyPI namespace.** The distribution names are `latence-*`. Confirm these are
  available (or owned) on PyPI before first publish, or pick/claim a namespace. `latence`
  as a lead name may already be taken — verify. Consider reserving the namespace even
  before enabling auto-publish.
- **Auto-publish.** Whether to ever remove `if: false` and let approved tags flow to PyPI,
  or keep publishing fully manual (`uv publish` from a maintainer machine). Both are
  supported; the gated workflow is the recommended path.
