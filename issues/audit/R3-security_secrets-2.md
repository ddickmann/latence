# R3-security_secrets-2 — the signed identity must be the identity we tell people to pin

**Severity:** major
**Area:** release supply chain (#178)
**Files:** `.github/workflows/release.yml`, `scripts/release-integrity.sh`, `docs/RELEASING.md`

## What is wrong

`scripts/release-integrity.sh` builds the verification identity from the `--tag` input alone:

```sh
CERT_IDENTITY="https://github.com/${REPO_SLUG}/.github/workflows/release.yml@refs/tags/${TAG}"
```

That string is baked into `VERIFY.md`, which is checksummed, covered by the Sigstore
signature, and used verbatim as the GitHub Release body.

But keyless Sigstore does not mint a certificate for the `tag` *input*. The Fulcio SAN is the
`job_workflow_ref` of the **running** workflow — i.e. the ref the run was triggered on. The
workflow is explicitly built to be dispatched from a branch with the tag passed as an input
(`ref: ${{ github.event.inputs.tag || github.ref }}` on every checkout is only meaningful when
`github.ref` is *not* the tag). Only the `push: tags: v*` path happens to agree with the
hardcoded string.

## Failure scenario

A maintainer follows `docs/RELEASING.md` §"Rehearsing a release (dry run)": Actions → Release →
Run workflow, `tag: v0.1.0`, `dry_run: true`, ref left at `main`.

* `release-integrity` signs `SHA256SUMS` under identity
  `.../release.yml@refs/heads/main`.
* the `VERIFY.md` in the same asset set instructs
  `--cert-identity '.../release.yml@refs/tags/v0.1.0'`.
* the doc then says "Download the artifact and run the two verification commands above against
  it" — which exits non-zero with an identity mismatch on a perfectly good bundle.

Worse, the doc's next sentence is "Re-run with `dry_run: false` … to publish for real". That
path satisfies `github-release`'s `if: ${{ github.event.inputs.dry_run != 'true' }}` and creates
a **real GitHub Release whose own signed instructions cannot verify its own signature** — which
an adopter cannot distinguish from tampering, because `VERIFY.md` says "Step 1 failing means the
checksum file is not the one this workflow produced — stop."

Nothing catches it: `test_dry_run_rehearses_without_publishing_anything` only checks that
`dry_run` appears in the job `if:` expressions, and
`test_verify_instructions_pin_the_tag_and_link_the_cvd_policy` asserts the `refs/tags/<tag>`
string is *present* — i.e. it asserts the wrong value is present rather than that it matches the
signer. The rehearsal therefore structurally cannot exercise the verification story it exists to
prove.

## What to build

1. **Never state an identity we did not sign under.** `release-integrity.sh` gains
   `--workflow-ref <owner/repo/.github/workflows/f.yml@ref>` — the value GitHub exposes as
   `github.workflow_ref`, which *is* the Fulcio SAN. `VERIFY.md` is generated from it. The
   default stays the tag-derived identity, so local `--no-sign` rehearsals are unchanged.
2. **Be loud when the identity is not a tag.** When the workflow ref is not `refs/tags/<tag>`,
   `VERIFY.md` carries an explicit rehearsal banner: this asset set must not be published.
3. **Fail closed on a real publish.** The `release-integrity` job refuses to run when
   `dry_run != 'true'` and the run's workflow ref is not `refs/tags/<tag>` — a real release must
   be triggered by pushing the tag, or dispatched with the tag selected as the ref.
4. **Fix the doc.** The rehearsal instructions must select the *tag* in "Use workflow from", so
   the dry run signs under the same identity a real release will.

## Acceptance criteria

* `release-integrity.sh --workflow-ref <ref>` emits a `--cert-identity` equal to
  `https://github.com/<workflow_ref>` — byte-identical to the Fulcio SAN.
* With no `--workflow-ref`, output is unchanged (`@refs/tags/<tag>`) — off-by-default is
  byte-identical.
* A non-tag workflow ref produces a `VERIFY.md` that (a) pins that ref and (b) says the set is a
  rehearsal and must not be published.
* `release.yml` passes `${{ github.workflow_ref }}` to the script.
* `release.yml` fails the `release-integrity` job when a non-dry-run runs under a ref that is not
  `refs/tags/<tag>`.
* `docs/RELEASING.md` tells maintainers to select the tag as the workflow ref.
* Regression tests fail on the pre-fix code and pass after. Fast suite stays ≥ 2985 passed /
  28 skipped.
