# R7-acceptance_criteria-13 — the go-live runbook's release rehearsal cannot run

**Severity:** major
**Area:** release supply chain / go-live docs (#189, interacts with #178)
**Files:** `docs/GO-LIVE-RUNBOOK.md` (Step 3), `docs/RELEASING.md`,
`.github/workflows/release.yml`

## What is wrong

Step 3 of the go-live runbook is the maintainer's first contact with the signed-release
flow. Its shell block is, in order:

```bash
gh workflow run release.yml -f tag=v0.1.0 -f dry_run=true   # 1
gh run watch                                                # 2
git tag -a v0.1.0 -m "latence-framework 0.1.0"              # 3
git push origin v0.1.0                                      # 4
```

Two independent defects:

1. **It cannot run.** Command 1 dispatches the release workflow with `tag=v0.1.0` *before*
   the tag is created (commands 3–4). Both `build` and `release-integrity` check out
   `ref: ${{ github.event.inputs.tag || github.ref }}`; a `workflow_dispatch` always
   supplies `tag` (it is `required: true`), so both jobs check out `v0.1.0`. `git tag`
   in this repo returns nothing — v0.1.0 does not exist on origin at that point — so the
   very first step of the very first job dies with `couldn't find remote ref v0.1.0`.

2. **Even once the tag exists, the command rehearses the wrong identity.** `gh workflow
   run` with no `--ref` dispatches on the default branch, so `github.workflow_ref` is
   `…/release.yml@refs/heads/main`. `docs/RELEASING.md` (§"Rehearsing a release") is
   explicit that keyless Sigstore certifies the ref the *run* is on, not the `tag` input,
   and that only a run on the tag signs under `…/release.yml@refs/tags/v0.1.0` — "the
   identity a published release's `VERIFY.md` pins". The runbook's own command therefore
   rehearses an identity a real release never uses.

There is also no pre-tag signed rehearsal available at all: a Sigstore signature needs the
workflow's OIDC identity, which needs a run, which needs the tag on origin — and pushing a
`v*` tag *is* the real release (`github-release` runs whenever `dry_run != 'true'`, and a
push event has no `dry_run` input). The runbook must say what is actually possible instead
of prescribing an impossible order.

## Failure scenario

Maintainer follows the runbook top to bottom on go-live day. Step 3, first command:
`gh workflow run release.yml -f tag=v0.1.0 -f dry_run=true`. The run appears, and the
`build` job fails immediately at `actions/checkout` with `couldn't find remote ref v0.1.0`.
Nothing in the runbook explains it. If they work around it by tagging first, the same
command then signs under `…@refs/heads/main` and produces a rehearsal of an identity the
real release will never carry.

## What to build

Rewrite Step 3's block into a sequence that is executable in the order written and
truthful about identities:

* a **local** unsigned rehearsal (`release-integrity.sh --no-sign`) as the pre-flight that
  is actually possible before the tag exists;
* the real cut (`git tag` → `git push origin`), which is the signed run on the tag ref;
* any `gh workflow run release.yml` in the docs dispatched with an explicit
  `--ref <tag>`, and only after that tag exists.

Keep `docs/RELEASING.md` consistent (its rehearsal section says "Before tagging" while
requiring "the same existing tag").

## Acceptance criteria

1. No `gh workflow run release.yml -f tag=<T>` appears anywhere in `docs/` before `<T>` is
   pushed to origin in the same procedure.
2. Every `gh workflow run release.yml` invocation in `docs/` passes an explicit `--ref`
   naming the release tag (never an implicit default-branch dispatch).
3. Step 3 still documents a pre-tag rehearsal — the local `--no-sign` one — and states why
   the signature cannot be rehearsed before the tag exists.
4. A regression test parses the runbook's Step 3 shell block and enforces (1)–(3);
   it fails on the pre-fix text.
5. `make test` stays green at or above 3320 passed / 29 skipped.
