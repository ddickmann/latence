# R7-acceptance_criteria-12 — the go-live runbook's PyPI-enable step is a deadlock

**Severity:** critical
**Stream:** #189 (go-live runbook) · `feat/release-fleet-integration`
**Sites:** `docs/GO-LIVE-RUNBOOK.md:143-160` (Step 4, items 4-5) ·
`.github/workflows/release.yml:112-131` (release-integrity's non-tag refusal) and `:236`
(`if: ${{ false && … }}`) · `docs/RELEASING.md:232` · `.github/workflows/release.yml:220-230`
(ENABLE CHECKLIST)

## What is wrong

Step 4 tells the maintainer to enable the PyPI publish in an order that can never publish:

> 4. In `.github/workflows/release.yml`, remove the `false &&` from the `pypi-publish` job's
>    `if:` condition. Commit that change on `main`.
> 5. Re-run the release workflow for the tag and approve the `pypi` environment gate.

GitHub Actions executes the workflow **definition from the ref the run is on**. A run on
`refs/tags/v0.1.0` therefore uses `release.yml` *as it exists at that tag* — which still
contains `false &&`, because the enable was committed on `main` afterwards. `pypi-publish`
evaluates to false and is **skipped**: exit 0, green run, nothing on PyPI, no error anywhere.

The maintainer cannot escape by dispatching from `main` instead. `release-integrity`'s
fail-closed *"Refuse to publish under a non-tag signing identity"* step (`release.yml:119-131`)
rejects any non-dry run whose `github.workflow_ref` does not end `@refs/tags/<tag>`. So the
only ref allowed to publish is exactly the ref that cannot carry the enable.

The only escape the current text leaves is deleting and re-cutting the tag on a post-enable
commit — which the runbook never mentions, and which invalidates the already-published,
Sigstore-signed v0.1.0 assets produced by Step 3.

#189's AC *"A written go-live runbook covers the manual steps: visibility flip, PyPI publish,
docs deploy, announcement"* is met in letter (the steps are listed) and not in intent
(following them verbatim cannot publish). `docs/RELEASING.md` and the workflow's own ENABLE
CHECKLIST carry the same latent trap: both describe the enable edit without stating that it
must land **before** the tag is cut.

## Failure scenario

1. Maintainer completes Step 3: pushes `v0.1.0`; build → release-integrity → github-release
   run green; signed assets are published and verify.
2. Maintainer follows Step 4 item 4: removes `false &&`, commits on `main`.
3. Maintainer follows Step 4 item 5: re-runs the release workflow for the tag.
4. The run checks out `refs/tags/v0.1.0` and executes that ref's `release.yml`, which still
   short-circuits `pypi-publish`. The job is skipped. The run is green.
5. Nothing is on PyPI. No job failed, no annotation was emitted, and a skipped job leaves no
   trace in the summary — the outcome is indistinguishable from a successful publish unless
   the maintainer independently checks PyPI.

## What to build

1. **Correct the ordering in the runbook.** Step 4 must state that the enable has to be
   merged to `main` *before* the tag is cut, explain why (the definition comes from the tag
   ref, and the integrity guard forbids publishing from any other ref), and give the two
   honest recovery options when the tag is already cut and signed — cut the next version tag,
   or delete and re-cut, with the cost of invalidating already-published signed assets stated
   plainly. The unreachable *"commit on `main`, then re-run for the tag"* sequence must go.
2. **Propagate the ordering constraint** to `docs/RELEASING.md`'s enable checklist and to the
   ENABLE CHECKLIST comment in `release.yml`, so neither can be followed into the same trap.
3. **Kill the silence.** Every release run must state, in its own job summary, whether the
   PyPI gate is open **at the ref that run is executing**, so a green run that published
   nothing can never be mistaken for a successful publish. A skipped job cannot report this
   about itself, so the report belongs in a job that always runs (`release-integrity`, which
   checks out the tag ref and therefore reads the definition actually executing).

## Acceptance criteria

- `docs/GO-LIVE-RUNBOOK.md` Step 4 no longer prescribes committing the enable on `main` and
  then re-running the workflow for an existing tag.
- Step 4 states the ordering constraint (enable merged before the tag is cut) and the
  already-tagged recovery options, including that re-cutting the tag invalidates the
  already-published signed assets.
- `docs/RELEASING.md` and `release.yml`'s ENABLE CHECKLIST carry the same ordering constraint.
- Every release run reports the PyPI gate status for its own ref in the job summary, via a
  step in a job that runs on both dry and real runs.
- The reporting logic is exercised directly: it says DISABLED for the current workflow,
  ENABLED for a copy with `false &&` removed, and fails loudly if the `pypi-publish` job has
  no executable `if:` at all.

## Invariants that must NOT move

- **PyPI auto-publish stays off, and stays auditable from the tree.** The gate remains the
  literal `false &&` short-circuit in the executable `if:`. It must **not** be relocated to a
  repository variable or secret: `docs/CONTINUITY.md` tells an adopter the gate is verifiable
  by reading `release.yml`, and `test_pypi_publish_disabled_claim_matches_the_workflow` +
  `test_pypi_gate_check_rejects_the_documented_enable_mutation` encode exactly that. A
  `vars.*` gate would pass a naive reading of this ticket while destroying that property.
- **Off-by-default is byte-identical.** The new reporting step writes only to
  `$GITHUB_STEP_SUMMARY`; the signed asset set produced by `release-integrity.sh` is
  unchanged bit-for-bit.
- ADR-0034 (no silent errors): a missing `pypi-publish` `if:` is a typed, loud failure, not a
  swallowed one.
