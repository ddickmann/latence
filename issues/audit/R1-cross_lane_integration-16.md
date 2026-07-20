# R1 — the docs-site Home and Concepts pages state a false ADR count (44 vs 52 on disk)

- **Severity:** major
- **Dimension:** cross-lane integration / docs-truth
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), issue #189 (flip public / "the front door may not lie")
- **Status:** **DUPLICATE — already fixed on this branch by commit `b3d4053`** (ticket
  `issues/audit/R1-docs_truth-8.md`). Verified, not re-fixed. See *Resolution* below.

## What was reported

`docs/adr/` holds 52 ADRs. #189 corrected `README.md` to 52 and edited `docs/adr/index.md` to
assert 52, but left `docs/index.md:70` ("The 44 architecture decisions behind the framework.")
and `docs/concepts/index.md:27` ("All 44 architecture decisions, in order") untouched. The
machine gate at `packages/latence-core/tests/test_front_door_truthfulness.py`
(`test_readme_adr_count_matches_the_tree`) regexed `_README.read_text()` only, so it stayed
green while two pages mkdocs publishes (`mkdocs.yml:139` `Home: index.md`, `:147`
`concepts/index.md`) shipped a count 8 records behind the log they link to.

## Failure scenario

`ls docs/adr/[0-9][0-9][0-9][0-9]-*.md | wc -l` → 52. An outsider lands on the docs Home page,
reads "44 architecture decisions", clicks through to the Decision Log, reads "the log holds 52
records numbered 0001–0053", and falsifies the front door in under a minute by counting the
table. The gate written to prevent exactly this class of claim had its document set silently
narrowed to `README.md`.

## Acceptance criteria

1. Every page the docs site publishes states an ADR count equal to the files under `docs/adr/`.
2. The gate reads the *published set* — README + root governance pages + every tracked
   `docs/**/*.md` — not `README.md` alone.
3. The gate is non-vacuous: it fails when any front-door page drifts, and it asserts that at
   least one page still *states* a count (deleting the claim is not a way to reach green).
4. `make test` stays at or above 2985 passed / 28 skipped.

## Resolution

Fixed on this branch before this lane ran:

- `docs/index.md:70` and `docs/concepts/index.md:27` → 52.
- `test_readme_adr_count_matches_the_tree` was widened and renamed to
  `test_adr_count_matches_the_tree_on_every_front_door_page`, backed by `_front_door_docs()`
  (README + `CONTEXT.md`/`CONTRIBUTING.md`/`SECURITY.md` + every **git-tracked** page under
  `docs/`) and an `_ADR_COUNT_CLAIM` regex that also matches the Decision Log's own
  "holds N records" phrasing. It additionally asserts `total_claims` is non-zero and that the
  README specifically keeps stating a count, closing the delete-the-claim escape.

Re-verified in this lane rather than re-fixed: reintroducing `44` into both pages makes
`test_adr_count_matches_the_tree_on_every_front_door_page` fail with
`docs/concepts/index.md:27: claims 44` / `docs/index.md:70: claims 44`; restoring 52 makes it
pass. All four ACs hold. No further change required.
