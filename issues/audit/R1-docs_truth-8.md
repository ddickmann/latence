# R1 — the docs-site home page and Concepts index understate the Decision Log by 8 ADRs

- **Severity:** major
- **Dimension:** docs-truth / anti-false-green (buyer-facing claims)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), issue #189 (flip public / front door may not lie)
- **Status:** fixed

## What is wrong

`docs/adr/` holds **52** records (`0001–0053`, `0040` unissued). Four published pages state
that number correctly:

- `README.md:13` — "(52 architecture decisions)"
- `README.md:170` — "the 52 architecture decisions"
- `docs/adr/index.md:10` — "the log holds 52 records numbered 0001–0053"
- `docs/GO-LIVE-RUNBOOK.md:115` — "the Decision Log (all 52 ADRs)"

Two do not:

- `docs/index.md:70` — "The **44** architecture decisions behind the framework."
  This is the mkdocs `Home` page: the first page an evaluator lands on after Step 2 of the
  go-live runbook.
- `docs/concepts/index.md:27` — "All **44** architecture decisions, in order".

Both files are inside this stream's diff. `git diff main...feat/release-fleet-integration --
docs/index.md` shows #189 rewriting lines 15–19 — two lines above the stale count it did not
touch.

The guard written for exactly this failure mode,
`packages/latence-core/tests/test_front_door_truthfulness.py::test_readme_adr_count_matches_the_tree`,
hardcodes `_README = _REPO_ROOT / "README.md"` and regexes the count out of **that file only**
(lines 128–142). No docs-site page is ever checked, so the two stale claims sail through and
the suite stays green — the same class of miss the module's own docstring says it exists to
prevent.

## Failure scenario

On day one of the public flip an evaluator opens the deployed site. The Home page card for
the Decision Log says the framework rests on 44 architecture decisions; the Concepts index
repeats it. They click through to the Decision Log, which says the log holds 52 records
numbered 0001–0053, and the README they read on GitHub says 52 twice. The published front
page understates the decision log by 8 records and contradicts three other published pages,
in a project whose entire pitch is that its reasoning is written down and true. Nothing in
CI notices, because the only count guard reads `README.md`.

Verified pre-fix: `ls docs/adr/0*.md | wc -l` → 52, while
`grep -rn "44 architecture decisions" docs/` → two hits, with
`test_front_door_truthfulness.py` green.

## What to build

1. **Correct the two pages** so every published ADR count equals the tree.
2. **Generalise the guard.** The count check must run over the whole front-door set — the
   root pages *and* every tracked page the docs site publishes — not `README.md` alone, and
   must recognise the phrasings actually in use across those pages ("N architecture
   decisions", "all N ADRs", "holds N records"), reporting the offending file and line.
3. Keep the existing README-must-state-a-count requirement: the front door may not go silent
   to satisfy the gate.

## Acceptance criteria

- [x] No published page states an ADR count that differs from `len(docs/adr/[0-9][0-9][0-9][0-9]-*.md)`.
- [x] The count guard scans every front-door document (README + root governance pages + every
      tracked `docs/**/*.md`), not just `README.md`, and names file:line on failure.
- [x] The regression test **fails on the pre-fix tree** (`docs/index.md:70` and
      `docs/concepts/index.md:27` at 44) and **passes** after the fix.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped.
