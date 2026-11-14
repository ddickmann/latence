# Build: drift becomes a new proposal, and the old entry is marked stale

Type: task
Mode: AFK
Status: resolved (14 Nov): built in the estate prototype, for concepts and relations
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "drift (a rename, split, retype or drop)
becomes a new proposal with the old entry marked stale")

## Question

When the catalog changes under an approved element (a column renamed, a table split, a type
changed, an object dropped), the next build states something else under the same id or no
longer has the element. Today a decision on it is listed as stale or unknown, and nothing says
what changed.

## Goal

The build compares the elements a person decided with what it states now and, for each that
changed, says what drifted (renamed, split, retyped, dropped), keeps the old decision on the
record as stale, and lists the element as it is now as a new proposal for review.

## Done check

Tests on the toy estate for each of the four kinds of drift.

## Result (14 Nov)

- `estate/drift.py`, run by every build before the decisions are applied. For each concept or
  relation a person decided, it reads from the ledger what was decided (the first statement on
  the decision's claim, and what the ledger knew of the object up to the end of that build) and
  holds it against the catalog's objects as read now:
  - **renamed**: a column the decided statement names is gone and the object has a new one
    (`columns`: old name to new, paired by type, or one to one);
  - **retyped**: a column of the decided concept has another type (`columns`: was and now);
  - **split**: the object is gone and two or more new objects hold its columns between them
    (`now`: their concepts);
  - **dropped**: the object is gone, or a named column is gone and nothing replaced it.
- **What follows:** the old decision stays in the decisions file, untouched, and is listed under
  the export's `drift` (with its verdict, its author and the claim it was made on) and in the
  build's summary; it is **not applied**, so the element as it is now (under its old id, or the
  new ones in `now`) is a proposal again, in state `proposed`. A later build reports the same
  drift, measured against what was decided, not against the build before.
- **The ledger's builds.** To read "what the ledger knew then", a build now notes where its
  statements begin in the ledger (`ledger_builds.jsonl` in the build folder); a folder written
  before reads its whole ledger as one build.
- **Limit:** neither a decision nor an assertion carries a time, so a change made after the first
  statement on a claim and before the decision is reported as drift too, the cautious side.
  Measures, structure and classes are not covered: their drift follows from their concepts'.
- **Tests** (`estate/tests/test_drift.py`, 3, on a copy of the toy estate changed between two
  builds; red on the tree before: no drift was read): nothing drifts when nothing changed; a
  renamed column and a retyped one (and the same on a further build); a split table and a dropped
  one.
