# Build: approved changes reach the catalog as staged proposals its owner applies

Type: task
Mode: AFK (the writer); HITL (applying it in a workspace)
Status: resolved (14 Nov): the writer is built; applying a file in a workspace is the owner's
Blocked by: nothing
Emerges from: 16 (decided by the owner, 14 Nov: "approved changes reach the catalog as staged
proposals its owner applies, never as direct writes")

## Question

Nothing on this side writes to Unity Catalog, and nothing proposes a change to it. An approved
comment on an object, a declared key or a tag is to reach the catalog as a file of statements
its owner reviews and applies, never by a call from the engine.

## Goal

A command that writes, from approved decisions only, a staged proposal: one statement per
change (a comment, a key, a tag), each with the decision it rests on, in the warehouse's
dialect, never run by the engine.

## Done check

- Tests: only approved (or certified) elements give a statement; each statement names its
  origin; text is quoted so that no statement can be broken by a name; the reader has no call
  that writes.
- Applying it in a workspace is the owner's.

## Result (14 Nov)

- `python -m estate stage --config … --out …` writes `staged-catalog.sql` in the build folder
  (`estate/staged.py`), for the catalog the estate file names (`catalog:`; without one it
  stops and says so). Nothing runs a statement; the module imports no reader and makes no call.
- **Staged, from a person's approval in force on the statement the build makes now:** a
  concept's key the catalog does not declare, as `ALTER TABLE … ADD CONSTRAINT <table>_pk PRIMARY
  KEY (…) NOT ENFORCED`; a relation between two tables, as `… FOREIGN KEY (…) REFERENCES …
  NOT ENFORCED` (Databricks' informational constraints: they state, they enforce nothing). Each
  statement is preceded by the element, its claim, who approved it and on which day.
- **Not staged, each reported:** an approval by the engine (`not_a_person`: only a person's
  approval goes to the catalog's owner), a decision on another statement (`stale`), what the
  catalog already declares (`declared_in_the_catalog`); a value set and a relation to one are no
  catalog objects and are left out. Every name is written in backticks with a backtick doubled,
  so no name can end a statement.
- **What it does not stage yet:** comments and tags. The engine's definitions of a concept are a
  model's or the catalog's own; a proposed comment for an object the catalog leaves without one
  is a further slice, with its own decision on which definitions may go.
- **Tests** (`estate/tests/test_staged.py`, 3; red on the tree before: no module and no command):
  the toy estate's approved key and relation and nothing else; what the catalog declares and
  what only the engine approved; a name with a backtick, and no import that could write.
- **Live request:** none needed to build it; applying a staged file is the catalog owner's act in
  a workspace, after review.
