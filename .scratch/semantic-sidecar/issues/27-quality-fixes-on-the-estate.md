# Build: the defects the vehicle estate shows in what the engine proposes

Type: task
Mode: AFK
Status: open
Blocked by: 26

## Question

The build on the vehicle and warranty estate shows defects a reviewer would meet in the first
minute. Each has a cause in the code, read on 2026-11-11:

| What the page shows | Cause |
|---|---|
| "Investigation cas" | `singular` strips `es` from every plural |
| campaign completion "is part of" dealers | the campaign table is keyed on two columns and the child carries one of them; a reference to part of a key is not mined as a relation, so only the dealer key is left and part-of is concluded from it |
| a roll-up from a free-text subject to the manufacturer | the subject column is nearly unique, and a nearly unique column determines every other column trivially |
| two concepts with the same label | the same table name in two layers |
| recall campaigns labelled by their key | no name column, and no way to say which column reads best |
| "entities" under a value set | the page's wording |

A stricter support floor is not the fix for the third: the real step manufacturer → make holds
for 34 of 36 values and would be lost first.

## Goal

- `singular`: strip `es` only after `ss`, `x`, `z`, `ch`, `sh`; a short exceptions map; a
  per-estate `labels:` override.
- A child column named after one column of a parent's composite key becomes a relation
  (reusing `_owner_of` and `Profiler.inclusion`), with its support counted.
- A value set whose distinct values are at least half its rows proposes no roll-up; it is
  listed under `structure.not_proposed` with a code.
- A roll-up that some values break names those values and gets no `broader` edges until a
  person approves it.
- No two concepts share a label: the layer is appended where they would.
- `label_columns:` per object in the estate file; the key is the fallback.
- The page says "values" for a value set.

## Done check

A rebuild on the vehicle estate shows none of the six, and the counts that change (relations,
part-of, roll-ups, `broader` edges) are recorded here with before and after.

## Tests that prove it

Each on a small planted estate, red before its fix:

- a table of plurals: cases, addresses, statuses, categories, boxes, batches;
- a child that carries one column of a two-column key gets a relation and no part-of;
- a nearly unique column proposes no roll-up and is listed as not proposed;
- a roll-up with two breaking values names both and writes no `broader` edge;
- an estate with one table name in two layers gives two different labels;
- an object with `label_columns` is labelled by them.
