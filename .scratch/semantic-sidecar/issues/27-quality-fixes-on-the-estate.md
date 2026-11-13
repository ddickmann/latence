# Build: the defects the vehicle estate shows in what the engine proposes

Type: task
Mode: AFK
Status: resolved
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

## Answer

Built on 2026-11-11. Two causes in the table above were wrong, and reading the data put them
right before any code was written.

**The roll-up from "subject" was not a nearly unique column.** In the table that reports on
recalls per period the subject repeats (69 values over 512 rows, 65 of them more than once),
so no rule about repetition removes it. What it is: the title of a recall or of an
investigation. Each campaign has exactly one subject (75 of 75) and each case one (21 of 21),
and a title determines the manufacturer because it names one thing. The rule that follows
from the data: **a value set with more than half as many values as rows in the table of an
entity with a single key names that entity, and groups nothing.** No roll-up is proposed from
or to it, and the export lists it under `structure.not_proposed` with the counts.

**A value the data puts under two values gets no edge.** The plan was to hold back every
`broader` edge of a step that some values break. That would have emptied the tree of
manufacturers for one value. Instead the step names its breaking values with the values the
data puts each under, the values that hold keep their edges, and the breaking ones are in no
tree until a person says where they belong.

**A third defect showed while the tests were written:** a key column named after a table
(`dealer_id`) was credited to the first table of that name. Where two layers hold a table of
one name, the second lost its key to a column called `name`. A column is now named after
every table of that name.

| What the page showed | Now, on the vehicle estate |
|---|---|
| "Investigation cas" | "Investigation case"; the estate file states "Manufacturer communication" for the one abbreviation |
| campaign completion "is part of" dealers | it links a campaign and a dealer; one relation more (87) for the reference to one column of the campaign's key |
| a roll-up from subject to manufacturer | gone; "Subject" is listed as the name of an investigation case (19 values over 21 rows). Investigation cases now sit under their manufacturer. 9 steps, were 10 |
| two concepts labelled "Vehicle" | "Vehicle (clean)" and "Vehicle (internal)" |
| a value under the more frequent of two parents, marked contested | no edge; manufacturer → make names "Volkswagen Group of America, Inc." under VOLKSWAGEN and AUDI. 301 `broader` edges, were 390 (89 fewer: the subject step and the one contested value) |
| recall campaigns labelled by their key | unchanged, on purpose: the table has no name column and the campaign number is what people call a recall. `label_columns:` in the estate file is there for an estate that needs it |

Unchanged by the fixes: 39 concepts, 38 measures, 2,514 nodes, the 938 linked records, 28
placed concepts.

**Tests: 80 pass (22 new), each fix red before it:** a table of 16 plurals; a label the estate
states; a reference to one column of a composite key is a relation, and the concept links two
things; a title that has a value per row of its entity is a name and not a level; label
columns the estate states, and a missing one is an error; no two concepts share a label, and a
key column named after a table is that table's key in every layer; a value under two values
gets no edge and the step names it; a linking concept is written as a part of neither in the
Turtle file (it was written as a part of both).

**Host page:** the step's detail lists the breaking values, a block lists what was not taken
for a level, a value set's count reads "values". 197 unit and route tests, the browser walk in
both languages.

**Not done:** the engine does not yet use a name it found as the label of the entity's rows
(the list of usual name columns already covers "title" and "subject"); the singular rule is
English only.

## From the map (moved 12 Nov)

Route table row: six defects in what the engine proposes on the vehicle estate | AFK | **done**
