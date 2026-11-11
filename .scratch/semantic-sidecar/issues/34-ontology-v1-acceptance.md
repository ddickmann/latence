# Decision: is the ontology capability at v1 quality?

Type: decision
Mode: HITL
Status: open
Blocked by: 26, 27, 28, 29, 30, 31, 32, 33

## Question

The owner wants the whole demonstration set up on the local backend, "but only once the
ontology capability is at v1 quality" (2026-11-11). This ticket is that gate: each line below
is checked, the evidence is written beside it, and the owner sees the page and decides.

| Check | Evidence |
|---|---|
| Decisions on every kind of element survive a rebuild and change export and Turtle; a decision whose claim changed is reported and not applied | |
| The export validates as `graph-export/3`; the German page shows no untranslated engine text | |
| The six named defects are gone, each with its test | |
| Classes, hierarchies and part-of are scored against the list the owner corrected | |
| The label loop runs end to end on the seeded source, and the page says how many labels came from a person | |
| One live class call made, its cost recorded, no call on the cached rebuild | |
| Usage signals sit on existing elements only, their source marked | |
| The reader gives the same model through the stand-in; the run on the real catalog is listed as the owner's step | |
| Two builds of the same inputs give byte-identical files | |

## What v1 does not hold, on purpose

Mentions from the estate's documents; classes from an outside vocabulary; SHACL and PROV-O
exports; a service route in the host; real packages in place of the prototype folder.

## Answer
