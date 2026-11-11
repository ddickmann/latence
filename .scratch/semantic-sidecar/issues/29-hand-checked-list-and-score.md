# Build: classes and hierarchies are measured against a list a person checked

Type: task
Mode: HITL
Status: open
Blocked by: 27, 28

## Question

Nothing says how right the proposed class tree and hierarchies are. "28 of 39 concepts placed"
is a count of proposals, not of right ones. v1 needs precision and recall against a list a
person has checked, for the one estate we have.

## Goal

- `estates/vehicles.gold.yaml`: the right class per concept (or "none of ours"), the right
  hierarchy steps, the right part-of. Drafted from the catalog and the data, then **corrected
  by the owner**. A draft nobody corrected is the build's own opinion and scores nothing.
- `python -m estate score --gold FILE --out DIR`: precision and recall per kind, once for what
  is proposed and once for what is in the layer after decisions; every miss and every wrong
  proposal listed by name.

## Done check

The owner has corrected the list; the numbers are recorded in this ticket with the date and the
commit; what the numbers do not cover (one estate, one reviewer) is stated beside them.

## Tests that prove it

- The scorer on a fixture with known counts gives the expected precision and recall per kind,
  including a kind with no proposal at all (recall 0, precision not defined, said so).
- An entry of the list that names a concept the export does not hold is an error, not a miss.

## Not tested, and why

The estate's numbers are not asserted in a test: the mirror is not in the repository, and a
floor on a number nobody has yet agreed is a test that pins an accident.

## State (2026-11-11): built, waiting for the owner's correction

- `estate/score.py` and `python -m estate score --out <build folder> --gold <list>`: precision
  and recall for classes (by top class, through the model's own classes), hierarchy steps,
  part-of and links; every wrong proposal and every miss by name; entries under `unsure` count
  on neither side. Where nothing was proposed precision is not a number, and where the list
  holds nothing recall is not: neither is printed as zero.
- `estates/vehicles.gold.yaml`: a draft of 38 classes, 9 steps, 3 part-of and 2 links, with 3
  entries under `unsure`. It was written from the catalog and the data before the score was
  run, and nothing in it was changed afterwards. `checked_by` is empty.
- **Tests: 4**, on an export with known content: the counts and the names per kind; a list
  nobody checked gives numbers that do not count; nothing proposed is not a precision of
  zero; a name the export does not hold, or a class that is no top class, is an error.

**The draft's numbers, which do not count** (the list is mine; `counts_as_measured` is false):

| Kind | Proposed | In the list | Right | Precision | Recall |
|---|---|---|---|---|---|
| Classes | 27 | 24 | 24 | 0.89 | 1.00 |
| Hierarchy steps | 8 | 9 | 8 | 1.00 | 0.89 |
| Part-of | 1 | 3 | 1 | 1.00 | 0.33 |
| Links | 2 | 2 | 2 | 1.00 | 1.00 |

What they point at, if the owner lets the entries stand:

- three placements the list calls wrong: a row per bulletin and component, the counts per
  campaign and dealer, and the labour-time tariff are each placed under a class, and are
  none of the six;
- one step missed: a dealer under its market. The engine puts master data under the finest
  level it carries (the dealer group), and 116 of 211 dealers carry none, so they sit under
  nothing although each has a market;
- two part-of missed: a bulletin's components and an investigation's rows. Neither table has a
  key the engine can find, so nothing is read off keys.

**Open:** the owner corrects the list and sets `checked_by`; then the numbers are measured and
recorded here, and the three findings above become tickets or are dropped.
