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
