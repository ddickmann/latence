# Build: the catalog reader reads lineage, row filters and column masks

Type: task
Mode: HITL (it needs the real catalog)
Status: open, after v1
Blocked by: 33 (the owner's run on the real catalog)
Emerges from: 33 ("Cut first if time runs short", "Not done"), `estate/HANDOFF.md` ("Reading
Unity Catalog instead of the mirror")

## Question

The Unity Catalog reader (`estate/reader_databricks.py`) reads the table listing, comments,
tags, declared keys and metric views, and runs counting statements. Ticket 33 states what it
leaves out:

- Lineage: "the cut named first".
- "Row filters and column masks are not read: the reader runs under one identity and sees what
  that identity may see."

The decision on the warehouse pipeline (ADR-0071) says filters and masks are labelled and
masked data is excluded, and names REST lineage among what v1 harvests. On the real catalog
neither holds yet.

## Goal

The reader reads lineage, and reads which tables carry a row filter and which columns a mask;
the build labels them and leaves masked columns out of every statement, as it does for a column
tagged sensitive.

## Done check

- The stand-in serves lineage, a filtered table and a masked column, and the toy estate's build
  labels each.
- A masked column appears in no statement the stand-in receives.
- One build on the real catalog lists what it labelled.
