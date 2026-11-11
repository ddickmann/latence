# Build: a reader for Unity Catalog, proven against a stand-in

Type: task
Mode: HITL
Status: open
Blocked by: 26

## Question

Only a DuckDB mirror reader exists. Done check 3 (UC-native) and the owner's plan to switch the
demonstration to Databricks need a reader for the real catalog. No workspace is reachable where
this is built, so it is proven against a stand-in and by one run of the owner's.

One defect has to go first: the estate code writes DuckDB's SQL. `induce._q` quotes a column as
`"col"` and casts with `CAST(x AS VARCHAR)`. Databricks reads `"col"` as a text value, so every
count would be wrong without an error.

## Goal

- **Dialect seam.** The `Reader` protocol gains `ident()` and `as_text()`; every statement in
  `induce`, `instances` and `structure` goes through them.
- **`estate/reader_databricks.py`** over plain HTTPS, no vendor SDK: the catalog listing with
  comments, tags and declared constraints, metric-view definitions, and statement execution
  with polling, result pages and cancel. The endpoint list is checked against the current
  documentation before coding. Credentials come from the shell's environment only.
- **A stand-in server** written for the tests, over a fixture catalog and DuckDB, that can
  disagree: 401 and 403, a listing in pages of two, a statement that is pending and then done,
  a result in two parts, a failed statement, one 429, values returned as text, and a refusal
  of double-quoted identifiers and a bare `VARCHAR`.

## Done check

- The toy estate gives the same model through both readers.
- The owner has run one build against the real catalog from the commands below, and the
  differences to the build on the mirror are listed here.

## Tests that prove it

- `induce` and `build_structure` give identical models through the DuckDB reader and through
  the stand-in. Red today, on the quoting.
- A column tagged sensitive appears in no statement the stand-in receives.
- A failed statement stops the build and names the statement's id; a 429 is retried once.
- A token that is refused gives one clear error and no retry.

## Not proven here

Databricks' own SQL semantics, types and limits. The stand-in proves the protocol only.

## Cut first if time runs short

Lineage.
