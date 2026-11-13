# Build: a reader for Unity Catalog, proven against a stand-in

Type: task
Mode: HITL
Status: open (built; the owner's run on the real catalog is outstanding)
Decided by the owner (12 Nov): When is the build against the real catalog run; At the switch to
Databricks, as its first step
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

## State (2026-11-11): built and proven against the stand-in; waits for the owner's run

- **The dialect seam.** The `Reader` has `ident()` and `as_text()`, and every statement in
  `induce`, `instances`, `structure` and `resolution` takes its names and its text casts from
  them. A test reader that speaks the other dialect and refuses a double-quoted name gives the
  same model as the mirror (red before the seam). The build on the vehicle mirror is
  byte-identical before and after, export and Turtle file.
- **`estate/reader_databricks.py`**, standard library only:
  - objects, columns, comments and ids from the paged table listing; an object's id is the
    catalog's table id;
  - tags from one statement over `information_schema.column_tags`; a column is sensitive when
    the estate file names it or it carries a tag the estate calls sensitive
    (`sensitive_tags:`);
  - declared keys from each table's own entry;
  - a metric view put together again from the stored definition and the comments, display
    names and synonyms on its listed columns;
  - statements through the statement API: polled until they end, every further part of a
    result fetched, values read by the types the result states. A statement that fails stops
    the build with its id; one that runs past the limit is cancelled.
  - Credentials from the environment only: a token, or a service principal's id and secret.
- **The stand-in** (`tests/uc_standin.py`) answers those calls over the toy estate and
  disagrees the way the services do: a refused token (401, 403), two tables per page, "too
  many requests" on the first listing, every third statement pending first, results in parts
  of five rows, every value as text, and a statement in the wrong dialect failed.
- **Where the endpoints and field names come from:** the host project's own adapter, which
  runs against a live workspace, and research note 01. The provider's documentation could not
  be opened from where this was built.

**Tests: 114 pass (10 new):** the same model through another dialect; the same model through
the catalog as through the mirror, with pages, a retry, pending statements and results in
parts on the way; a column tagged sensitive is in no statement; a metric view is put together
again; declared keys; a refused token, a failed statement and one that never ends each stop
with a clear word, and a refused token is not tried twice; values come back in the types the
result states; the reader over a real socket; a build from the catalog says what is missing in
the environment; the compare command.

**Not done:** lineage (the cut named first). Row filters and column masks are not read: the
reader runs under one identity and sees what that identity may see.

## The owner's run (on the main machine, inside the workspace's reach)

```
cd .scratch/semantic-sidecar/estate
export DATABRICKS_HOST=https://<workspace>        # in the shell, never in a file
export DATABRICKS_WAREHOUSE_ID=<id>
export DATABRICKS_TOKEN=<token>                    # or DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET

.venv/bin/python -m estate build --config estates/vehicles.yaml --catalog <catalog> \
    --verified <metis>/tools/synth/estates/catalog.yaml \
    --class-answer estates/vehicles.classes_answer.gpt-6-luna.json --out out/vehicles-uc

.venv/bin/python -m estate compare --a out/vehicles --b out/vehicles-uc     # the mirror's build beside it
```

What to report back: whether the build ran; the output of `compare` (what only one side holds,
and what differs); and the first error if there was one, with the statement id it names. The
seeded second source is not in the catalog unless `seeded_accounts.sql` was loaded, so the
catalog's build links no records; that difference is expected.

## From the map (moved 12 Nov)

Route table row: a reader for Unity Catalog, proven against a stand-in | HITL | built; **waits for the owner's run on the real catalog**

## Decision brief for the owner (12 Nov)

- **Question:** Do you run block 7 on the main machine now, or at the switch to Databricks?
- **Why his:** only his machine reaches the workspace and holds the credentials.
- **Options:**
  - A. Now.
  - B. At the switch (ticket 39).
  - C. Not for the demonstration (the page stays on the mirror's build).
- **Facts:**
  - The reader is "proven against a stand-in server only"; the stand-in "does not prove the
    warehouse's SQL" (`HANDOFF.md`).
  - The run is two commands and reports back `compare`'s output (33, "The owner's run").
  - Lineage, row filters and masks are not read (now ticket 41, after v1).
- **Recommendation:** B, as the first step of the switch. It needs the same session as the
  three check commands, and a failure only matters if the page is to be shown from Unity
  Catalog.
- **Urgency:** before the demonstration if the demonstration goes to Databricks. Blocks 34's
  eighth check and 39.
- **Cost:** two commands on the main machine, one build (about a minute on the mirror).
- **Decided by the owner (12 Nov):** At the switch to Databricks, as its first step.
