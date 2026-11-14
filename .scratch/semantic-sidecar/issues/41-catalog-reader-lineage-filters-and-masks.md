# Build: the catalog reader reads lineage, row filters and column masks

Type: task
Mode: HITL (it needs the real catalog)
Status: open: built and proven against the stand-in (14 Nov); the build on the real catalog is the owner's (live request below)
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

## Result (14 Nov): built against the stand-in

- **Row filters and column masks** are read from each table's own entry
  (`GET /api/2.1/unity-catalog/tables/{full_name}`), which the reader already fetched for the
  declared keys and now fetches once per table for both. A table's `row_filter.function_name`
  becomes the object's `row_filter`; a column whose `mask.function_name` is set is `masked` and
  also sensitive, so every check that keeps a sensitive column out of the engine's statements
  keeps it out too.
- **Lineage** is read from the table lineage endpoint
  (`GET /api/2.0/lineage-tracking/table-lineage?table_name=…`), once per listed table, its
  upstream tables (an upstream that is no table, such as a notebook, is skipped). An upstream
  in a layer the estate reads keeps the engine's name; any other keeps the catalog's full name
  and is marked outside. `Reader.lineage()` returns `None` for the DuckDB mirror, which keeps no
  lineage, and the export then says nothing of it.
- **The build labels each** (`graph-export/3`, optional fields stated in its schema): a concept
  over a filtered table carries `row_filter`; a masked column's attribute carries
  `"masked": true` beside `"sensitive": true`, and the Turtle export says `v:masked` for it, not
  `v:personal` (a mask does not say the column is personal); the export's `lineage` holds the
  edges between the estate's concepts (`source: "catalog"`), and `meta.lineage` counts them and
  the edges from outside. Lineage is not yet a voice of induction: it is read and shown, and
  using it (a derived table's relations, its place in a hierarchy) is a step of its own.
- **The stand-in** now governs: a filter and masks stated only in the table's own entry, the
  lineage endpoint, and a masked column's values replaced by the mask's value in any statement
  that reads it, as a warehouse answers an identity the mask hides the column from (wrong
  without an error, which is why the reader must never select it).
- **Tests, red first:** `test_lineage_row_filters_and_column_masks_are_read_and_labelled_and_a_masked_column_is_in_no_statement`
  (the toy estate's whole build through the catalog reader against the stand-in; red: the
  stand-in and the reader knew none of it) and `test_a_mirror_reads_no_lineage_and_its_export_says_nothing_of_it`
  (red: no `lineage` on the mirror's reader). Checked by mutation: with a masked column no
  longer treated as sensitive, the first test fails.
- **The vehicle estate's export from the mirror** is unchanged by this ticket (compared byte for
  byte as JSON with the build before it, apart from ticket 44's `proposal` mark).

**Live request (the owner's main machine, read-only on the workspace):** block 7 of
`estate/HANDOFF.md`, as written, after `.venv/bin/pip install -q -r requirements.txt`. Then:

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
.venv/bin/python - <<'PY'
import json
e = json.load(open("out/vehicles-uc/ontology.json"))
print("row filters:", {c["id"]: c["row_filter"] for c in e["concepts"] if "row_filter" in c})
print("masked:", [(c["id"], a["column"]) for c in e["concepts"] for a in c.get("attributes") or [] if a.get("masked")])
print("lineage:", e.get("lineage"), e["meta"].get("lineage"))
PY
```

Report the three lines, or the first error with the call it names (an endpoint the workspace
refuses, or a field it names otherwise, is exactly what the stand-in cannot show).
