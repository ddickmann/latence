# Build: the Ontology page over the estate

Type: build
Mode: HITL
Status: open
Blocked by:

## Question

After TalkToData and TalkToDashboards, the owner wants to open an Ontology entry in the same
application and show the ontology and knowledge graph the engine built over the same data
(owner, 2026-11-11). This is the first vertical slice of the re-scoped engine: read an estate,
induce its schema level, build the instance level, and show both in the host.

**Decided by the owner on 2026-11-11:**

- the estate is the vehicle and warranty estate (the one with metric views);
- built and tested here on the DuckDB mirror; the Databricks run is the owner's;
- the screen is written natively in the host's user interface, right under Dashboards; a
  shared registry package comes later ([ADR-0074](../../../docs/adr/0074-a-host-shows-the-engines-export-in-a-read-only-screen-of-its-own-off-the-answer-path.md));
- files first, a service route later; the files may be served without sign-in as long as they
  hold synthetic and public data only and personal columns are excluded by a test;
- the resolution part rests on real manufacturer names and documents, and on a seeded second
  dealer source that is labelled as seeded;
- a third cut reads the questions people ask and feeds them back as signals.

## Cuts

| Cut | Holds | State |
|---|---|---|
| A, the ontology | reader, metric views, induction with signals and support, ledger, `graph-export/2`, the host's page | built 2026-11-11 |
| B, the knowledge graph | table roles, aggregated fact edges, resolution (manufacturers, seeded dealers), document mentions, node view, Turtle download | open |
| C, usage signals | joins and terms from recorded questions and queries as evidence and proposals | open |
| Later | a reader for Unity Catalog run against a live workspace; a service route in the host; the generic workbench as a registry package | open |

## Cut A, as built

Code: [`../estate/`](../estate/) (a prototype with its own pinned environment and tests; the
shipped contracts are document-shaped, so real packages wait for the contract generalisation).

- **Reader** (`estate/reader.py`, `reader_duckdb.py`): objects with columns, comments and
  sensitivity; declared keys; metric views; one query call. Only the DuckDB reader exists.
- **Metric views** (`estate/metric_views.py`): the YAML the catalog stores, with joins,
  dimensions and measures.
- **Induction** (`estate/induce.py`): concepts, relations, value sets, hierarchies between
  value sets, attributes and measure bindings; each with its signals; each mined relation with
  the rows that hold and break it; mined relations under 90% support are dropped and listed.
  A relation a metric view or a host-verified join states is `declared`, a mined one
  `proposed`. A sensitive column is never queried.
- **Ledger** (`estate/ledger.py`): assertions and decisions as append-only files; a decision
  is keyed on the element's id and survives a rebuild; a rejected element leaves the layer.
- **Export** (`estate/export.py`, `schemas/graph-export-2.schema.json`): schema level only, no
  row values.
- **The host's page:** a draft pull request in the host's repository (its number 5).

**On the mirror of the vehicle and warranty estate** (28 objects, 13 metric views, about a
second, 823 aggregate queries): 38 concepts (15 entities, 6 facts, 10 value sets, 7 derived),
86 relations (8 stated by metric views, 78 proposed), 38 measures, 8 mined relations dropped.
Of the host's twelve verified joins, nine are found; one runs through a personal column and
is not checked; two point at a derived table and are left out by design.

**What the first run on the real mirror got wrong, each now fixed with a test:** a source line
number was taken as a key; the vehicle, whose own key is personal, was keyed by its owner;
relations were mined into the aggregate tables; a hierarchy that follows from two others was
listed; and the metric view's join condition was not read at all, because YAML reads the bare
key `on` as a boolean.

**Limits of cut A.**
- Which schemas are derived and which columns are load columns is told to the engine in
  `estates/vehicles.yaml`. In Unity Catalog the first comes from lineage; that reader is not
  written.
- Value sets are found by shared column names; two columns that mean the same under different
  names are not joined.
- Labels come from table names and metric-view display names; nothing writes definitions.
- No reader for Unity Catalog exists yet, so nothing has run against the estate in Databricks.
