# Research: the Databricks semantic surface the engine reads and writes

Type: research
Mode: AFK
Status: open
Blocked by: 

## Question

What exactly can the engine read from, and write back to, Databricks, as of November 2026?

- **Metric views:**
  - the YAML spec version and its fields (dimensions, measures, joins, filters, synonyms,
    display names, formats);
  - GA status (our earlier sources disagree);
  - how definitions are listed and read (REST, `information_schema`, `DESCRIBE`);
  - whether the open-sourced Spark core changes any of this.
- **Unity Catalog metadata:**
  - table and column comments;
  - tags and governed tags;
  - certification and deprecation status (`system.certification_status` or its successor);
  - primary and foreign key constraints (informational);
  - `information_schema` coverage;
  - row filters and column masks, and what they hide from a profiler.
- **Usage and lineage:**
  - `system.access.table_lineage` and `column_lineage`;
  - `system.query.history` (what it holds, retention, who may read it);
  - the lineage REST API;
  - the cost of reading each one.
- **Genie:**
  - Genie Agents (formerly spaces): instructions, example SQL, trusted assets, benchmarks,
    entity-matching columns, and the API to read them;
  - **Genie Ontology:** what it ingests, how "authority scores" and citations work, whether
    anything can be exported or inspected, its status and GA timing.
- **Write-back:**
  - `COMMENT ON`, `ALTER … SET TAGS`, creating and altering metric views;
  - which privileges each needs;
  - whether a change can be staged as a proposal rather than applied.
- **Identity:** stable UC object ids (table id, column id, metric view id) that an overlay can
  key on, and how they behave across rename, drop and recreate.

Metis ticket 203 decided "no `system` tables" for Metis. State plainly whether this engine
needs them and what it loses without them.

Resolves with a report at `research/01-databricks-semantic-surface.md`. Every claim carries a
source URL and a status tag (GA / PREVIEW / UNVERIFIED).
