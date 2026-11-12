# Research: the Databricks semantic surface the engine reads and writes

Type: research
Mode: AFK
Status: resolved
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

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/01-databricks-semantic-surface.md](../research/01-databricks-semantic-surface.md). The Databricks docs, blog and forum were blocked from the session. Their claims come from search extracts, and release notes were read through a public GitHub mirror that cites the primary URL. Code was read directly on GitHub (apache/spark, unitycatalog, databricks-sdk-py, the Terraform provider).

- **Reading needs no `system` tables.**
  - The UC REST listing, `information_schema` and `DESCRIBE … AS JSON` give metric view YAML, comments, tags, keys, row-filter and mask bindings, the certification tag and `table_id`.
- **Metric views are GA (spring 2026)**, which settles the R1/R4 conflict in R4's favour.
  - Still in preview: synonyms and display names, parameters, relationships between views (Beta, 30 Oct 2026), and materialization.
  - Sources disagree on Delta Sharing support and on the minimum runtime.
- **UC stores metric views normalised** (observed live in Metis ticket 203).
  - The stored YAML keeps only names and expressions. Comments, display names, synonyms and formats move to column `comment` and `metadata`.
  - Metric views have no row in `information_schema.views`.
- **Open-source Spark accepts only YAML 0.1** (Spark PR 59405), and UC OSS added metric views in 0.5 and 0.6. That makes it a target for a 0.1 export, not a richer source.
- **Certification and keys.**
  - Certification is still the governed tag `system.certification_status` (`certified` or `deprecated`).
  - Primary and foreign keys are informational and GA.
  - The live `row_filters` and `column_masks` columns differ from the docs.
  - **Row filters and masks silently bias any profile run under one identity.**
- **Usage and lineage.**
  - `system.query.history` is Public Preview and admin-only by default.
  - Since 22 Jun 2026 its `statement_text` reads `<Redacted>` unless the reader is an account admin or in `databricks_pii_access`.
  - The docs disagree on lineage table retention (365 vs 395 days).
  - The lineage REST API needs no warehouse but takes one call per object.
- **Genie.**
  - `include_serialized_space` (needs CAN EDIT) returns an Agent's whole setup as JSON: instructions, example SQL, joins, snippets, benchmarks and entity-matching flags.
  - A second, typed format is coming (Terraform, Private Preview), so the reader needs an adapter.
  - Genie Ontology is Public Preview with no GA date, and no inspect or export API was found.
- **Write-back is plain SQL or REST.**
  - Comments need MODIFY, or ownership on views.
  - Tags need APPLY TAG, plus ASSIGN for governed tags.
  - Creating a metric view needs CREATE TABLE plus SELECT on its sources; editing one needs ownership.
  - **Databricks has no proposal or staging state**, so proposals must live in the engine's own store.
- **Identity.**
  - `table_id` covers tables and metric views; drop-and-recreate gives a new id.
  - Columns and metric-view fields have no id, so the overlay key is `table_id` plus the field name.
  - Whether `table_id` survives a rename is UNVERIFIED and needs a live probe in the sandbox.
- **System tables are an optional harvester.** Without them the engine loses usage ranking, mining of real query text, bulk lineage history, and who made a direct UC edit. The structural layer stays complete.
- **For other tickets:**
  - the rename probe for `table_id`, and the grants for query-text access (*get a Databricks sandbox*);
  - the Genie format adapter and profiling under row filters and masks (*the warehouse pipeline*);
  - proposals living in the engine's own store (*the review, certification and versioning lifecycle*);
  - the overlay key, `table_id` plus the field name (*the boundary with Metis*).

## From the map (moved 12 Nov)

[The Databricks semantic surface](issues/01-databricks-semantic-surface.md) — reading needs no system tables (REST, `information_schema` and `DESCRIBE AS JSON` cover structure, metric views, tags, keys and certification). Metric views are GA, and UC stores them normalised. Query history is admin-only, with text redacted by default. Genie Agent configuration is readable as JSON, but Genie Ontology has no export API. Write-back is plain SQL with no staging, so proposals live in our store. The overlay key is `table_id` plus the field name; whether it survives a rename is unverified. System tables are an optional harvester.
