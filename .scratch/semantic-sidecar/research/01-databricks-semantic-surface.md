# Research 01: the Databricks semantic surface the engine reads and writes

As of 8 Nov 2026. Answers [issue 01](../issues/01-databricks-semantic-surface.md).

**Tags.** GA and PREVIEW (Public Preview, Beta, Experimental or Private Preview) give the vendor's
release status. UNVERIFIED means no primary source confirmed the claim.

**Sourcing.** The Databricks and Microsoft docs, blog and forum could not be fetched from here.
Their claims come from search-engine extracts, with the page URL cited. Release notes were read
through a public mirror that quotes the cited primary URL. GitHub was read directly. "Metis 203"
is a live observation from Metis ticket 203 (6–7 Nov 2026) on one workspace.

## Summary

- **Reading:** almost everything the engine needs comes without `system` tables: the UC REST
  listing, `information_schema` and `DESCRIBE … AS JSON`. That covers metric view YAML, comments,
  tags, keys, filter and mask bindings, the certification tag and stable table ids.
- **Metric views are GA** (announced spring 2026). Several sub-features are not: semantic metadata
  (display names, synonyms, formats), parameters, relationships between views and materialization
  are still preview. UC stores a metric view *normalised*: the YAML keeps names and expressions, and
  display names, synonyms and formats move into column metadata.
- **Open-source Spark** runs only YAML 0.1 so far. It is a portability target, not a richer source.
- **Write-back** is plain SQL or REST, gated by MODIFY, APPLY TAG + ASSIGN, or ownership.
  **Databricks has no native "proposal" state.** The engine must stage proposals itself.
- **Identity:** tables and metric views have a `table_id`. Columns and metric-view fields have
  none, so the overlay keys on `table_id` plus the field name.
- **Genie Agent configuration** can be read as JSON. **Genie Ontology** is Public Preview, has no
  export or inspection API that this research found, and has no GA date.
- **System tables:** the engine does not *need* them. Without them it loses usage evidence and
  query-text mining, which are the strongest signals for ranking and alias discovery. Make them an
  optional harvester.

## 1. Metric views

**Status: GA, with sub-features still in preview.** Databricks announced "General Availability and
Open Sourcing of Unity Catalog Business Semantics", which covers metric views
([blog](https://www.databricks.com/blog/redefining-semantics-data-layer-future-bi-and-ai)) — GA.
A third-party source dates it 2 Apr 2026
([colrows](https://colrows.com/blogs/databricks-metric-views/)) — UNVERIFIED.

This resolves the earlier conflict: R1 (28 Sep) cited a 2025 note, so R4's "GA 2 Apr 2026"
stands. Older third-party pages still say Beta
([hex](https://learn.hex.tech/docs/connect-to-data/semantic-models/semantic-model-sync/unity-catalog-metric-views)).
The minimum runtime conflicts: DBR 16.4 (colrows), 17.3
([typedef](https://www.typedef.ai/blog/what-are-metrics-in-unity-catalog-databricks-governed-metric-layer-explained)),
17.2 for `WITH METRICS` DDL (community) — UNVERIFIED.

Sub-features:

| Feature | Status | Source |
|---|---|---|
| Semantic metadata (`display_name`, `synonyms`, `format`): spec ≥ 1.1, DBR ≥ 17.2 | PREVIEW (docs banner; may be stale) | [semantic-metadata](https://docs.databricks.com/aws/en/metric-views/semantic-metadata) |
| Window measures | GA (June 2026 notes) | [release notes](https://docs.databricks.com/aws/en/release-notes/product/) |
| Parameters (DBR ≥ 18.2, 26 Jun 2026) | PREVIEW | [June 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/june#metric-view-parameters-are-now-in-public-preview) |
| Relationships between views (`ALTER VIEW … ADD CONSTRAINT … RELATIONSHIP`), DBR 19+ or a warehouse. Not expressible in YAML, not enforced, not used for optimisation | PREVIEW (Beta, 30 Oct 2026) | [Oct 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/october#relationships-between-metric-views-beta), [docs](https://docs.databricks.com/aws/en/uc-semantics/metric-views/relationships) |
| Materialization | PREVIEW (Experimental) | [docs](https://docs.databricks.com/aws/en/metric-views) |
| Local (dashboard) metric views | GA (27 Aug 2026) | [AI/BI notes](https://docs.databricks.com/aws/en/ai-bi/release-notes/2026) |
| Sharing via OpenSharing | GA per Sept 2026 notes. **Conflict:** the main docs page still says metric views do not support Delta Sharing | [release notes](https://docs.databricks.com/aws/en/release-notes/product/), [docs](https://docs.databricks.com/aws/en/metric-views) |

**YAML spec.** Databricks accepts versions 0.1 and 1.1. The top-level fields are `version`,
`source` (a table or a query), `filter`, `joins`, `dimensions`, `measures` and `materialization`,
plus `parameters` on DBR 18.2+.

- Joins have `name`, `source`, `on` or `using`, nested joins, and `cardinality` (spec 1.1, DBR 18.1+).
- Each dimension and measure has `name`, `expr`, `comment`, `display_name` (≤ 255 chars),
  `synonyms` (≤ 10, each ≤ 255 chars) and `format`.
- `format` types include number, currency (`currency_code`), percentage, byte, date and date_time.
- Measures can have a `window` with `semiadditive`.
- In 1.1, `#` comments are stripped on save.

Sources: [semantic metadata](https://docs.databricks.com/gcp/en/metric-views/data-modeling/semantic-metadata),
[YAML reference](https://docs.databricks.com/gcp/en/business-semantics/metric-views/yaml-reference).
GA for the core; the 1.1 metadata fields are PREVIEW. There is **no language field**: one view
holds one language (German and English need a tag per view or two views).

**How UC stores a metric view (live, Metis 203).** The stored `view_definition` keeps only
`version`, `source`, `joins`, and each field's `name` and `expr`. Everything else moves:

- a field's comment becomes the listed column's `comment`;
- `display_name`, `synonyms` and `format` become the column's `metadata` (a percentage is
  `{"number_percent": {}}`);
- the view's comment becomes the table comment.

A reader must reassemble the definition from the YAML plus the columns — GA behaviour, observed
live.

**Listing and reading:**
- `information_schema.tables` with `table_type = 'METRIC_VIEW'` (live, Metis 203; the SDK enum has
  `METRIC_VIEW`, [SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)) — GA.
- **Not** `information_schema.views`: a metric view has no row there, even for its owner (live,
  Metis 203) — GA behaviour.
- The UC REST listing `GET /api/2.1/unity-catalog/tables` returns the YAML in `view_definition` and
  needs no warehouse (live, Metis 203) — GA. What an identity that does not own the view gets back
  is UNVERIFIED.
- `DESCRIBE TABLE EXTENDED … AS JSON` gives `view_text`, `language` (metric views only), an
  `is_measure` flag per column and `parameters` (DBR 18.2+)
  ([DESCRIBE](https://docs.databricks.com/sql/language-manual/sql-ref-syntax-aux-describe-table.html)) — GA.
  `SHOW CREATE TABLE` also returns the YAML (live, Metis 203).
- Creating one: `CREATE VIEW … WITH METRICS LANGUAGE YAML AS $$…$$`
  ([SQL](https://docs.databricks.com/aws/en/metric-views/create/sql)) — GA.

**The open-source Spark core changes the portability story, not what the engine reads.**
- Spark has a metric-view YAML serde and a CREATE command
  ([PR 53146](https://github.com/apache/spark/pull/53146),
  [PR 53158](https://github.com/apache/spark/pull/53158), Dec 2025).
- An open Oct 2026 PR shows the error "Supported YAML versions: **0.1**"
  ([PR 59405](https://github.com/apache/spark/pull/59405)), so OSS Spark lacks the 1.1 metadata.
- `CREATE OR REPLACE` is broken in the session catalog
  ([PR 59386](https://github.com/apache/spark/pull/59386), open).
- Unity Catalog OSS 0.5.0 (18 Jun) added metric views as "experimental", and 0.6.0 (20 Aug) gave
  "full support" with Spark 4.2
  ([UC OSS releases](https://github.com/unitycatalog/unitycatalog/releases); year inferred from
  release order) — PREVIEW. A community post says Spark 4.2 shipped with metric views — UNVERIFIED.

**Consequence:** a YAML 0.1 export runs on both Databricks and OSS Spark/UC. Synonyms and display
names need 1.1 and Databricks.

## 2. Unity Catalog metadata

**Comments.** Table and column `comment` come from REST, `information_schema` and DESCRIBE — GA.

**Tags and governed tags.**
- Tag assignments are in `information_schema.{catalog,schema,table,column}_tags` (live, Metis 203).
- The REST API is `/api/2.1/unity-catalog/entity-tag-assignments`
  ([SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)) — GA.
- Governed tags (also called "tag policies") limit who may assign a key and which values
  ([permissions](https://docs.databricks.com/aws/en/admin/tag-policies/manage-permissions)).
  **Status conflict:** an Azure page says it needs the tag-policy beta — UNVERIFIED as GA.
- `system.tags.governed_tags` lists the governed keys
  ([July 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/july#governed-tags-system-table-is-in-beta)) — PREVIEW (Beta).

**Certification and deprecation.** This is the governed tag `system.certification_status` with
the values `certified` and `deprecated`.
- It applies to catalogs, schemas, tables, views, volumes, functions, models, dashboards and
  notebooks ([docs](https://docs.databricks.com/data-governance/unity-catalog/certify-deprecate-data)) — GA.
- Older pages describe separate `system.Certified` and `system.Deprecated` tags under a private
  preview. Treat that as superseded.
- Rule-based automation of the tag is PREVIEW (Beta).
- Whether the tag is readable in `information_schema.table_tags` is UNVERIFIED live (the Metis
  importer assumes so).

**Primary and foreign keys.**
- They are informational only and not enforced. `RELY` lets the optimiser use them. GA since DBR
  15.2 ([blog](https://www.databricks.com/br/blog/primary-key-and-foreign-key-constraints-are-ga-and-now-enable-faster-queries)).
- They are read from `information_schema.table_constraints`, `key_column_usage` and
  `referential_constraints`, and from `TableInfo.table_constraints` (not set by the list call,
  [SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)).
- `UNDROP` does not restore them
  ([UNDROP](https://learn.microsoft.com/en-in/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-undrop-table)) — GA.
- They are an evidence source for joins, with low trust because nothing enforces them.

**`information_schema` coverage.**
- It holds tables (including metric views), columns, views, routines, volumes, the tag views,
  constraints, `row_filters` and `column_masks`.
- It is filtered to what the caller may see and needs no SELECT grant
  ([reference](https://docs.databricks.com/sql/language-manual/sql-ref-information-schema.html)) — GA.
- It needs a SQL warehouse, so every read costs compute.
- **Live columns differ from the docs** (Metis 203):
  - `row_filters` has `table_catalog, table_schema, table_name, filter_name, target_columns`;
  - `column_masks` has `…, column_name, mask_name, using_columns`;
  - the documented `filter_col_usage` and similar are refused.
- The `READ METADATA` privilege exposes grants, filters and masks read-only
  ([Aug 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/august#read-metadata-privilege-is-ga)) — GA.

**Row filters and column masks: what they hide from a profiler.**
- A profiler under one identity sees only *its* rows and *its* masked values. Nothing in the result
  says rows were dropped (live, Metis 203: a principal in no group read no filtered rows).
- Value distributions, entity values and cardinality checks are therefore silently biased.
- The engine must read the bindings first and mark every profile "under filter X / mask Y".
- Related limits ([docs](https://docs.databricks.com/aws/tables/row-and-column-filters)) — GA:
  - no vector-search index on such a table;
  - no time travel or clones;
  - metric views do not support data profiling
    ([docs](https://docs.databricks.com/aws/en/metric-views)).
- A filter and a mask survive `CREATE OR REPLACE TABLE` (live, Metis 203).

## 3. Usage and lineage

| Source | What it holds | Status | Access | Retention | Cost |
|---|---|---|---|---|---|
| `system.access.table_lineage`, `column_lineage` | read/write events per table and column, with entity (job, notebook, query) | PREVIEW per the lineage page ([lineage](https://docs.databricks.com/aws/en/admin/system-tables/lineage)) | none by default; a user who is both metastore admin and account admin grants USE + SELECT ([system tables](https://docs.databricks.com/aws/admin/system-tables)) | **Conflict:** 365 days free on the system-tables page, a "rolling 395-day window" on the lineage page | warehouse compute |
| `system.query.history` | SQL warehouse and serverless notebook/job statements, region-wide across workspaces; `statement_text`, `error_message`, timing, user | PREVIEW ([query history](https://docs.databricks.com/aws/en/admin/system-tables/query-history)) | admins by default; share through a dynamic view. **Since 22 Jun 2026, `statement_text` reads `<Redacted>`** unless the reader is an account admin or in `databricks_pii_access` ([June 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/june#redaction-of-the-statement-text-in-the-query-history-system-table-begins)) | UNVERIFIED (not found). Configurable retention, 30–3,650 days, is Beta ([Sept 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/september#configurable-retention-for-system-tables-is-in-beta)) | warehouse compute; a large table, so filter by time |
| Lineage REST: `GET /api/2.0/lineage-tracking/table-lineage`, `…/column-lineage` | upstream and downstream per table or column | GA (no preview label found; UNVERIFIED) ([lineage](https://learn.microsoft.com/azure/databricks/data-governance/unity-catalog/data-lineage)) | the caller's own rights; read live in Metis 203 | follows lineage retention | no warehouse; one call per object |
| External lineage and external metadata REST (`/api/2.0/lineage-tracking/external-lineage`, `…/external-metadata`) | **writable** lineage to non-Databricks objects, with column mappings | PREVIEW ([SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)) | — | — | none |

Column lineage needs source and target referenced by table name and does not cover Lakeflow
(DLT) pipelines; the docs prefer system tables over REST where a region has them.
`system.access.audit` records UC metadata changes and Page activity
([Pages](https://docs.databricks.com/aws/en/uc-semantics/pages)).

## 4. Genie

**Genie Agents** (renamed from Genie Spaces in July 2026; the API paths still say `spaces`)
([July 2026](https://learn.microsoft.com/azure/databricks/release-notes/product/2026/july#genie-spaces-is-now-genie-agents)) — GA.

- **Read API:** `GET /api/2.0/genie/spaces/{id}?include_serialized_space=true` returns the whole
  configuration as a JSON string. It needs CAN EDIT. Create and update take the same payload
  ([SDK genie.rst](https://github.com/databricks/databricks-sdk-py/blob/main/docs/workspace/dashboards/genie.rst)) — GA.
- **Payload (`version: 2`)**, per a Databricks solutions repo
  ([schema.md](https://github.com/databricks-solutions/databricks-genie-workbench/blob/main/backend/references/schema.md)):
  - `data_sources.tables` and `.metric_views`, each with `column_configs` (description, synonyms,
    exclude, `enable_entity_matching`, `enable_format_assistance`);
  - `instructions.text_instructions` (at most 1);
  - `example_question_sqls` (parameterised, with `usage_guidance`);
  - `sql_functions` (trusted UC functions);
  - `join_specs` with a relationship-type annotation;
  - `sql_snippets.{filters, expressions, measures}` with synonyms;
  - `benchmarks.questions` with SQL answers.

  The shape is real, but this repo is not an official API reference — UNVERIFIED as a contract.
- **Benchmarks:** an eval-run API (`genie_create_eval_run`, `list_eval_results`) — GA per the
  SDK, status not stated.
- **Entity matching:** up to 120 columns × 1,024 values. The value lists are built by Genie, stored
  in the workspace bucket and not returned by the API
  ([community](https://community.databricks.com/t5/generative-ai/purpose-and-usage-of-entity-matching-in-genie-spaces/m-p/155861)) — UNVERIFIED.
- **Knowledge store** edits do not change UC metadata
  ([concepts](https://docs.databricks.com/aws/en/genie-agents/concepts)) — GA.
- **New Genie Agent definition format:** the Terraform `databricks_genie_agent` resource takes a
  typed `definition` and states that the old `serialized_space` format is not accepted
  ([TF doc](https://github.com/databricks/terraform-provider-databricks/blob/main/docs/resources/genie_agent.md)) — PREVIEW (Private Preview).
  **A second format is coming; build the reader behind an adapter.**
- Since November 2026, an agent answers from only what a partially entitled user may see
  ([AI/BI notes](https://learn.microsoft.com/azure/databricks/ai-bi/release-notes/2026#genie-agents-enhancements)) — GA.

**Genie Ontology** — PREVIEW (Public Preview, earlier "gated")
([docs](https://docs.databricks.com/aws/en/genie/genie-ontology)).
- **GA:** no date or announcement found. One secondary source says snippets opened to all
  customers without a request on 13 Aug 2026 — UNVERIFIED.
- **Inputs:** modeled (metric views, Domains, Pages, certified assets) and inferred "snippets"
  from tables, queries, dashboards, notebooks, pipelines and Genie Agents
  ([blog](https://www.databricks.com/blog/how-genie-ontology-powers-product-development-databricks),
  [answers](https://answers.databricks.com/ai-learn-business-definitions-from-dashboards-queries-we)).
- **Authority scores** weigh origin, usage, ties to certified assets, and freshness; a definition
  asserted by hand beats an inferred one. The formula is unpublished — UNVERIFIED in detail.
- **Permissions and citations:** snippets come only from assets the user may see; an answer's
  Snippet icon lists its sources. There is no lineage view of the ontology itself.
- **Export and inspection:** none found. There is no ontology API in the Python SDK, the Terraform
  provider or the CLI (searched in Nov 2026) — UNVERIFIED as a definitive absence.
- **Pages** (the human layer) are PREVIEW (Beta). They have a draft/publish lifecycle and Genie
  Code can bulk-create them as drafts. No public API was found
  ([Pages](https://docs.databricks.com/aws/en/uc-semantics/pages),
  [blog](https://www.databricks.com/blog/unity-catalog-pages-governed-home-your-business-knowledge-genie-ontology)).
- **Implication:** the engine should *feed* the Ontology through metric views, Pages, comments,
  synonyms and certification. It cannot read the Ontology back as evidence.

## 5. Write-back

| Change | Mechanism | Privileges | Status |
|---|---|---|---|
| Table or column comment | `COMMENT ON` / `ALTER … ALTER COLUMN … COMMENT` (SQL, needs a warehouse). The REST table update changes only the owner ([SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)) | owner or MODIFY on a table; **owner** on a view, since views have no MODIFY ([AI comments](https://docs.databricks.com/aws/en/comments/ai-comments); [community](https://community.databricks.com/t5/data-engineering/unable-to-update-view-comment-via-sql-api-unless-owner-but/td-p/144179)) | GA. The view rule is UNVERIFIED in primary docs |
| Tags | `ALTER … SET TAGS` / `UNSET TAGS`, or the REST entity-tag-assignments call | owner, or APPLY TAG + USE SCHEMA + USE CATALOG; a governed tag also needs ASSIGN or MANAGE on its policy ([SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py), [tags](https://docs.databricks.com/aws/database-objects/tags)) | GA |
| Certify or deprecate | set `system.certification_status` | ASSIGN on that governed tag | GA |
| Create a metric view | `CREATE VIEW … WITH METRICS` | CREATE TABLE + USE SCHEMA (+ USE CATALOG), SELECT on sources ([SQL](https://docs.databricks.com/aws/en/metric-views/create/sql)) | GA |
| Alter a metric view | `ALTER VIEW … AS $$yaml$$` | **owner only**; give ownership to a group for shared editing | GA |
| Relationship between views | `ALTER VIEW … ADD CONSTRAINT … RELATIONSHIP` | UNVERIFIED (probably owner) | PREVIEW |
| External lineage | REST external-lineage | UNVERIFIED | PREVIEW |

Saving a comment runs an ALTER that "can disrupt pipelines and jobs" (AI comments page). In
Metis's workspace the service principal held only `USE CATALOG` and `CREATE SCHEMA` (Metis 203),
so write-back to objects others own needs grants from their owners.

**Proposals:** Databricks has **no staged-change state** for comments, tags or metric views.
AI-comment Accept/Edit is UI-only and writes at once; Page drafts are Beta with no API found;
Request for Access (`RfaAPI`) covers permissions only. The engine therefore holds proposals in its
own store (target `table_id` and field, the SQL or YAML diff, evidence, actor) and applies one only
after approval. A candidate metric view may be dry-run in a throwaway schema, as Metis's drift
probe does, but never where consumers read.

## 6. Identity

- **Tables, views and metric views** carry `table_id` (`TableInfo.table_id`, "the unique
  identifier of the table"; [SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/service/catalog.py)) — GA.
  - **Drop and recreate gives a new id.** Dropped tables are listed by id (`SHOW TABLES DROPPED`,
    `UNDROP TABLE WITH ID`) ([UNDROP](https://learn.microsoft.com/en-in/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-undrop-table)) — GA.
  - `CREATE OR REPLACE TABLE` keeps the table's identity and history
    ([drop/replace](https://docs.azure.cn/databricks/delta/drop-table)) — GA. It is UNVERIFIED for
    `CREATE OR REPLACE VIEW … WITH METRICS`.
  - **Rename** keeping `table_id`: no source states it — UNVERIFIED. Test it live.
- **Columns have no id** (`ColumnInfo` has name, position, type and comment only), and neither do
  metric-view dimensions or measures. The overlay key is `(table_id, column or field name)`, with
  type and position kept as a fingerprint. A column or field rename looks like a delete plus an
  add, so the engine must propose the re-link rather than guess it ("no silent over-merge").
- Schemas and catalogs have ids too; whether they survive a rename is UNVERIFIED.

## What the engine needs vs. what it loses without system tables

**It does not need them.** The harvesters that matter run on REST, `information_schema` and
DESCRIBE: definitions, comments, tags, certification, keys, filter and mask bindings, ids, and
per-table lineage over REST. A deployment like Metis's, with no `system` access, gets a complete
*structural* layer.

**What it loses:**
1. **Usage evidence:**
   - which tables, columns and metric views are actually queried, by whom and how often;
   - this is the input Genie's own authority scores use;
   - without it the engine cannot rank competing definitions by use, only by declared trust.
2. **Query-text mining:**
   - joins, filters and measure expressions people really write, and aliases from column
     aliases;
   - even with access, `statement_text` is redacted for anyone who is not an account admin or in
     `databricks_pii_access`. The miner needs that membership, which is a sensitive grant.
3. **Bulk, historical lineage:** the REST lineage is per object and many calls. The system tables
   give region-wide history (365 or 395 days).
4. **Actor history of UC edits** (`system.access.audit`): who changed a comment or a tag, and
   when. Without it, a human edit made directly in UC has no actor in the evidence chain; the
   engine sees only that the value changed between runs.
5. **The governed-tag registry** (`system.tags.governed_tags`, Beta). Minor: assignments are still
   in `information_schema`.

**Substitutes:** exported query logs, Genie example SQL and benchmarks (curated, higher trust),
the engine's own SQL, and per-object REST lineage.

**Recommendation:** an optional `system`-table harvester behind a capability flag. Every run that
lacks it records the reduced evidence, so no run passes off a ranking made without usage data as
one made with it.

## Open questions

1. Does `table_id` survive a rename and `CREATE OR REPLACE VIEW … WITH METRICS`? (Live probe.)
2. What do `view_definition` and column `metadata` return to an identity that does not own the
   metric view (BROWSE, SELECT or READ METADATA only)?
3. Is `system.certification_status` readable in `information_schema.table_tags`, and are governed
   tags GA in the target region?
4. `system.query.history` free retention; lineage tables 365 or 395 days?
5. Is semantic metadata (synonyms, display names) still PREVIEW? This decides whether writing them
   to UC is a supported contract.
6. When does the typed Genie Agent `definition` replace `serialized_space`, and does it expose
   entity-matching values?
7. Will Genie Ontology or Pages get an API? Pages are the natural write target for the engine's
   glossary.
8. Which privileges does `ADD CONSTRAINT … RELATIONSHIP` need?
9. When will Apache Spark accept YAML 1.1? This sets the open export version.
