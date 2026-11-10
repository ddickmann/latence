# Warehouse metadata enters the Pipeline as Sources, and row values are matched inside the workspace

**Status:** Accepted (2026-11-10)

The re-scoped engine (ADR-0067) reads a Databricks estate as well as documents. This record
fixes how: the warehouse readers are new Source kinds in the existing Pipeline, they meet the
text path in one resolution phase, and the comparison of row values runs inside the customer's
own workspace. The owner decided each point on 2026-11-10 (map ticket *Warehouse pipeline
shape*). Nothing here is built yet.

## Context

The Pipeline is built for documents: a Source, document-level Stages, then one corpus-level
phase that resolves entities and assembles the graph (ADR-0003). The warehouse side has no
documents. It has definitions, comments, keys, lineage and curated usage, and for the Entity
crosswalk (ADR-0069) it has rows of master data that must be compared with each other.

Three facts about Unity Catalog shape the design. Everything structural is readable without
system tables. A row filter or a column mask changes what one identity sees, and nothing in a
result says that rows were dropped, so statistics computed under a filter are silently biased.
Query history needs administrative grants and its text is redacted by default.

## Decision

1. **Harvesters in v1.** Unity Catalog metadata (names, types, comments, tags, declared keys,
   and the row-filter and mask bindings); metric views; lineage per table and column over the
   REST API; Genie agent configuration (example SQL, instructions, benchmarks); and existing
   business glossaries as a source of terms and synonyms. Query history is optional behind a
   capability flag, with exported query logs as its substitute. BI models wait.
2. **One Pipeline, one resolution phase.** Each harvester is a new Source kind in the existing
   Pipeline and Runner. It emits Assertions with a Locator keyed on Unity Catalog object ids,
   skips the document Stages, and flows into the same corpus-level resolution phase as the text
   mentions. Unity Catalog master data is the anchor set: mentions link to it first, and what
   links to nothing is clustered as a proposed new Entity. Everything lands in one ledger.
3. **Row values are matched inside the customer's workspace.** Record-to-record matching runs
   as a Spark job in the customer's Databricks, only on tables the customer lists as master
   data. Row values never leave the workspace. Outside it the engine receives match scores,
   counts and salted hashes. Hosted models see metadata only, unless a deployment makes its own
   explicit egress decision.
4. **Row filters and masks are labelled, and masked data is excluded.** The engine runs under
   its own dedicated service identity and reads which filters and masks apply before it reads
   any data. Every profile and every match is labelled with the filter or mask it was computed
   under. A masked column is never used as matching evidence. Columns tagged as sensitive are
   left out unless the customer explicitly allows them.
5. **Signals come in three tiers, and a more certain signal outranks a weaker one.**
   - *Metadata only, always on:* declared keys and metric-view definitions first; then names,
     comments and tags; existing glossary terms; embedding similarity of name plus comment, as a
     ranking feature that never decides alone; lineage; and joins seen in curated example SQL or
     in query logs. Literals in SQL are values and are gated like values.
   - *Aggregates computed inside the warehouse:* column profiles, salted hashes that test
     whether two columns share values, and key checks.
   - *A hosted model as referee, off by default:* it handles what is still undecided, sees
     names, comments, types and statistics, never values, and every verdict is recorded.
   The result is always a proposal with a calibrated score that a human approves (ADR-0069).

## Alternatives considered

- **A second pipeline for the warehouse, joined at the end.** Cleaner separation. Rejected:
  entities would be resolved twice and reconciled afterwards, and linking text mentions to
  warehouse records is the point of having both inputs.
- **Warehouse first, text connected later.** A smaller first version. Rejected: it postpones
  the cross-modal linking that sets the engine apart.
- **The engine pulls the rows and matches on its own compute.** Simpler to build and run.
  Rejected as the production design: row values would leave Databricks for wherever the engine
  runs. It may remain a development mode.
- **Metadata only, with no row-level matching.** Safest. Rejected: there would be no
  crosswalk, which ADR-0069 names as a delivered output.
- **Requiring an exemption from filters and masks.** Unbiased results. Rejected: it is a grant
  many security teams will not give, and the engine can stay honest by labelling instead.
- **Skipping every filtered or masked table.** Simplest. Rejected: those are often exactly the
  important master-data tables.
- **The hosted referee on by default.** More columns resolved without an expert. Rejected:
  every deployment would send metadata to a model provider unless it opted out.

## Consequences

- Whatever needs record text runs inside the workspace: the matcher, the neural re-scoring of
  uncertain pairs, and the surface where an expert reviews them. Where exactly is decided in
  the map ticket *Runtime, storage and deployment*.
- The Source contract has to carry a structured Locator and a filter or mask label, which is a
  `SCHEMA_VERSION` change.
- A run without query history records the reduced evidence, so a ranking made without usage
  data is never passed off as one made with it.
- The glossary formats supported in v1 are chosen when the first glossary harvester is
  specified.
