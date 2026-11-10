# Grilling: the shape of the warehouse pipeline

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 01, 03, 10

## Question

How does warehouse metadata flow through the engine?

- Which harvesters exist in v1: UC metadata, metric views, lineage, query history, Genie
  agents, glossaries, BI models.
- Do they become Source and Stage kinds in the existing Pipeline (ADR-0003), or a parallel
  pipeline?
- Which matching signals are used, and in what order (names, comments, values and sketches,
  lineage, co-usage)?
- What may be read from the data (egress and sensitivity), and how are row filters and masks
  respected?
- How do warehouse assertions and text mentions meet in one resolution phase?

## Answer

Decided by the owner on 2026-11-10, one question at a time. Recorded as
[ADR-0071](../../../docs/adr/0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md).

1. **Harvesters in v1.**
   - Unity Catalog metadata: names, types, comments, tags, declared keys, and the row-filter and
     mask bindings.
   - Metric views.
   - Lineage per table and column, over the REST API.
   - Genie agent configuration (example SQL, instructions, benchmarks).
   - **Existing business glossaries** (files or a catalog tool) as a source of terms and
     synonyms. The owner added this to the recommended set.
   - Query history is optional, behind a capability flag; exported query logs substitute for it.
   - BI models (Power BI, Tableau) wait for a later version.
2. **One pipeline, one resolution phase.** Each harvester is a new Source kind in the existing
   Pipeline and Runner. It emits Assertions with a Locator keyed on UC ids, skips the document
   Stages, and flows into the same corpus-level resolution phase as the text mentions. UC master
   data is the anchor set: mentions link to it first, and what links to nothing is clustered as a
   proposed new Entity. Everything lands in one ledger.
3. **Row values are matched inside the customer's workspace.** Record-to-record matching runs as
   a Spark job in the customer's Databricks (Splink on Spark), only on tables the customer lists
   as master data. Row values never leave the workspace; outside it the engine receives match
   scores, counts and salted hashes. Hosted models see metadata only, unless a deployment makes
   its own explicit egress decision.
   - Consequence: the neural re-scoring of uncertain pairs and the expert workbench both need
     record text, so they run inside the workspace too. Where that runs is for *Runtime, storage
     and deployment*.
4. **Row filters and masks: label and exclude.**
   - The engine runs under its own dedicated service identity.
   - It reads which filters and masks apply before it reads any data.
   - Every profile and every match is labelled with the filter or mask it was computed under.
   - A masked column is never used as matching evidence.
   - Columns tagged as sensitive are left out unless the customer explicitly allows them.
5. **Signals, in three tiers.** A cheaper and more certain signal outranks a weaker one.
   - *Tier 0, metadata only, always on:* declared keys and metric-view definitions first; then
     names, comments and tags; existing glossary terms; embedding similarity of name plus
     comment (the agreed embedder, as a ranking feature that never decides alone); lineage; and
     joins seen in Genie's example SQL or in query logs. Literals in SQL are values and are
     gated like values.
   - *Tier 1, aggregates computed inside the warehouse:* column profiles, salted hashes that
     test whether two columns share values, and key checks.
   - *Tier 2, a hosted model as referee, off by default:* it handles what is still undecided,
     sees names, comments, types and statistics, never values, and every verdict is recorded.
   - The result is always a proposal with a calibrated score that an expert approves, as decided
     for every schema-level element.

**How the warehouse side and the text side meet** follows from answer 2 and the resolution
design: anchor-first linking against UC master data, clustering of what links to nothing, and at
most two rounds of joint resolution.

**Handed to other tickets.** Where the in-workspace jobs, the neural re-scoring and the
workbench run goes to *Runtime, storage and deployment*. The glossary formats to support in v1
are chosen when the first glossary harvester is specified.
