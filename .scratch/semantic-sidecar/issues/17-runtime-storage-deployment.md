# Grilling: runtime, storage and deployment

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 01, 11

## Question

Decide:

- the runners: local, a Databricks Jobs / Lakeflow Runner (ADR-0003 names it a seam-compatible
  follow-on), Airflow kept or dropped;
- storage: files on Storage (ADR-0010/0017), Delta tables in UC, Iceberg REST (`iceberg.py`),
  Lakebase for review state;
- whether the review UI is a Databricks App;
- the sidecar transport to Metis (an MCP plugin host or tables);
- GPU and model serving on Databricks versus endpoints;
- secrets.

## Answer

Decided by the owner on 2026-11-10, one question at a time. Recorded as
[ADR-0072](../../../docs/adr/0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md).

1. **Runners.** The local Runner stays, for standalone use, development and CI. A new Databricks
   Jobs Runner is the production path on Databricks: the same Pipeline, executed as jobs in the
   customer's workspace. The Airflow adapter stays as it is, optional and unchanged, with no new
   work on it.
2. **Storage on Databricks: everything in Delta tables in Unity Catalog,** in a schema the
   engine owns: the ledger of Assertions and Decisions, the published results (the Entity
   crosswalk and the ontology) and the review state. The customer's access rules, lineage and
   audit apply, and Metis and Genie read the published tables directly. Standalone, the same
   data stays as files on Storage. A transactional store for review state is added only if the
   workbench prototype shows Delta is too slow for many experts clicking at once.
3. **The expert review surface is one web app with two modes:** deployed as a Databricks App in
   the customer's workspace for production (experts sign in with their Databricks login, the app
   reads and writes the Delta tables under their own permissions, and no row value leaves the
   workspace), and run locally against files for standalone use and prototyping. The current
   feature set of Databricks Apps was not verified; *the expert resolution workbench* prototype
   confirms it before this is committed.
4. **The neural models run as batch steps on GPU job compute inside the workspace,** as part of
   the Databricks Jobs Runner: the reranker, the decision model and the extraction model. The
   work is batch scoring, so a GPU cluster that exists only for the job is the cheapest fit and
   needs no always-on service. A deployment that already has a model endpoint inside its network
   can point the engine at it instead.
5. **Review-item explanations are built in by default; the hosted one is opt-in.** The default
   explanation comes from the matcher's own per-field evidence (which fields agree, which
   conflict, how strongly), with no model and no egress. The hosted, written-out explanation is
   available only where a deployment has made the egress decision that lets row values leave.
   This settles a collision between two earlier decisions: a hosted model for explanations
   (the model set) and hosted models seeing metadata only (*Warehouse pipeline shape*).
6. **Secrets** (settled without a question): API keys and tokens live in Databricks secret
   scopes, or environment variables standalone, never in Pipeline files.

**Already decided elsewhere.** The transport to Metis is the proposal file plus UC tables
(ADR-0070), so no MCP plugin host is needed in v1. Iceberg stays an optional export.

**Unverified.** Product facts about Databricks Apps and Lakebase could not be checked from this
environment. *Get a Databricks sandbox* and the workbench prototype confirm them.
