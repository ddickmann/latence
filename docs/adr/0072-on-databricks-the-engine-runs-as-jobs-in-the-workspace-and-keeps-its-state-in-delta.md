# On Databricks the engine runs as jobs in the workspace and keeps its state in Delta

**Status:** Accepted (2026-11-10)

This record fixes where the re-scoped engine (ADR-0067) runs and stores its state when the
estate is on Databricks. It follows from ADR-0071, which requires that row values never leave
the customer's workspace. The owner decided each point on 2026-11-10 (map ticket *Runtime,
storage and deployment*). Nothing here is built yet, and two Databricks product capabilities it
relies on are still to be confirmed on a sandbox.

## Context

The engine has a local Runner and an optional Airflow adapter, and it writes files to Storage.
ADR-0071 moved three things inside the customer's workspace: the record-to-record matcher, the
neural re-scoring of the pairs the matcher leaves uncertain, and the surface where an expert
reviews them. Each needs record text, so each has to run where that text may be shown. ADR-0069
made the ledger of Assertions and Decisions the source of truth, and ADR-0070 made Unity
Catalog the way results reach other consumers.

## Decision

1. **Runners.** The local Runner stays, for standalone use, development and CI. A Databricks
   Jobs Runner is the production path on Databricks: the same Pipeline, executed as jobs in the
   customer's workspace. The Airflow adapter stays optional and unchanged, with no new work.
2. **State lives in Delta tables in Unity Catalog,** in a schema the engine owns: the ledger,
   the published results (the Entity crosswalk and the ontology) and the review state. The
   customer's access rules, lineage and audit apply to all of it. Standalone, the same data
   stays as files on Storage. A transactional store for review state is added only if the
   review prototype shows Delta is too slow for many experts working at once.
3. **The review surface is one web app with two modes.** In production it is deployed as an
   app inside the customer's workspace: experts sign in with their workspace login, the app
   reads and writes the Delta tables under their own permissions, and no row value leaves the
   workspace. For standalone use and prototyping the same app runs locally against files.
4. **Neural models run as batch steps on GPU job compute inside the workspace:** the reranker,
   the decision model and the extraction model. A deployment that already has a model endpoint
   inside its network can point the engine at it instead.
5. **Review-item explanations are built in by default.** The default explanation comes from the
   matcher's own per-field evidence, with no model and no egress. A hosted, written-out
   explanation is available only where a deployment has made the egress decision that lets row
   values leave.
6. **Secrets** live in the workspace's secret scopes, or in environment variables standalone,
   never in Pipeline files.

## Alternatives considered

- **No Databricks-native Runner; start the engine from Airflow or by hand.** Less to build.
  Rejected: it does not fit matching inside the workspace, and it makes the customer's
  orchestrator a prerequisite.
- **Dropping the Airflow adapter.** Less to maintain. Rejected for now: it costs nothing while
  unchanged, and removing it is a separate decision with its own users to ask.
- **Files on cloud storage, registered as external tables.** Least new code. Rejected: the
  ledger and the review state would sit under weaker governance, and concurrent review on
  files is awkward.
- **A transactional database for review state from the start.** Faster writes. Rejected as the
  starting point: Decisions are append-only, which suits Delta, and a second store is one more
  component to run and secure. It stays the fallback if the prototype shows the need.
- **A web app hosted outside the workspace.** Full control over the interface. Rejected: row
  values would leave the workspace, and hosting, login and security would become ours.
- **Review in notebooks.** Cheapest. Rejected: the product's promise is a surface that makes
  the expert's part quick.
- **Model-serving endpoints for the neural models.** Good for live re-scoring. Rejected as the
  default: the work is batch scoring, and an endpoint costs money while idle.
- **A hosted model explaining review items by default.** Readable prose. Rejected: an
  explanation quotes row values, so it would reverse ADR-0071's metadata-only default.

## Consequences

- A Databricks Jobs Runner is a new Runner adapter. The Stage contract stays as it is; only
  the substrate changes (ADR-0003).
- Storage gains a Delta-in-Unity-Catalog backend next to files. The ledger tables are a
  contract and are versioned.
- The review app has to work against both backends, so its data access goes through one seam.
- Two product capabilities are assumed and not yet verified: that an in-workspace app can do
  what the review surface needs, and that GPU job compute can run the chosen models. Both are
  checked on a sandbox before anything is built on them.
- The built-in explanation needs the matcher to expose its per-field evidence in the records
  it emits.
