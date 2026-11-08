# Grilling: runtime, storage and deployment

Type: grilling
Mode: HITL
Status: open
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
