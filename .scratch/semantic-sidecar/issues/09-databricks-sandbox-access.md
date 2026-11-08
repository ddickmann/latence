# Task: get a Databricks sandbox for the engine

Type: task
Mode: HITL
Status: open
Blocked by: 

## Question

Provision what the warehouse path needs to be judged and built:

- a workspace;
- a Unity Catalog catalog loaded with the synthetic estate (reuse Metis's synthetic estate and
  its 13 metric views if possible);
- a serverless SQL warehouse;
- a service principal with read on the catalog and, if the owner allows, on the `system`
  schemas (lineage, query history);
- a separate principal or grant for staged write-back (comments, tags, metric views).

Record where the credentials live (as `secret://` references, never values), the URLs, and the
row and table counts.
