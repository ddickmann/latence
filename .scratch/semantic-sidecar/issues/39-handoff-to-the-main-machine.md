# Build: handoff to the main machine and the switch to Databricks

Type: task
Mode: HITL
Status: open
Blocked by: 38

## Question

The owner finishes here on the local backend, then updates the main machine and switches to
Databricks. The handoff has to say what was verified where, so that nothing is taken as proven
on Databricks because it was green on DuckDB.

## Goal

One document in the host's repository:

- pull both branches; build the estate and the ontology export; start the demonstration;
- the hand rehearsal on real models, step by step;
- the switch: the same commands with the Databricks binding; the Postgres variable no longer
  needed there; the seeded second source loaded by its script; the ontology rebuilt through
  the Unity Catalog reader (ticket 33);
- a table: each use case, what was verified on this machine with scripted models, what the
  queue verified on real models, what is left to the main machine;
- the defects fixed in the host (ticket 37) for review;
- the open points for the owner.

A short starting prompt for the session on the main machine is given to the owner directly.

## Done check

The owner has the document and the prompt, and every command in the document was run here as
written, except those that need Databricks, which are marked.

## Tests that prove it

None. The document's commands are run once as written.
