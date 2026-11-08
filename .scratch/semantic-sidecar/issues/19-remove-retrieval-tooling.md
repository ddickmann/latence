# Build: remove the retrieval tooling

Type: task
Mode: AFK
Status: open
Blocked by: 07

## Question

Build slice (graduated at charting because its decision, ADR-0068, is already made).

## Build

Remove everything ADR-0068 and *Inventory the carve-out* list:

- the packages;
- the core modules;
- the capability protocols and descriptors;
- the hyperedge contracts (bump `SCHEMA_VERSION`);
- the export columns;
- the CLI group;
- the Quality Report sections;
- the stacks, benchmark lanes, tests and docs.

Keep the embedders as resolution signals.

## Done when

- `scripts/verify-local.sh` is green;
- no import, entry point, stack or doc references a removed module;
- the doc-truth tests pass with corrected counts;
- `uv lock` is clean;
- a seeded text-path run produces the same nodes and edges as before the removal (only
  hyperedges gone).
