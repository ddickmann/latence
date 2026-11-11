# Build: every use case walked in a browser on the local backend

Type: task
Mode: AFK
Status: open
Blocked by: 36

## Question

"So that all use cases work properly" (owner, 2026-11-11). On the local backend, with scripted
models, each use case is walked in a real browser, and what breaks is fixed.

## Goal

- Run the host's existing browser specs on the scripted world: workbench, samples,
  pre-screening, console (overview, kill switch, changes, why), shell, ontology, TalkToData;
  the acceptance run; the dashboards spec on its world.
- **One new spec**, the only path no spec covers today: TalkToData on the estate world, through
  DuckDB and pgvector: an answer that cites a document, and the notice a person sees when a row
  filter applies to them.
- The host's Python tests that need Postgres and pgvector.
- **Defects** (owner: fix everything, the host's core included): each with a test that failed
  first, in a pull request of its own in the host's repository, listed in the table below.

## Done check

Every spec above passes on this machine; the table of defects is complete; what was not walked
is named.

| Defect | Where | Test that was red | Fix |
|---|---|---|---|

## Tests that prove it

The specs themselves. No further tests are added beyond the one new spec and one test per
defect.

## Limit

Scripted models answer prepared questions only. What a real model does on this backend is
ticket 38 and the owner's hand rehearsal.
