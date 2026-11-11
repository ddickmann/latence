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

## The merge on the main machine (owner, 2026-11-11: "I want a seamless merge on the main machine later")

The owner runs architecture and code improvements on the main machine at the same time as
this work. What is known and what is not, on 2026-11-11 at 22:40 UTC:

- **Known.** The host's `main` on the hosted repository is `a6ed5cf0` (10 November). The
  working branch starts exactly there: 14 commits ahead, none behind.
- **Not known from here.** Whether the main machine holds commits that are not on the hosted
  repository. Two signs say it may: the gate-queue runner there uses a git server on that
  machine as its remote, and the companion repository on the hosted side ends on 5 November.
- **What the branch touches in the host** (34 files, 11,676 lines added, 4 removed). New
  files: the Ontology page's folder, its browser test and fixture, two dev-tool modules with
  their tests. Small edits to shared files, where a conflict can happen: the route table
  (one line), the navigation (one entry), the two message files (blocks added under
  `ontology`), the page's style sheet, the dev tool's command list and settings registry, one
  refactored condition in the UI stack command, one handbook page and its rendered site
  (rendered: rebuild it instead of merging it).

Steps this ticket has to carry out, none of them optional:

1. The owner runs the three check commands (given in chat) on the main machine and says
   whether its `main` is ahead of `a6ed5cf0`.
2. If it is, the owner pushes that state to the hosted repository (a branch is enough), and
   the working branch is rebased onto it here, the gates rerun, before anything is merged.
3. Only then the handoff document is written against the code that will be shown.

## Done check

The owner has the document and the prompt, and every command in the document was run here as
written, except those that need Databricks, which are marked.

## Tests that prove it

None. The document's commands are run once as written.
