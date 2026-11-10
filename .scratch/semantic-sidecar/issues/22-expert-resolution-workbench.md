# Prototype: the expert resolution workbench (UI/UX)

Type: prototype
Mode: HITL
Status: resolved
Blocked by:

## Question

What does the expert's side of the human-machine split look like?

The machine prepares and resolves as much as it can. The expert needs a very convenient
workbench to resolve the review band and to QA everything against enterprise standards.

Prototype a clickable, rough UI to react to, using mock data first and then the feasibility
spike's real review items. Cover:

- **The queue:**
  - ranked by impact × uncertainty;
  - grouped by cluster, so one action resolves many pairs;
  - diverse across types and sources.
- **Evidence cards:**
  - mentions in their source context, with the spans highlighted;
  - UC object cards: comment, tags, lineage and profile statistics, never raw personal values;
  - the match-weight waterfall;
  - the calibrated probability;
  - why the machine was unsure.
- **AI suggestions:** a pre-drafted suggestion with its rationale. The reviewer confirms or
  overrides, and every disagreement is recorded.
- **Decisions:**
  - keyboard-first, one key each: accept, reject, split, merge, not-sure, skip;
  - generalising a decision into a rule proposal.
- **Cluster graph view,** with bridge edges highlighted, for split decisions.
- **Conflicts:** new evidence that contradicts a human decision appears as a queue item.
- **QA:**
  - sampled audits of auto-accepts and auto-rejects, with precision intervals;
  - four-eyes approval for critical types;
  - version certification;
  - reviewer agreement;
  - throughput (decisions per hour, auto-resolution rate, review-band trend).

**Where it lives** is decided in *runtime, storage and deployment* and *the review, certification
and versioning lifecycle*: standalone (for example a Databricks App) and inside Metis's review
routes. References: Splink's comparison viewer and cluster studio (MIT), Metis's review routes,
and its reference UI.

Resolves when the owner has reacted to the prototype. The answer records the interaction model
it settled on.

## Progress (2026-11-10): first clickable cut

Status stays `open`: the ticket resolves when the owner has reacted to the prototype.

- **Where:** [`../workbench/`](../workbench/README.md). One React component
  (`<ResolutionWorkbench />`) plus a standalone app around it. Mounting it in a Carbon host is
  written up in [`../workbench/INTEGRATION.md`](../workbench/INTEGRATION.md); no host has been
  changed.
- **Data:** real review items from the spike, synthetic estate only. `run_a.py` gained
  `--export-review` (the review band with per-field evidence, `review-export/1`) and
  `--export-graph` (the machine clusters as a graph, `graph-export/1`).
- **Covered from the list above:** the queue (ranked, grouped by cluster, filters); the
  evidence card for record pairs (records side by side, per-field match weights, calibrated
  probability against the bands, why the machine was unsure); keyboard-first decisions (same,
  different with a reason, not sure, skip) on an append-only log; the cluster drawing with the
  bridge warning; a read-only audit sample of auto-accepts; decisions per hour.
- **Added on the owner's request:** a graph view. The schema level as a layered map with
  counts, and a walkable neighbourhood view per node; pairs the review band has not decided
  show as open links into their review.
- **Not covered yet:** mentions in text and UC object cards (the spike has no text items in
  the review band yet); AI suggestions; rule proposals from a decision; split and merge as
  applied actions (they are recorded statements); conflict items; audit of auto-rejects with
  precision intervals; four-eyes approval; version certification; reviewer agreement.
- **Checked:** 26 unit and mount tests; a browser run of both builds at phone and desktop
  width in light and dark (82 checks, including WCAG 2.2 AA rule scans with axe).
- **Not checked:** the workbench has not been mounted in a real host. The mount test uses a
  stand-in router, chrome and button.

## Answer (2026-11-10)

The owner reacted to the prototype (seen as screenshots and as a clickable preview) and
settled the interaction model:

- **The direction is right.** Review queue grouped by cluster and ranked by records touched
  times uncertainty; evidence per field with the matcher's weights and a built-in explanation;
  one key per decision on an append-only log.
- **Group actions apply.** One decision for a whole group decides its open pairs: "all one"
  makes them the same; a split, given as a letter per record, makes pairs inside a letter the
  same and pairs across letters different. A pair already decided is never replaced. A group
  decision can be undone, by appended withdrawals. Built as the second cut.
- **The graph is a tab inside the workbench,** next to the review, with links both ways between
  a pair and its organisations. It is a schema map with counts and a neighbourhood view per
  node, walked one step at a time.
- **In Metis it is one mounted component** (own package, the host's router, chrome, controls,
  translation and data passed in). The mount itself waits for the owner's go.

**What the prototype does not cover** moves to the review-surface build slice and to *The
review, certification and versioning lifecycle*: mentions in text and UC object cards;
AI suggestions with a rationale; rule proposals from a decision; acting on the objections a
split records against machine-accepted pairs; conflict items; audit of auto-rejects with
precision intervals; four-eyes approval; version certification; reviewer agreement.

**Checked on the second cut:** 38 unit and mount tests; 106 browser checks on the built pages
at phone and desktop width in light and dark, including WCAG 2.2 AA rule scans. **Not
checked:** a mount in a real host.
