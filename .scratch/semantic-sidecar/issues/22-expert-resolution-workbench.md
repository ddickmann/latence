# Prototype: the expert resolution workbench (UI/UX)

Type: prototype
Mode: HITL
Status: open
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
