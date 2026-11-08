---
label: wayfinder:map
title: Latence as a traceable semantic-resolution and ontology engine — standalone and as a Metis sidecar
---

## Destination

**v1 of an Apache-2.0 engine, evolved from Latence, that builds a traceable semantic and
ontology layer** from two input modalities:

- (a) Databricks Unity Catalog, metric views, and warehouse metadata and usage;
- (b) unstructured documents.

**Done** when (draft; *Destination and domain model* refines it):

1. Every concept, alias, binding, merge and axiom links to Evidence and to the actor that made
   it (engine, model or human). A run replays byte-identically.
2. Human decisions survive rebuilds and delta runs.
3. On Databricks it emits UC-native overlays and proposals keyed on UC object ids, never a
   parallel definition of something UC defines. It also emits open-standard exports.
4. It runs standalone, and as a sidecar that feeds Metis's semantic catalog through one
   declared seam.
5. The upgraded text path beats the recorded baseline on quality and throughput.
6. No retrieval code remains.

## Notes

- **Execution is carried into this map** (it overrides "plan, don't do"). Decision tickets come
  first. Build tickets (type `task`, AFK, one vertical slice with tests each) graduate from
  *Not yet specified* once the decisions they depend on close.
- **Grilling:** one AskUserQuestion per decision, with the recommended option first. Save
  progress to the ticket every few answers. When the owner answers with a criterion instead of
  an option, decide, and record "owner delegated with criterion X; chose Y".
- **Tracker:** local markdown, per `issue-tracker-local.md`.
  - Tickets live at `issues/NN-<slug>.md` with `Type:` / `Mode:` / `Status:` / `Blocked by:`
    lines, then `## Question`. The answer goes under `## Answer`.
  - Research reports live at `research/NN-<slug>.md`, linked from their ticket.
  - Never truncate-then-read when scripting ticket edits. After any multi-file edit, check that
    every ticket still starts with `# ` and has `## Question`.
- **Hygiene:**
  - No mention of the coding assistant or its vendor in commits, docs or reports, and no
    attribution trailers.
  - Neutral names: no client or employer names.
  - Push only to the designated branch.
- **Latence's gates apply to every build:**
  - `scripts/verify-local.sh`;
  - the doc-truth tests (any count a page states must equal the tree, and linked files must be
    `git add`ed);
  - a `SCHEMA_VERSION` bump on any contract change;
  - no fake green, and every fix needs a test that was red first.
  - `AGENTS.md` is the authority.
- **Standing principles:**
  - **Unity Catalog is the semantic source of truth on Databricks.** This engine overlays what
    UC lacks and proposes changes upstream; it never shadows a UC definition.
  - **No silent over-merge.** An unknown link stays unknown.
  - **German and English** throughout.
  - Sending samples to a model needs an explicit egress decision.
- **Reference reads** (read, never copied):
  - Metis: tickets 94, 99, 140, 147, 203, 207 and 211 in `metis/.scratch/metis/issues/`, and
    `metis-core/src/metis_core/{semantics,layer}/`.
  - Prior research: `metis-companion/research-holding/2026-11-03/`, which holds the review of
    an open-source auto-ontology project and the ticket 206 problem classes.
  - These pointers live only here, never in shipped docs.
- **Owner decisions of 2026-11-08** that charted this map:
  - sidecar plus standalone;
  - the text path stays, with an upgraded extraction stack;
  - ULTRA graph completion stays;
  - warehouse and Databricks ingestion is new;
  - the whole retrieval part goes.

## Decisions so far

<!-- the index — one line per closed ticket: enough to judge relevance, then zoom the link -->

- [ADR-0067 — Latence is re-scoped into a semantic-resolution and ontology engine, standalone and as a Metis sidecar](../../docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md) — the direction above. Warehouse metadata is a new modality; the text path and Graph Completion stay.
- [ADR-0068 — Query-time retrieval tooling is removed](../../docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) — retrieval, hyperedges and Context Enrichment go. Embedders stay as a resolution signal only.
- [Inventory the carve-out](issues/07-carve-out-inventory.md) — no kept package imports a dropped one. Retrieval reaches core through about five import sites plus protocols, hyperedge contracts (`SCHEMA_VERSION` 21), 19 test files and 6 stacks. `graph_features` goes with retrieval; the embedders stay as a resolution signal.
- [The landscape: who builds semantic and ontology layers](issues/06-landscape-and-positioning.md) — every rival traces answers, not semantic elements, and auto-drafting tools lose curation on rebuild. Positioning: a ledger of evidence and decisions that proposes into UC and exports open formats. Three demo claims: element-level trace and replay; curation that survives rebuilds; add-only proposals on UC.
- [The state of the art in text extraction](issues/02-sota-text-extraction.md) — the glinker linker (73%) and per-chunk LLM induction are the bottlenecks, not extraction. GLiNER-Relex leads zero-shot relation extraction, and the GLiNER bi-encoder removes the label-count ceiling. Fabricated relations are fixed by requiring offset-grounded triples plus an entailment check. Bake-off shortlist: Relex + bi-encoder, then with an LLM checker, then a small-LLM extractor, with GLiNER 2.5 as the control.

## Not yet specified

- **Build slices**, which graduate as the grillings close:
  - contract generalisation: a structured locator next to char offsets, an actor on every
    decision, and the ontology contracts;
  - warehouse harvesters;
  - the resolution engine upgrade: matcher, calibration, constraints, scale;
  - ontology induction;
  - writers and UC write-back;
  - the review surface;
  - the Databricks runner and its deployment;
  - the Metis integration;
  - the extraction Provider swap.
- Migrating the shipped stacks, the setup wizard and the CLI to the new product.
- A docs and site rewrite (README, docs index, ROADMAP, CONTEXT.md glossary), and whether the
  product keeps its name.
- Release and versioning of the re-scoped engine, given ADR-0066's gates.
- Other warehouses (Snowflake, Postgres, Trino) behind the warehouse seam, once the Databricks
  path is proven.

## Out of scope

- Query-time retrieval tooling (owner, 2026-11-08; ADR-0068).
- Answering questions over data (NL-to-SQL, agents). Metis and Genie own that; this engine
  feeds them.
- BI dashboards and report generation.
