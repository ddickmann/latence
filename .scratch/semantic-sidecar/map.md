---
label: wayfinder:map
title: Latence as a traceable semantic-resolution and ontology engine — standalone and as a Metis sidecar
---

## Destination

**v1 of an Apache-2.0 engine, evolved from Latence, that builds a traceable semantic and
ontology layer** from two input modalities:

- (a) Databricks Unity Catalog, metric views, and warehouse metadata and usage;
- (b) unstructured documents.

**Done** when all seven checks hold (owner, 2026-11-09; *Destination and domain model*):

1. **Trace.** 100% of Concepts, Entities, Bindings, merges and axioms link to at least one
   Assertion and an Actor (engine, model or human). The same inputs run twice give
   byte-identical output.
2. **Curation survives.** A test set of human Decisions is 100% intact after a full rebuild and
   after a delta run.
3. **UC-native.** On Databricks it emits overlays and proposals keyed on UC object ids. Zero
   elements redefine something UC defines. Open-standard exports validate (OWL 2 QL, SHACL,
   SKOS).
4. **One seam.** It runs standalone end to end, and Metis reads it through exactly one declared
   interface.
5. **Text path.** Strictly better than the recorded baseline on extraction F1, linking F1 and
   documents per second.
6. **No retrieval code remains.** Met (2026-11-10): the packages and core modules are gone, no
   script that could run imports a removed package, and a Pipeline that still carries a removed
   config key is refused. One recorded report script under `results/` keeps its old import as
   a dated record; a test exempts it by name.
7. **Resolution quality.** Auto-accept precision is at least 0.95 with its confidence stated,
   on the warehouse side and on the text side. On the synthetic estate the machine alone finds
   at least 60% of the true matches at that precision, so the bar cannot be passed by accepting
   almost nothing. Review rate and expert minutes are reported.

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
- **Resolution is the critical path** (owner, 2026-11-08). No contract or harvester build starts
  before *the resolution feasibility spike* reports go or no-go. The candidate design is
  [design/resolution-at-scale.md](design/resolution-at-scale.md).
  - **The spike reported go** (owner, 2026-11-11): see *Decisions so far*. Building may start.
- **The machine prepares and the expert decides** (owner, 2026-11-08). AI prepares, resolves and
  processes as much as possible, and calibrated bands decide what it may accept alone. The
  expert gets a very convenient workbench to resolve the review band and to QA everything
  against enterprise standards. Success is measured as machine precision **and** expert effort.
- **Standing principles:**
  - **Unity Catalog is the semantic source of truth on Databricks.** This engine overlays what
    UC lacks and proposes changes upstream; it never shadows a UC definition.
  - **No silent over-merge.** An unknown link stays unknown.
  - **English first** (owner, 2026-11-08). v1 is built, measured and gated in English only.
    German follows once English passes the go/no-go. The German items found in research (on
    text extraction, schema matching, ontology learning and relation resolution: a German
    gold set, compound splitting, DE labels) wait in *Not yet specified*. Designs stay
    language-neutral: labels carry a language tag, and nothing hard-codes English, so German
    is an extension, not a rewrite.
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

## Route to ontology v1 and the local demonstration

Charted on 2026-11-11 from the owner's order: **first the ontology capability at v1 quality,
only then the whole demonstration on the host's local backend** (DuckDB and pgvector), after
which the owner updates the main machine and switches to Databricks.

**Owner's decisions for this route (2026-11-11):**

- v1 holds the reviewer loop, a quality pass, a live model call, a Unity Catalog reader and
  usage signals. Document mentions come after v1.
- A small hosted model, and only where the builder needs one. No local models where this is
  built. Spend limit USD 25 for all hosted-model calls of this route; usage is tracked and the
  work stops at the limit.
- Live checks run through the host project's gate queue on the owner's computer.
- Defects found in the host are fixed there, its core included, each with a test that failed
  first and listed for review.
- Tests guard a behaviour someone relies on. No test is written to raise a count.

**Ontology v1.** Its gate is [Decision: is the ontology capability at v1 quality?](issues/34-ontology-v1-acceptance.md). The tickets of this route are listed under "Open work" and, once closed, under "Decisions so far".

**The demonstration on the local backend** follows the gate: the local backend, the launcher and runbook, the browser walks, the live checks and the handoff.

**What this route needs from the owner:** the hand-checked list corrected; one build against the real catalog; the look at v1; the runner started. Each is a ticket under "Open work".

## Open work (12 Nov)

Every open ticket, once, with its state (an override of the format's default that open tickets are found by query).

- [Build: the proposal file for the host, written from approved decisions](issues/40-proposal-file-for-the-host.md): new, before the demonstration: the host's ticket on the ontology feeding its semantic layer waits on it
- [Build: classes and hierarchies are measured against a list a person checked](issues/29-hand-checked-list-and-score.md): built; waits for the owner to correct the hand-checked list
- [Build: a reader for Unity Catalog, proven against a stand-in](issues/33-unity-catalog-reader.md): built and proven against a stand-in; waits for the owner's run on the real catalog
- [Decision: is the ontology capability at v1 quality?](issues/34-ontology-v1-acceptance.md): seven of nine checks met; the owner's decision is open
- [Build: every use case walked in a browser on the local backend](issues/37-every-use-case-walked.md): 28 browser specs pass; still to do: the acceptance run and the host's other tests that need Postgres
- [Build: live checks on real models, through the owner's runner](issues/38-live-checks-through-the-gate-queue.md): open: the requests through the owner's runner (the landing gate with the guard checks, the user-interface lane, one stability run)
- [Build: handoff to the main machine and the switch to Databricks](issues/39-handoff-to-the-main-machine.md): open: first the owner's three check commands on the main machine, then the handoff document
- [Task: defects the documentation pass found and did not fix](issues/24-defects-found-by-the-documentation-pass.md): open: a spend cap as a hard stop is the owner's to decide; the compose and Helm commands were not run; five defects found and not fixed
- [Task: record the baseline: one text-path run and ULTRA's real link-prediction numbers](issues/08-baseline-measurement.md): open: the baseline of the text path and of link prediction is not recorded
- [Task: get a Databricks sandbox for the engine](issues/09-databricks-sandbox-access.md): open
- [Grilling: the upgrade of the resolution engine](issues/13-resolution-engine-upgrade.md): open grilling; the feasibility spike's input is recorded in it
- [Grilling: the text-path upgrade](issues/15-text-path-upgrade.md): open grilling; blocked by the missing baseline
- [Grilling: the review, certification and versioning lifecycle](issues/16-review-and-versioning-lifecycle.md): open grilling
- [Build: the catalog reader reads lineage, row filters and column masks](issues/41-catalog-reader-lineage-filters-and-masks.md): new, after v1
- [Build: what ontology v1 leaves out: document mentions, then SHACL and PROV-O exports](issues/42-what-ontology-v1-leaves-out.md): new, after v1

## The owner's decisions (12 Nov)

The owner answered on 12 Nov: every recommendation is agreed. Each decision is recorded in its ticket, with the brief behind it. The demonstration runs on Databricks; the host's main machine makes sure it works there.

**Still open**

| The question | Ticket | What it needs |
|---|---|---|
| In which order follows what v1 leaves out, and when do the grillings resume? | [Build: what ontology v1 leaves out: document mentions, then SHACL and PROV-O exports](issues/42-what-ontology-v1-leaves-out.md) | Taken up after the demonstration |

**The owner's actions** (in the order that unblocks most)

| The action | Ticket | What it costs him |
|---|---|---|
| Run the three check commands on the main machine; push its main branch if it is ahead | [Build: handoff to the main machine and the switch to Databricks](issues/39-handoff-to-the-main-machine.md) | Three commands, perhaps one push |
| Start the runner with both lanes; the checks run at low reasoning | [Build: live checks on real models, through the owner's runner](issues/38-live-checks-through-the-gate-queue.md) | Starting the runner; spend inside the USD 25 limit |
| Correct the hand-checked list and sign it | [Build: classes and hierarchies are measured against a list a person checked](issues/29-hand-checked-list-and-score.md) | 52 entries and 3 unsure ones, then four commands |
| Declare ontology v1 once the list's score is recorded | [Decision: is the ontology capability at v1 quality?](issues/34-ontology-v1-acceptance.md) | A signature |
| Export the host's workspace credentials in one shell (read only) | [Task: get a Databricks sandbox for the engine](issues/09-databricks-sandbox-access.md) | One shell |
| Run one build against the real catalog, as the first step of the switch to Databricks | [Build: a reader for Unity Catalog, proven against a stand-in](issues/33-unity-catalog-reader.md) | Two commands on the main machine |

**Decided, with nothing to do now:** the benchmark campaign's spend cap is decided when the campaign is next used; the resolution workbench is mounted in the host after v1, with the review-surface slice.

## Decisions so far

<!-- the index — one line per closed ticket: enough to judge relevance, then zoom the link -->

- [ADR-0067 — Latence is re-scoped into a semantic-resolution and ontology engine, standalone and as a Metis sidecar](../../docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md) — the direction above. Warehouse metadata is a new modality; the text path and Graph Completion stay.
- [ADR-0068 — Query-time retrieval tooling is removed](../../docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) — retrieval, hyperedges and Context Enrichment go. Embedders stay as a resolution signal only.
- [Destination and domain model](issues/10-destination-and-domain-model.md) — two levels: Entities (the engine may accept merges in the calibrated band) and Concepts
  - Recorded as [ADR-0069 — The semantic layer is a replay of an append-only ledger](../../docs/adr/0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md).
- [Boundary with Metis](issues/11-boundary-with-metis.md) — one Metis-specific seam: a reviewable proposal file in Metis's own proposal shape. The crosswalk and the ontology reach Metis through UC Delta tables. No shared code
  - Recorded as [ADR-0070 — Metis consumes the engine through one proposal file](../../docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md).
- [Warehouse pipeline shape](issues/12-warehouse-pipeline-shape.md) — v1 harvests UC metadata, metric views, REST lineage, Genie agent configuration and existing glossaries; query history is optional. Harvesters are Source kinds in the one Pipeline and meet text mentions in one resolution phase
  - Recorded as [ADR-0071 — Warehouse metadata enters the Pipeline as Sources](../../docs/adr/0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md).
- [Evaluation and the acceptance bars](issues/18-evaluation-and-acceptance.md) — three layers of gold: our own seeded synthetic data (where the hard bars are measured), a small expert-labelled realistic slice, and public benchmarks for comparability. Current Latence is the only gating baseline
- [Runtime, storage and deployment](issues/17-runtime-storage-deployment.md) — the local Runner stays and a Databricks Jobs Runner is the production path; Airflow stays unchanged. The ledger, the results and the review state are Delta tables in UC
  - Recorded as [ADR-0072 — On Databricks the engine runs as jobs and keeps its state in Delta](../../docs/adr/0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md).
- [Ontology induction and the output standards](issues/14-ontology-induction-and-standards.md) — declared constraints, data statistics and names vote, with the hosted model as one voice. Every proposed constraint is run against the data before review, and weak ones are dropped
  - Recorded as [ADR-0073 — The schema level is induced by voting signals, validated on data, and exported in W3C standards](../../docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md).
- [Prototype: the expert resolution workbench](issues/22-expert-resolution-workbench.md) — the owner settled the interaction model on a clickable prototype: a queue by cluster, evidence per field, one-key decisions on an append-only log, group merge and split that decide the open pairs and can be undone
- [Remove the retrieval tooling](issues/19-remove-retrieval-tooling.md) — done: four packages, the core retrieval modules, contracts and seams, three stacks and the retrieval benchmark lanes are gone (`SCHEMA_VERSION` 22, 30 packages). Kept pipelines give the same nodes and edges
- [Prototype: resolution feasibility spike](issues/20-resolution-feasibility-spike.md) — **go** (owner, 2026-11-11). Auto-accept precision 0.986 to 0.990 at 10⁵ records, 0.9955 at 10⁶ and 0.96 on text; the machine alone finds 71% to 90% of true matches
- [The Ontology page over the estate](issues/25-estate-demo-slice.md) (resolved; what is open continues in the route above) — the host writes the page in its own style and shares only the export's file format with the engine; read-only, off the answer path; a file first, a service route later
  - Recorded as [ADR-0074 — A host shows the engine's export in a read-only screen of its own](../../docs/adr/0074-a-host-shows-the-engines-export-in-a-read-only-screen-of-its-own-off-the-answer-path.md).
- [Task: what the retrieval removal left behind](issues/23-retrieval-removal-leftovers.md) — what the removal missed is closed: the release gates, the purge, the benchmark leftovers, stale help texts, two run-time defects, `ROADMAP.md`, anatomy chapters 2 and 3
- [Build: the export says what it means in codes, and every proposal carries its claim](issues/26-export-contract-codes-and-claims.md) — the export speaks in codes, and every proposal carries its claim (`graph-export/3`)
- [Build: the defects the vehicle estate shows in what the engine proposes](issues/27-quality-fixes-on-the-estate.md) — six defects in what the engine proposes on the vehicle estate, each gone with its test
- [Build: a reviewer's decision on a hierarchy step or a class changes the layer](issues/28-decisions-on-structure-and-classes.md) — a decision on a hierarchy step or a class changes the layer
- [Build: the matcher asks for the labels it needs and takes them from a reviewer](issues/30-label-loop.md) — the matcher asks for labels and takes them from a reviewer
- [Build: the model's voice on classes runs live on a small hosted model](issues/31-live-class-voice.md) — the model's voice on classes runs live on a small hosted model
- [Build: what people ask is a signal on the elements they use](issues/32-usage-signals.md) — what people ask is a signal on the elements they use
- [Build: the host's local backend runs on this machine](issues/35-local-backend-for-the-host.md) — Postgres with pgvector on this machine, for the host's local backend
- [Build: one command starts the demonstration, one page says how to give it](issues/36-demo-launcher-and-runbook.md) — one command starts the demonstration, one page says how to give it
- [Inventory the carve-out](issues/07-carve-out-inventory.md) — no kept package imports a dropped one. Retrieval reaches core through about five import sites plus protocols, hyperedge contracts (`SCHEMA_VERSION` 21), 19 test files and 6 stacks. `graph_features` goes with retrieval; the embedders stay as a resolution signal.
- [The landscape: who builds semantic and ontology layers](issues/06-landscape-and-positioning.md) — every rival traces answers, not semantic elements, and auto-drafting tools lose curation on rebuild. Positioning: a ledger of evidence and decisions that proposes into UC and exports open formats
- [The state of the art in text extraction](issues/02-sota-text-extraction.md) — the glinker linker (73%) and per-chunk LLM induction are the bottlenecks, not extraction. GLiNER-Relex leads zero-shot relation extraction, and the GLiNER bi-encoder removes the label-count ceiling
- [The state of the art in entity resolution](issues/04-sota-entity-resolution.md) — embeddings retrieve, cross-encoders decide. Calibrated probabilities with Learn-then-Test give accept, review and reject bands, and an LLM is used only in the review band
- [The state of the art in schema matching](issues/03-sota-schema-matching.md) — cheap candidates first, then a model judges a short list, and rules drive precision. Metis already has the statistical core (exact key and inclusion checks, calibrated owner decisions)
- [Ontology learning methods and the standards to emit](issues/05-ontology-learning-and-standards.md) — no method is good enough to auto-accept (taxonomy F1 about 0.5–0.6). Warehouse constraints and evidence outrank models
- [The Databricks semantic surface](issues/01-databricks-semantic-surface.md) — reading needs no system tables (REST, `information_schema` and `DESCRIBE AS JSON` cover structure, metric views, tags, keys and certification). Metric views are GA, and UC stores them normalised
- [Relation resolution and canonicalization at scale](issues/21-relation-resolution-at-scale.md) — today one edge is emitted per mention and the fuzzy label match is unsafe (`not_located_in` folds into `located_in`). The unit of work is the distinct (label, head class, tail class) triple

## Not yet specified

- **Build slices**, which graduate as the grillings close:
  - contract generalisation: a structured locator next to char offsets, an actor on every
    decision, and the ontology contracts;
  - warehouse harvesters;
  - the resolution engine upgrade: matcher, calibration, constraints, scale;
  - ontology induction;
  - writers and UC write-back;
  - the review surface (the prototype in `workbench/` settled the interaction model; what it
    leaves out is listed in *Prototype: the expert resolution workbench*);
  - the Databricks runner and its deployment;
  - the Metis integration;
  - the extraction Provider swap.
- Migrating the shipped stacks, the setup wizard and the CLI to the new product.
- The rest of the docs and site rewrite. Done on 2026-11-10: README, docs index, ROADMAP and
  the CONTEXT.md glossary describe the current code first and the decided direction second;
  the product keeps the name Latence for now (owner). Still old: `AGENTS.md`'s opening
  framing and the pages outside the site navigation.
- Release and versioning of the re-scoped engine, given ADR-0066's gates.
- **The German extension**, once English passes:
  - our own German gold sets for entities, relations and merges (no German relation benchmark
    exists, and REBEL and WebNLG don't allow commercial use);
  - German compound splitting for names and headers;
  - re-measuring the multilingual extractors and the cross-encoder;
  - DE labels and definitions in the ontology.
- Other warehouses (Snowflake, Postgres, Trino) behind the warehouse seam, once the Databricks
  path is proven.

## Out of scope

- Query-time retrieval tooling (owner, 2026-11-08; ADR-0068).
- Answering questions over data (NL-to-SQL, agents). Metis and Genie own that; this engine
  feeds them.
- BI dashboards and report generation.
