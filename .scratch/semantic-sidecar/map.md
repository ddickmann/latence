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
    German follows once English passes the go/no-go. The German items found in research
    (02, 03, 05, 21: a German gold set, compound splitting, DE labels) wait in *Not yet
    specified*. Designs stay language-neutral: labels carry a language tag, and nothing
    hard-codes English, so German is an extension, not a rewrite.
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

**Ontology v1** (the gate is ticket 34; nothing below it starts before the owner has seen it):

| Ticket | What | Mode | Blocked by, or state |
|---|---|---|---|
| [26](issues/26-export-contract-codes-and-claims.md) | the export speaks in codes, every proposal carries its claim (`graph-export/3`) | AFK | **done** |
| [27](issues/27-quality-fixes-on-the-estate.md) | six defects in what the engine proposes on the vehicle estate | AFK | **done** |
| [28](issues/28-decisions-on-structure-and-classes.md) | a decision on a hierarchy step or a class changes the layer | AFK | **done** |
| [29](issues/29-hand-checked-list-and-score.md) | classes and hierarchies scored against a list the owner checked | HITL | built; **waits for the owner to correct the list** |
| [30](issues/30-label-loop.md) | the matcher asks for labels and takes them from a reviewer | AFK | 26, 28 |
| [31](issues/31-live-class-voice.md) | the model's voice on classes runs live on a small hosted model | HITL | 26 |
| [32](issues/32-usage-signals.md) | what people ask is a signal on the elements they use | AFK | 26 |
| [33](issues/33-unity-catalog-reader.md) | a reader for Unity Catalog, proven against a stand-in | HITL | 26 |
| [34](issues/34-ontology-v1-acceptance.md) | **is the capability at v1 quality?** | HITL | 26–33 |

**The demonstration on the local backend:**

| Ticket | What | Mode | Blocked by |
|---|---|---|---|
| [35](issues/35-local-backend-for-the-host.md) | Postgres with pgvector on this machine | AFK | 34 |
| [36](issues/36-demo-launcher-and-runbook.md) | one command starts it, one page says how to give it | AFK | 35 |
| [37](issues/37-every-use-case-walked.md) | every use case walked in a browser; defects fixed | AFK | 36 |
| [38](issues/38-live-checks-through-the-gate-queue.md) | live checks on real models through the owner's runner | HITL | 37 |
| [39](issues/39-handoff-to-the-main-machine.md) | handoff and the switch to Databricks | HITL | 38 |

**What this route needs from the owner:** the hand-checked list corrected (29); the key through
the secure form (31); one build against the real catalog (33); the look at v1 (34); the runner
started (38).

## Decisions so far

<!-- the index — one line per closed ticket: enough to judge relevance, then zoom the link -->

- [ADR-0067 — Latence is re-scoped into a semantic-resolution and ontology engine, standalone and as a Metis sidecar](../../docs/adr/0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md) — the direction above. Warehouse metadata is a new modality; the text path and Graph Completion stay.
- [ADR-0068 — Query-time retrieval tooling is removed](../../docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) — retrieval, hyperedges and Context Enrichment go. Embedders stay as a resolution signal only.
- [ADR-0069 — The semantic layer is a replay of an append-only ledger](../../docs/adr/0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md) and [Destination and domain model](issues/10-destination-and-domain-model.md) — two levels: Entities (the engine may accept merges in the calibrated band) and Concepts (classes, properties, measures; proposals a human approves). Assertions and Decisions are append-only and the layer is their replay. Bindings are keyed on UC ids, for Concepts and for Entity rows (the crosswalk). The engine never defines a table or metric; drafting new measures as proposals is in v1. Seven measurable Done checks.
- [ADR-0070 — Metis consumes the engine through one proposal file](../../docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md) and [Boundary with Metis](issues/11-boundary-with-metis.md) — one Metis-specific seam: a reviewable proposal file in Metis's own proposal shape. The crosswalk and the ontology reach Metis through UC Delta tables. No shared code; with Metis present the engine reads its verified keys and joins as Assertions. One route per source for new measures. No importer plugin, live tool calls or interchange-file seam in v1.
- [ADR-0071 — Warehouse metadata enters the Pipeline as Sources](../../docs/adr/0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md) and [Warehouse pipeline shape](issues/12-warehouse-pipeline-shape.md) — v1 harvests UC metadata, metric views, REST lineage, Genie agent configuration and existing glossaries; query history is optional. Harvesters are Source kinds in the one Pipeline and meet text mentions in one resolution phase. Row values are matched inside the customer's workspace on listed master-data tables. Filters and masks are labelled and masked data is excluded. Signals come in three tiers; the hosted referee sees metadata only and is off by default.
- [Evaluation and the acceptance bars](issues/18-evaluation-and-acceptance.md) — three layers of gold: our own seeded synthetic data (where the hard bars are measured), a small expert-labelled realistic slice, and public benchmarks for comparability. Current Latence is the only gating baseline; the Metis effect is measured inside Metis and reported. Three hard bars: precision at least 0.95, the text path better than the baseline, and machine-only recall of at least 0.60 on the synthetic estate. Everything else is reported until the release candidate.
- [ADR-0072 — On Databricks the engine runs as jobs and keeps its state in Delta](../../docs/adr/0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md) and [Runtime, storage and deployment](issues/17-runtime-storage-deployment.md) — the local Runner stays and a Databricks Jobs Runner is the production path; Airflow stays unchanged. The ledger, the results and the review state are Delta tables in UC. The review surface is one web app: an in-workspace app in production, local for standalone use. Neural models run on GPU job compute in the workspace. Review explanations come from the matcher's per-field evidence by default; hosted ones are opt-in. App and GPU-compute capabilities are still to be confirmed on a sandbox.
- [ADR-0073 — The schema level is induced by voting signals, validated on data, and exported in W3C standards](../../docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md) and [Ontology induction and the output standards](issues/14-ontology-induction-and-standards.md) — declared constraints, data statistics and names vote, with the hosted model as one voice. Every proposed constraint is run against the data before review, and weak ones are dropped. v1 exports OWL 2 QL, SHACL 1.1, SKOS, PROV-O and metric-view YAML. IRIs are opaque under a per-customer prefix; the private `lg:` vocabulary goes.
- [Prototype: the expert resolution workbench](issues/22-expert-resolution-workbench.md) — the owner settled the interaction model on a clickable prototype: a queue by cluster, evidence per field, one-key decisions on an append-only log, group merge and split that decide the open pairs and can be undone, and the graph as a tab beside the review. It is one component a Carbon host mounts; the mount in Metis waits for the owner's go.
- [Remove the retrieval tooling](issues/19-remove-retrieval-tooling.md) — done: four packages, the core retrieval modules, contracts and seams, three stacks and the retrieval benchmark lanes are gone (`SCHEMA_VERSION` 22, 30 packages). Kept pipelines give the same nodes and edges. The ten stale pages were rewritten and the secret-scan gate was made independent of the network on 2026-11-10. What the removal missed was closed in [What the retrieval removal left behind](issues/23-retrieval-removal-leftovers.md): the release gates, the purge, the benchmark leftovers (including the two datasets that only fed retrieval), the stale help texts, two run-time defects, `ROADMAP.md`, anatomy chapters 2 and 3, and the doc statements that had not been checked against code. What that pass found is in [Defects the documentation pass found and did not fix](issues/24-defects-found-by-the-documentation-pass.md): eight of nine items are fixed (removed config keys are refused, two tuner and two campaign defects, dead benchmark scripts, two stale tests, an unsupported speed figure). Open: the compose and Helm commands have never been executed, and whether a spend cap in the benchmark campaign should be a hard stop.
- [Prototype: resolution feasibility spike](issues/20-resolution-feasibility-spike.md) — **go** (owner, 2026-11-11). Auto-accept precision 0.986 to 0.990 at 10⁵ records, 0.9955 at 10⁶ and 0.96 on text; the machine alone finds 71% to 90% of true matches; 10⁶ records fit the 4-hour budget on one 24 GB GPU. Splink is the backbone, GLiDE 0.4B the decision model, the reranker second; zero-shot Decide is dropped. Labels drawn from the undecided pairs replace uniform labels. Shown on synthetic data and one public person dataset, where the models add nothing; real data follows. Report: [spike/REPORT.md](spike/REPORT.md). **Put right the same day:** on two real company registers the first run found 36.5% of the true links; a scorer fitted on the same number of reviewer labels finds 97% at 99.7% precision and beats every earlier arm on the other estates, on CPUs ([research note](research/24-company-record-linkage-methods.md)).
- [ADR-0074 — A host shows the engine's export in a read-only screen of its own](../../docs/adr/0074-a-host-shows-the-engines-export-in-a-read-only-screen-of-its-own-off-the-answer-path.md) and [The Ontology page over the estate](issues/25-estate-demo-slice.md) (resolved; what is open continues in the route above) — the host writes the page in its own style and shares only the export's file format with the engine; read-only, off the answer path; a file first, a service route later; reviewer decisions travel back as a file. Cut A induces the schema level of the vehicle and warranty estate from its metric views and data statistics: 38 concepts, 86 relations with their support, 38 measures. Cut B added entities, counted fact edges, a seeded second source linked under a certified line, hierarchies the data proves, a class tree proposed by two voices and a Turtle export. Document mentions are open and not in v1.
- [Inventory the carve-out](issues/07-carve-out-inventory.md) — no kept package imports a dropped one. Retrieval reaches core through about five import sites plus protocols, hyperedge contracts (`SCHEMA_VERSION` 21), 19 test files and 6 stacks. `graph_features` goes with retrieval; the embedders stay as a resolution signal.
- [The landscape: who builds semantic and ontology layers](issues/06-landscape-and-positioning.md) — every rival traces answers, not semantic elements, and auto-drafting tools lose curation on rebuild. Positioning: a ledger of evidence and decisions that proposes into UC and exports open formats. Three demo claims: element-level trace and replay; curation that survives rebuilds; add-only proposals on UC.
- [The state of the art in text extraction](issues/02-sota-text-extraction.md) — the glinker linker (73%) and per-chunk LLM induction are the bottlenecks, not extraction. GLiNER-Relex leads zero-shot relation extraction, and the GLiNER bi-encoder removes the label-count ceiling. Fabricated relations are fixed by requiring offset-grounded triples plus an entailment check. Bake-off shortlist: Relex + bi-encoder, then with an LLM checker, then a small-LLM extractor, with GLiNER 2.5 as the control.
- [The state of the art in entity resolution](issues/04-sota-entity-resolution.md) — embeddings retrieve, cross-encoders decide. Calibrated probabilities with Learn-then-Test give accept, review and reject bands, and an LLM is used only in the review band. Human must-links and cannot-links are stored on stable keys, replayed first and used as hard vetoes. Homonym splits are proposed. Deltas use a pair cache and stable ids by overlap. Splink (MIT) suits the warehouse side.
- [The state of the art in schema matching](issues/03-sota-schema-matching.md) — cheap candidates first, then a model judges a short list, and rules drive precision. Metis already has the statistical core (exact key and inclusion checks, calibrated owner decisions). v1 adds five signals that send no values out: UC structure and declared keys, the lexicon, multilingual embeddings, column lineage and query-history mining. A model judge that sees metadata only is optional and gated by egress. Learned column-type models and any model that reads values are deferred.
- [Ontology learning methods and the standards to emit](issues/05-ontology-learning-and-standards.md) — no method is good enough to auto-accept (taxonomy F1 about 0.5–0.6). Warehouse constraints and evidence outrank models. Standards: OWL 2 QL (querying over Databricks via Ontop), SHACL 1.1, SKOS DE/EN, PROV-O, SSSOM, and named graphs now with RDF 1.2 later. Opaque IRIs and content-hashed proposals. Ossie's ontology spec is an unreleased draft. P4 sketch: an OntologyProposal contract over ten stages. The preserved inducer code is not in the tree.
- [The Databricks semantic surface](issues/01-databricks-semantic-surface.md) — reading needs no system tables (REST, `information_schema` and `DESCRIBE AS JSON` cover structure, metric views, tags, keys and certification). Metric views are GA, and UC stores them normalised. Query history is admin-only, with text redacted by default. Genie Agent configuration is readable as JSON, but Genie Ontology has no export API. Write-back is plain SQL with no staging, so proposals live in our store. The overlay key is `table_id` plus the field name; whether it survives a rename is unverified. System tables are an optional harvester.
- [Relation resolution and canonicalization at scale](issues/21-relation-resolution-at-scale.md) — today one edge is emitted per mention and the fuzzy label match is unsafe (`not_located_in` folds into `located_in`). The unit of work is the distinct (label, head class, tail class) triple. EDC-style definition retrieval plus a calibrated scorer, with domain and range as vetoes and unmapped relations as proposals. One aggregated edge per fact (Knowledge Vault fusion, functional properties). Contradictions go to review. Joint ER is bounded to two rounds. UC foreign keys and joins become schema-level properties.

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
