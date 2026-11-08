# Research 06: the landscape, and what each product makes traceable

As of 8 Nov 2026. Resolves [ticket 06](../issues/06-landscape-and-positioning.md).

**Method and limits.** Web search (standard, then extended). `docs.databricks.com`,
`databricks.com` and `learn.microsoft.com` did not resolve from this network, so Databricks and
Fabric claims come from search-result summaries of those pages, not a full read. Claims no primary
source confirmed are **UNVERIFIED**. Prior research (the Auto Ontology review, R1, R2, R4) is
summarised, not redone.

## Summary

- **Each vendor makes the answer traceable, not the semantic element.** Genie Ontology shows each
  snippet's origin, author and authority score; Palantir logs who ran each action; Snowflake keeps
  the verified queries a view came from. None records, per alias, merge or axiom, the evidence
  span and the actor, with a replayable run.
- **Curation and generation share the same store in every auto-drafting tool.**
  - NVIDIA Auto Ontology loses certified edits on a rebuild (its own issue #259) and has no
    element-level provenance (#260).
  - Fabric IQ generates an ontology once from a semantic model, after which the user edits it by
    hand. What happens on regeneration is not documented (UNVERIFIED).
  - Genie Ontology keeps the human layer (UC Pages) apart from the inferred layer, which is the
    right shape, but the inferred snippets are a closed, continuously refreshed store. No export or
    write API for them was found (UNVERIFIED).
- **Few products take in both warehouse usage and documents and turn them into one ontology.**
  - Genie Ontology does, but only inside Databricks, and its store is closed.
  - Stardog and Neo4j extract from documents but are weak on warehouse usage.
  - Snowflake Autopilot and NVIDIA Auto Ontology mine query history but read no documents.
- **The open standards are moving.** Apache Ossie (the former OSI) has an Ontology working group
  but no ontology spec. Snowflake reads and writes Ossie natively; Fabric imports TTL/RDF/OWL and
  exports TTL/RDF. No Databricks metric-view converter for Ossie was found.
- **Positioning:** an evidence-and-decision ledger that drafts ontology elements, keeps human
  decisions across rebuilds, and pushes proposals into UC (and exports) for Genie and Metis.

## Comparison table

Columns:
- **Ingests:** M = metadata, S = data samples, Q = query logs or usage, D = documents, A = BI and
  notebook assets.
- **Trace (src / actor):** whether each element links to its source evidence and to the actor that
  made it.

| Product | Ingests | Outputs | Trace (src / actor) | Curation survives rebuild | Standards in / out | Licence / deploy | Status |
|---|---|---|---|---|---|---|---|
| **Databricks Genie Ontology** (+ UC Pages) | M, Q, A (dashboards, notebooks, Genie agents, pipelines), D (docs, Pages) | Inferred "snippets" ranked by authority (OntoRank), plus human Pages; cited in Genie answers | Snippet shows origin, author and authority score; answer-level citations. No element lineage view and no replay | Pages are human-owned UC objects and take precedence. Whether edits to inferred snippets persist: UNVERIFIED | In: UC assets. Out: Genie One MCP for consumption. Snippet export or API not found (UNVERIFIED) | Proprietary, Databricks-only | Ontology Public Preview; Pages Beta |
| **UC semantics** (metric views, glossary, domains, certification tags) | M (human-authored) | Metric views, domains, certified/deprecated system tags, glossary | Object-level audit logs. No evidence for a definition | Yes, human-authored by construction | Metric views being open-sourced into Spark. No Ossie converter found | Proprietary service; UC OSS is Apache-2.0 | Metric views GA; domains Public Preview; glossary status unclear |
| **NVIDIA Auto Ontology** | M, S (1,000 rows), Q (24 h); optional YAML or Ossie import. No documents in code | Terms (one per table), column descriptions, semantic FKs, mined SQL attributes | **No** element provenance (#260). LLM calls not recorded, so no replay | **No**: certified edits lost on rebuild (#259) | In: native YAML, Ossie 0.2. Out: YAML, Ossie | Apache-2.0; self-hosted Compose/Helm stack | No releases; no external contributions |
| **Palantir Foundry Ontology** | Datasets (pipeline outputs), models, actions | Object, link, action and interface types; OSDK | Dataset lineage; action log and object edit history (opt-in, not pipeline edits). Type definitions: proposal history | Yes: hand-modelled, changed via branch proposals with review | Proprietary; OSDK and APIs out. No OWL/RDF (UNVERIFIED) | Proprietary SaaS or on-prem | GA; Foundry Branching limited beta |
| **Microsoft Fabric IQ ontology** | M (semantic models, lakehouse, warehouse, eventhouse, OneLake) | Entity types, properties, relationships, rules, bindings | Bindings to sources. No evidence or actor per element found | Generated once from a semantic model, then hand-edited. Behaviour on regeneration: UNVERIFIED | In: TTL, RDF, OWL (lossy subset). Out: TTL, RDF | Proprietary SaaS (Fabric) | Preview |
| **Snowflake semantic views + Autopilot / Cortex Analyst** | M, Q (role's query history), user-supplied SQL | Semantic view: tables, relationships, facts, dimensions, metrics, synonyms, verified queries | Verified queries trace to the SQL they came from. Per-element actor not found | Human-authored DDL persists. Merging an Autopilot re-run: UNVERIFIED | Ossie YAML in and out (native functions); vendor extension for non-Ossie parts | Proprietary SaaS | Semantic views GA; Autopilot GA Feb 2026 (third-party date) |
| **dbt Semantic Layer / MetricFlow** | M (dbt project YAML) | Semantic models, metrics; compiled SQL | Git history of hand-written YAML. Nothing drafted, so nothing to trace | Yes (code) | Ossie documents accepted from dbt Core 1.12. Ossie↔MetricFlow converter | MetricFlow Apache-2.0 from 0.209; hosted APIs proprietary | GA |
| **Timbr** | M (warehouse schemas) | SQL ontology: concepts, inheritance, relationships, measures as virtual schemas | Model-level only (UNVERIFIED) | Yes: hand-modelled | SQL/JDBC/ODBC/REST out; OWL import not found (UNVERIFIED) | Commercial; licence and deployment UNVERIFIED | GA |
| **Stardog** (Designer, Voicebox) | M via virtual graphs (incl. Databricks), D (ontology-driven extraction, v12) | RDF knowledge graph, ontology suggestions from competency questions | RDF named graphs make it possible. Default provenance per element: UNVERIFIED | Yes for hand-modelled parts. For LLM suggestions: UNVERIFIED | OWL, RDF, SHACL, SPARQL in and out | Commercial (free tier); cloud or self-hosted | GA (Voicebox service 1.1 GA Sep 2026) |
| **Neo4j** (LLM Graph Builder, GraphRAG Python) | D (chunks); schema given or LLM-extracted | Property graph of entities linked to source chunks | Entity→chunk links (UNVERIFIED in this pass). No actor or replay | Re-extraction overwrites (UNVERIFIED) | Cypher; RDF via the n10s plugin | Builder Apache-2.0 (conflicting sources); DB GPLv3 or commercial | Labs project; schema extraction "experimental" |
| **Graphwise** (GraphDB + PoolParty) | D (auto-classification), taxonomies, RDF data | SKOS thesauri, OWL ontologies, tagged documents | Taxonomy change history (UNVERIFIED for 2026) | Yes: editorial workflows | SKOS, OWL, RDF, SHACL, SPARQL | Commercial; GraphDB Free tier | Moving to "Graphwise Platform" (2026) |
| **TopBraid EDG** | Ontologies, glossaries, crosswalks | SHACL-governed vocabularies | **Strongest seen:** per-change history with undo, voting workflows, PROV-O bundled (9.2) | Yes | RDF 1.2, SHACL 1.2, OWL, SKOS | Commercial | GA (9.2, Jun 2026) |
| **Collibra, Atlan** (catalogs with glossaries) | M, lineage, usage; AI-drafted descriptions | Glossary terms, descriptions, owners, certification | Workflow and approval logs per term. Evidence for an AI draft: UNVERIFIED | Yes (human terms); AI drafts need review | Vendor APIs; glossary sync into Snowflake Horizon (Collibra) | Proprietary SaaS | GA |
| **This engine (target)** | M, Q, S (gated by egress decision), D | Concepts, aliases (DE/EN), bindings, merges, axioms; UC proposals | **Every element → Evidence + actor; byte-identical replay** | **Yes, by design (decision ledger separate from drafts)** | Out: UC overlays, Ossie, OWL/SKOS/SHACL + PROV-O (to be decided) | Apache-2.0; standalone or Metis sidecar | Planned v1 |

## Per-product notes

**Databricks Genie Ontology.** Announced 16 Jun 2026 with Genie One and Genie Agents (R1).
- **Two layers:** modelled context (UC semantics: metric views, domains, certification, Pages) and
  inferred snippets, extracted continuously from dashboards, notebooks, SQL queries, Genie agents
  and documentation.
- **Authority score:** from origin, usage and freshness; the ranking (OntoRank) also weighs
  certification and author authority. Snippets respect UC permissions. A user can open a snippet
  to see its definition, origin, author and score.
- **Pages** carry a body, related assets and sources; Genie prefers and cites them.
- **Status:** Public Preview (Pages Beta), enabled through the account team.
- **Not documented (UNVERIFIED):** a snippet export or write API, snippet-level lineage, a public
  Pages API (one community repo calls page creation "manual").
- **Vendor claim:** 84.5% first-attempt accuracy against 52.4% for a general coding agent
  (internal benchmark).
- **For us:** the closest competitor and the main consumer. Pages and certification tags are the
  natural write-back targets.
- **Sources:**
  - [docs](https://docs.databricks.com/aws/en/genie/genie-ontology)
  - [product-dev blog](https://www.databricks.com/blog/how-genie-ontology-powers-product-development-databricks)
  - [operationalizing blog](https://www.databricks.com/blog/operationalizing-genie-ontology-your-data-stack)
  - [Pages blog](https://www.databricks.com/blog/unity-catalog-pages-governed-home-your-business-knowledge-genie-ontology)
  - [Pages docs](https://docs.databricks.com/aws/en/uc-semantics/pages)
  - [itdaily](https://itdaily.com/blogs/cloud/databricks-genie-ontology/)
  - [atlan explainer](https://atlan.com/know/ai-agent/databricks/genie-ontology/)
  - [Norrin](https://norrin.com/insights/what-genie-ontology-means-for-the-databricks-platform)

**Unity Catalog semantics.** UC is the source of truth on Databricks (map principle).
- **Status:**
  - metric views are securable objects (GA per R4);
  - domains are Public Preview;
  - Pages are Beta;
  - certified and deprecated are system tags;
  - the Business Glossary's status conflicts across sources (UNVERIFIED).
- **For us:** comments, tags, metric-view synonyms and Pages are where proposals land.
- **Sources:**
  - [UC semantics](https://docs.databricks.com/aws/en/uc-semantics)
  - [DAIS 2026 UC](https://www.databricks.com/blog/whats-new-unity-catalog-data-ai-summit-2026)
  - [neocarta #206](https://github.com/neo4j-labs/neocarta/issues/206)

**NVIDIA Auto Ontology** (summary of the 3 Nov review). An Apache-2.0 full NL-to-SQL stack.
- **Build step:** an LLM drafts one Term per table, column descriptions and inferred FKs, and mines
  filter and aggregate expressions from 24 h of query history (sqlglot).
- **Not in the code:** documents, dimensions and measures, multilingual vocabulary, build-time
  ambiguity detection.
- **Admitted gaps:** edits lost on rebuild (#259), no element provenance (#260), ungoverned
  sampling (#264), no atomic promotion (#268). Not replayable; no releases.
- **Verdict:** borrow ideas (SQL-log convention mining, zero-overlap join prune, "verbatim-only"
  synonyms, build-time ambiguity scan); do not adopt. It is the clearest example of the gap this
  engine closes.
- **Source:** [repo](https://github.com/NVIDIA/auto-ontology).

**Palantir Foundry Ontology.**
- **Modelling:** dataset-backed object types, plus link, action and interface types; hand-modelled,
  no evidence-backed induction found. Proprietary.
- **Change control:** Foundry Branching proposals, reviewed per task, approval required on
  protected branches (limited beta; legacy ontology proposals are being retired).
- **Audit:** instance-level only: action log and opt-in edit history (excludes pipeline edits).
- **Sources:**
  - [object types](https://palantir.com/docs/foundry/object-link-types/object-types-overview/)
  - [proposals](https://palantir.com/docs/foundry/ontologies/review-ontology-proposals/)
  - [legacy](https://palantir.com/docs/foundry/ontologies/ontology-branches-legacy/)
  - [action log](https://palantir.com/docs/foundry/action-types/action-log/)
  - [edit history](https://palantir.com/docs/foundry/workshop/widgets-edits-history/)

**Microsoft Fabric IQ ontology.** Preview.
- **Model:** entity types, properties, relationships and rules, bound to OneLake-backed sources.
- **Generation** from a semantic model creates entity and relationship types; time-series and
  relationship bindings are finished by hand.
- **Standards:** imports TTL, RDF/RDFS and OWL into an empty item; exports TTL and RDF; "not
  lossless".
- An "Ontology Agent" that drafts from plain English is reported only by a blog (UNVERIFIED).
- **Sources:**
  - [overview](https://learn.microsoft.com/en-us/fabric/iq/ontology/overview)
  - [generate](https://learn.microsoft.com/ar-sa/fabric/iq/ontology/concepts-generate)
  - [import/export](https://learn.microsoft.com/en-us/fabric/iq/ontology/how-to-import-export)
  - [bind data](https://learn.microsoft.com/en-us/fabric/iq/ontology/how-to-bind-data)

**Snowflake semantic views and Cortex Analyst.**
- **Autopilot** uses the creating role's query history to suggest relationships and verified
  queries; supplied SQL is validated and added as verified queries (`AI_VERIFIED_QUERIES` in DDL
  since Apr 2026).
- **Ossie YAML** is read and written natively (`SYSTEM$READ_OSSIE_YAML_FROM_SEMANTIC_VIEW`), with a
  SNOWFLAKE vendor extension.
- **Missing:** documents, and per-element actor or evidence beyond verified queries.
- **Sources:**
  - [Autopilot](https://docs.snowflake.com/en/user-guide/views-semantic/autopilot)
  - [verified queries RN](https://docs.snowflake.com/en/release-notes/2026/other/2026-04-05-semantic-views-verified-queries)
  - [Ossie read](https://docs.snowflake.com/en/sql-reference/functions/system_read_ossie_yaml_from_semantic_view)

**dbt Semantic Layer and MetricFlow.**
- MetricFlow is Apache-2.0 from 0.209 (latest 0.213, Sep 2026); hosted APIs stay proprietary.
  dbt Core 1.12 accepts Ossie documents. Hand-authored, so traceability is git blame. For us: an
  export target, not a competitor.
- **Sources:**
  - [MetricFlow](https://docs.getdbt.com/docs/build/about-metricflow)
  - [Atlan licence note](https://atlan.com/know/ai-agent/semantic-layer/metricflow-apache-2-license-dbt-semantic-layer/)
  - [OSI is now Ossie](https://www.getdbt.com/blog/osi-is-now-apache-ossie)

**Timbr.**
- An ontology-based SQL semantic layer mapping concepts, relationships and measures onto Delta
  tables without moving data; exposes model metadata in UC and pushes queries down. Authored in a
  visual UI or SQL DDL. OWL import, licence and provenance features not found (UNVERIFIED).
- **Sources:**
  - [Databricks page](https://timbr.ai/data-platforms/databricks-lakehouse/)
  - [semantic layer](https://timbr.ai/solutions/semantic-layer/)

**Graph and ontology vendors.**
- **Stardog:** v12 adds ontology-driven extraction from text, linked to structured data; 11.2 added
  Voicebox ontology suggestions from competency questions; Voicebox service 1.1 GA 9 Sep 2026;
  Databricks virtual-graph source. [v12](https://info.stardog.com/whats-new-in-stardog-12),
  [11.2](https://community.stardog.com/t/whats-new-in-stardog-11-2/5512),
  [Voicebox RN](https://docs.stardog.com/release-notes/stardog-voicebox).
- **Neo4j:** LLM Graph Builder (Labs, v0.8.6 Jun 2026, licence sources conflict); LLM schema
  extraction "experimental".
  [KG builder guide](https://neo4j.com/docs/neo4j-genai-python/current/user_guide_kg_builder.html).
- **Graphwise:** PoolParty and GraphDB merging into one platform (Feb 2026); GraphDB 11 added LLM,
  GraphRAG and MCP.
  [transition](https://help.graphwise.ai/en/graphwise-documentation/graphwise-sub-components-at-a-glance/poolparty-semantic-suite-to-graphwise-platform-transition-guide.html).
- **TopBraid EDG 9.2** (Jun 2026): per-change history with undo, voting workflows per governance
  role, RDF 1.2 reification, PROV-O bundled. The best match for a "decision ledger", but for
  hand-curated vocabularies, not induction. [9.2 notes](https://topquadrant.com/release-note/9-2).

**Catalogs with glossaries (Collibra, Atlan; also Purview).**
- AI-drafted descriptions and terms, to be reviewed before publishing, with workflow approval per
  term. Collibra syncs glossaries into Snowflake Horizon (Jun 2026); its Databricks integration is
  access policy, not semantics. Missing: evidence-backed induction and replay.
- **Sources:**
  - [Atlan AI terms](https://atlan.com/demos/use-atlan-ai-document-terms.md)
  - [Purview RAI FAQ](https://learn.microsoft.com/en-us/purview/unified-catalog-responsible-ai-faq)
  - [Collibra–Snowflake](https://www.pulse.bot/ai/news/snowflake-and-collibra-expand-integration-to-bring-governed-business-context-and-semantics-across-th-566caab1-f0f3-4910-990f-18f65fac43cc/)

**Apache Ossie.**
- Incubating since Jul 2026, spec unchanged by the rename. Converters to MetricFlow, Snowflake and
  Polaris, plus NVIDIA's; no Databricks metric-view converter found. The Ontology working group has
  no published output yet.
- **Sources:** [Snowflake](https://www.snowflake.com/en/blog/apache-ossie-open-semantic-interchange-incubator/).

## Positioning

The engine is not a semantic layer, a catalog or a question-answering agent. Genie, Cortex, Fabric
data agents and Metis answer questions. Unity Catalog, Snowflake semantic views and MetricFlow hold
the governed definitions. This engine sits in front of them and **drafts and resolves the meaning**
those layers need, from warehouse metadata, usage and documents, in German and English.

Every concept, alias, binding, merge and axiom it produces carries the evidence it rests on (a UC
object id, a query-log statement, a document span) and the actor that made it (engine version,
model and prompt, or a named human). Every run replays byte-identically. A human decision is a
first-class record that outlives every rebuild, so curation is never overwritten by the next
draft. An unknown link stays unknown, never silently merged.

On Databricks it overlays UC and never shadows it: its output is proposals keyed on UC object ids
(comments, tags, metric-view synonyms, Pages drafts), and Genie Ontology then ranks it as
certified context. Off Databricks it exports to open standards. Genie Ontology wins on reach inside
one platform. This engine wins on **auditability, durability of human judgement and portability**:
the three things every auto-drafting competitor either lacks (Auto Ontology #259/#260) or keeps
inside a closed store (Genie's inferred snippets).

## Three claims the demo must prove

1. **Click-through provenance, and replay.** Pick any element, say the German alias
   "Großbritannien" → `country = 'GB'`, or a merge of two customer concepts. The demo shows its
   Evidence (UC object id and column, query-log statement, document character span) and its actor
   (engine version, model id and prompt hash, or a reviewer's name). Re-running from the recorded
   inputs gives a byte-identical output digest.
2. **Human decisions survive a rebuild.** A reviewer rejects a merge, renames an alias and
   certifies a concept. The demo then changes the source (new columns, new logs, an edited
   document), wipes the drafts, and runs a full rebuild and a delta run. All three decisions hold.
   Where new evidence contradicts a decision, the engine raises a conflict for review; it never
   silently overwrites or merges. This is the exact failure of Auto Ontology #259.
3. **Additive and UC-native, with open exports.** Against a real UC catalog with metric views, the
   engine emits only proposals keyed on UC object ids, with no parallel definition of anything UC
   defines (shown by a diff). Once applied they are visible in UC and used by Genie or Metis. The
   same run exports Ossie YAML and OWL/SKOS with provenance, which re-imports without losing the
   provenance links.

## Open questions

- **Genie Ontology APIs.** Is there an export or system-table view of inferred snippets, and a write
  API for Pages? This decides whether write-back goes to Pages or only to comments, tags and
  metric-view synonyms. Needs `docs.databricks.com` reachable, or a workspace test.
- **Do edits to inferred Genie snippets persist across refresh?** If they do, the
  "curation-survives-rebuild" claim is weaker against Genie than against Auto Ontology.
- **Provenance vocabulary.** PROV-O (as TopBraid ships it) or our own Evidence/actor contract with a
  PROV-O mapping on export? And which RDF 1.2 reification, if any?
- **Ossie Ontology working group:** track it and any Databricks metric-view converter; do not get
  ahead of a spec that may land in 2027.
- **UC Business Glossary:** GA status and API unclear. With an API it may beat Pages as the alias
  target.
- **Later targets:** Snowflake (native Ossie) and Fabric (TTL/OWL import) look cheap. Does Fabric's
  "lossy subset" keep annotation properties, i.e. provenance?
- **Unverified:** Timbr's licence and deployment, the Neo4j Graph Builder licence, Stardog's
  per-element provenance. Each needs a primary read before it is quoted externally.
