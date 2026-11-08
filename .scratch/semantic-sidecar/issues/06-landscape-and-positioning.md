# Research: the landscape: who builds semantic and ontology layers, and what each makes traceable

Type: research
Mode: AFK
Status: resolved
Blocked by: 

## Question

Against whom is this engine positioned, and on which axis does it win?

Cover:

- Databricks Genie Ontology;
- NVIDIA Auto Ontology (summarise `metis-companion/research-holding/2026-11-03/nvidia-auto-ontology-review.md`;
  don't redo it);
- Palantir Foundry Ontology;
- Microsoft Fabric's ontology / IQ offering;
- Snowflake semantic views;
- the dbt Semantic Layer / MetricFlow;
- Timbr;
- the graph and ontology vendors (Stardog, Neo4j, Graphwise and similar);
- data catalogs with glossaries.

For each:

- what it ingests (metadata, data samples, query logs, documents);
- what it outputs;
- whether each element is traceable to its source and actor;
- whether curation survives a rebuild;
- open standards in and out;
- licence and deployment.

End with a one-paragraph positioning statement and the three claims this engine must be able
to prove in a demo.

Resolves with a report at `research/06-landscape-and-positioning.md`.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/06-landscape-and-positioning.md](../research/06-landscape-and-positioning.md). The Databricks and Fabric sections rest on search-result summaries, because their docs sites were unreachable from the session.

- **The gap is per-element traceability.** Every product traces the *answer*, not the semantic *element*. Genie Ontology shows each snippet's origin, author and authority score. Palantir logs actions. Snowflake keeps the verified queries a view came from. None links each alias, merge or axiom to its evidence and its actor, and none can replay a run.
- **Curation is fragile in the auto-drafting tools.** Human edits share a store with the drafts. The open-source auto-ontology project loses certified edits on rebuild (#259) and has no element-level provenance (#260).
- **Genie Ontology is both the nearest competitor and a consumer.** It is in Public Preview, with Pages in Beta. It keeps human Pages apart from inferred snippets. No export or write API for snippets was found (UNVERIFIED).
- **Other platforms:**
  - Fabric IQ builds an ontology once from a semantic model, then it is edited by hand (Preview). It imports TTL/RDF/OWL and exports TTL/RDF.
  - Snowflake reads and writes Apache Ossie YAML natively.
  - dbt MetricFlow and Timbr are hand-modelled.
  - TopBraid EDG has the best change history (PROV-O, workflows), but only for hand-curated vocabularies.
  - Collibra and Atlan draft terms with AI and approve them per term, with no evidence attached.
- **Positioning:** not a semantic layer or a chat tool. The engine drafts and resolves meaning from warehouse metadata, usage and documents, and keeps a ledger of evidence and decisions. It writes proposals into UC keyed on UC ids and exports to open formats for Genie and Metis.
- **Three demo claims:**
  1. Pick any element and show its evidence and actor, then replay the run byte for byte.
  2. A reviewer's decisions survive a full rebuild and a delta run, and contradicting new evidence is raised for review, never silently applied.
  3. Against a real UC the output only adds proposals and never redefines anything UC defines, and it exports to Ossie and OWL/SKOS with provenance intact.
- **Open, for other tickets:**
  - Genie snippet and Pages APIs, and whether edits survive a refresh (*the Databricks semantic surface*);
  - PROV-O versus our own provenance contract, and the Ossie ontology working group (*ontology learning and standards*);
  - whether the UC Business Glossary is GA;
  - the Timbr, Neo4j and Stardog provenance claims (UNVERIFIED).
