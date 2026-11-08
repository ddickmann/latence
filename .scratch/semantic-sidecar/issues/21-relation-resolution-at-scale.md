# Research: relation resolution and canonicalization at scale

Type: research
Mode: AFK
Status: resolved
Blocked by:

## Question

How should relations be resolved at corpus and Unity Catalog scale, so that the knowledge graph
has one canonical predicate per meaning and one aggregated edge per fact, each with traceable
evidence?

Cover:

- canonicalization of open relation phrases (CESI, SIST, CMVC, JOCL, EDC and newer work);
- schema-guided relation typing, with domain and range constraints used as vetoes;
- unmapped relations handled as property proposals;
- edge aggregation and truth discovery, including contradiction detection for functional
  properties;
- collective resolution of entities and relations together;
- canonical relations from UC foreign keys, joins, lineage and query history, and how to reconcile
  them with relations extracted from text;
- controlling relation fabrication;
- evaluation metrics and benchmarks.

The research 04 report and the design note leave this gap open.

Resolves with a report at `research/21-relation-resolution-at-scale.md`.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/21-relation-resolution-at-scale.md](../research/21-relation-resolution-at-scale.md). Primary sites (arXiv, ACL Anthology, CEUR, Databricks docs) were blocked, so 17 claims are marked UNVERIFIED. Licences read directly from GitHub are confirmed.

- **Today the engine folds labels; it does not resolve relations.**
  - The graph builder emits one edge per relation mention, so ten mentions of one fact become ten parallel edges, each with the extractor's raw confidence.
  - Edge Evidence lists only the two endpoint mentions, so the exported evidence count is always 2.
  - The fuzzy label match is unsafe: `not_located_in` folds into `located_in`, while `employed_by` and `works_for` never meet.
- **Unit of work: one distinct (raw label, head class, tail class) triple, not one mention.** That means thousands of items, not millions, so a model in the review band is cheap.
- **Canonicalising open relation phrases.** CESI (Apache-2.0), SIST, JOCL and CMVC cluster noun phrases well, but their quality on relation phrases is barely measured. The best fit is **EDC** (MIT): extract, write a definition, retrieve schema candidates, verify. KGGen, AutoSchemaKG (both MIT) and TRACE-KG follow the same cheap-candidates-then-model-verifies shape.
- **Typing against the ontology.**
  - Each property carries a DE/EN definition, domain, range and characteristics.
  - Candidates come from definition retrieval; a calibrated scorer decides; an LLM is used only in the review band.
  - Domain and range are hard vetoes.
  - An unmapped relation becomes a property proposal, never dropped and never a silent new label.
- **One edge per fact.**
  - Group mentions by (head, property, tail, qualifier key) and fuse per-document evidence into a calibrated probability.
  - Knowledge Vault and knowledge-fusion rules apply: count distinct documents, declare per property whether it is functional, and don't count hierarchical values as conflicts.
- **Contradictions and time.**
  - Only functional properties (SHACL `maxCount 1`) can conflict.
  - Conflicts are queued for review, ranked by probability, never auto-resolved.
  - Edges record observation time and validity time separately.
- **Joint entity–relation resolution** (Bhattacharya & Getoor, PARIS, SiGMa) is bounded:
  - at most two deterministic rounds;
  - only accepted edges count as evidence;
  - UC anchors are fixed seeds;
  - relations alone never trigger a merge.
- **Warehouse side.**
  - Foreign keys, metric-view joins, verified joins and query-history joins become schema-level properties with cardinality, emitted as overlays and R2RML.
  - Text predicates are aligned to them through the same definition matcher, as alignment proposals. UC is never silently overridden.
- **Fabrication and evaluation.**
  - Only grounded, entailment-checked mentions count as support.
  - Edge precision is estimated by sampling with an error bound (Gao et al., 2019).
  - No benchmark is German. REBEL and the WebNLG data inside Text2KGBench are non-commercial or share-alike.
- **v1 design:** stages R0–R8 with new contracts — a predicate binding, an aggregated edge and a conflict set. This needs a `SCHEMA_VERSION` bump.
- **For other tickets:**
  - stages R0–R8 extend the candidate design (*the upgrade of the resolution engine*);
  - property definitions, functional flags and validity time belong to the ontology contract (*ontology induction and the output standards*);
  - the conflict queue and edge-level QA belong in the workbench (*the expert resolution workbench*);
  - a German relation gold set (*evaluation and the acceptance bars*).
