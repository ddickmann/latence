# Research: ontology learning methods and the standards to emit

Type: research
Mode: AFK
Status: resolved
Blocked by: 

## Question

How do we get from canonical entities and types to an ontology, meaning the schema level
(classes, hierarchy, properties, constraints) and not only instances? And which standards
should it speak?

- **Methods:**
  - LLM-based ontology learning and taxonomy induction;
  - axiom and constraint induction;
  - SHACL shape induction from data (and ShEx);
  - formal concept analysis;
  - aligning induced types to an existing vocabulary.
- **Standards:**
  - OWL 2 profiles (EL, QL, RL), and which one fits;
  - SKOS for vocabularies, with multilingual labels;
  - SHACL (including 1.2 work);
  - PROV-O for Evidence;
  - RDF 1.2 / triple terms versus reification for edge annotations;
  - stable IRI policy.
- **Interchange:**
  - Apache Ossie: the 0.1 versus 0.2.x mismatch, the Ontology working group and what it is
    producing, and the Databricks converter status;
  - how any of this maps to Unity Catalog objects and metric views.
- The ROADMAP's P4 item ("concepts, taxonomy and axioms … SHACL and OWL") has no design doc.
  Give enough to write one.

Resolves with a report at `research/05-ontology-learning-and-standards.md`.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/05-ontology-learning-and-standards.md](../research/05-ontology-learning-and-standards.md). Only GitHub could be read directly; w3.org, arXiv, TIB and Databricks were blocked, so many claims come from search excerpts and are marked UNVERIFIED.

- **No method is good enough to auto-accept.** The best blind taxonomy-discovery F1 in the 2025 LLM ontology-learning challenge is about 0.5–0.6, and non-taxonomic relations are weak at every model size.
- **Permissive building blocks:**
  - QSE (MIT) and sheXer (Apache-2.0) for SHACL shape mining;
  - formal concept analysis;
  - OntoAligner and DeepOnto (Apache-2.0) for alignment.

  DL-Learner (GPL-3) is excluded. AMIE (CC BY 4.0) is reimplemented, not vendored.
- **Signal order:** declared warehouse constraints (types, NOT NULL, keys) and row and usage evidence outrank model output. An LLM is an optional signal, behind the egress gate, and never a proposal's only one.
- **Standards:**
  - **OWL 2 QL**, so the ontology can be queried over Databricks without copying data (Ontop, R2RML). Keys, cardinality and value sets go in SHACL.
  - **SHACL 1.1**; 1.2 is still a Working Draft (August 2026).
  - **RDF 1.2** is still a Candidate Recommendation. Annotate edges as named graphs now, laid out to map one-to-one to RDF 1.2 reifiers later.
  - **SKOS** labels in DE and EN, with raw induced labels as hidden labels.
  - **PROV-O** for Evidence and actors.
  - **SSSOM** for mappings to outside vocabularies (schema.org, FIBO).
- **IRI policy:**
  - an HTTP base the adopter configures;
  - opaque ids minted once, never derived from labels;
  - proposals identified by a content hash, so human decisions survive rebuilds;
  - nothing minted for UC objects, which are referenced by their own ids. Whether those ids are stable is UNVERIFIED and goes to *the Databricks semantic surface*.
- **Ossie:**
  - The only tag is `osi-0.1.1-rc1`, with no ontology.
  - The ontology spec exists only on `main`, as an unreleased 0.2.0.dev0 draft. It is fact-based, not OWL, and its mapping documents have no parser yet.
  - A Databricks metric-view converter exists in Java (metric-view YAML v1.1).
  - Ossie is an experimental writer pinned to a commit. The Metis seam stays on 0.1.1.
- **UC mapping:** classes, properties, labels and constraints become overlays or upstream proposals. Measures stay owned by UC. Genie Ontology reportedly has no export and is not a formal ontology (UNVERIFIED).
- **P4 sketch:**
  - a new `OntologyProposal` contract (signals, Evidence, actor, status) and ten stages, from concepts to writers;
  - a subclass link needs two independent signals or a human;
  - every candidate shape is validated against the data before acceptance;
  - model-only proposals are never auto-accepted;
  - the private `lg:` vocabulary is replaced by standard RDF and SKOS terms.
- **For other tickets:**
  - whether v1 needs a hierarchy at all (*the destination and the domain model*, *ontology induction and the output standards*);
  - the ConceptInducer / Taxonomy / AxiomInducer code that ADR-0005 calls preserved is **not in the tree**. It must be found before the P4 design doc is written; this is a task for *ontology induction and the output standards*.
  - Nine further open questions are listed in the report.
