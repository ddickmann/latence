# Research: ontology learning methods and the standards to emit

Type: research
Mode: AFK
Status: open
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
