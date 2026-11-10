# The schema level is induced by voting signals, validated on data, and exported in W3C standards

**Status:** Accepted (2026-11-10)

This record fixes how the re-scoped engine (ADR-0067) arrives at the schema level of its
semantic layer (classes, their hierarchy, properties and constraints), how a proposed
constraint is checked before a human sees it, and in which formats the result leaves the
engine. The owner decided each point on 2026-11-10 (map ticket *Ontology induction and the
output standards*). Nothing here is built yet.

## Context

ADR-0069 made every schema-level element a proposal that a human approves. That leaves open
where proposals come from. Research for the re-scope found that no single method is reliable
enough to trust: blind taxonomy discovery tops out around F1 0.5 to 0.6, at every model scale.
It also found that constraints are the strongest thing that can be induced, because a warehouse
already declares many of them, and that pruning mined constraints on their support removes
most spurious ones.

Today the knowledge-graph export writes Turtle in a private `lg:` vocabulary under a
non-dereferenceable base. An ontology an enterprise can audit and load into other tools has to
use public vocabularies and identifiers that survive a rename.

## Decision

1. **Several explainable signals vote; a model is one voice among them.** In order of weight:
   - what the warehouse declares: types, keys, NOT NULL, value lists;
   - statistics on the data: a class whose members are always a subset of another's is
     proposed as its subclass;
   - names: the lexical head of a term;
   - a hosted model that proposes and names candidates from metadata.
   Every proposal lists the signals behind it, and a human approves it.
2. **A proposed constraint is run against the data before a human sees it.** The check runs
   inside the workspace (ADR-0071). The reviewer sees the rule with its support, for example
   how many records satisfy it and how many break it, and can open the violations. A rule with
   too little support is dropped before it reaches the queue. Accepting a rule never changes
   data: its violations become review items.
3. **Standards exported in v1:**
   - **OWL 2, held to the QL profile,** for classes, hierarchy and properties;
   - **SHACL 1.1** for constraints;
   - **SKOS** for labels and synonyms, with language tags (English first);
   - **PROV-O** for Evidence and for who decided what;
   - **metric-view YAML** for the measure proposals that go to Unity Catalog (ADR-0070).
4. **Identifiers are opaque and stable.** Every Concept and Entity gets an IRI that carries no
   name, under a prefix set per customer, with a neutral default for standalone use. Readable
   names live in labels.
5. **The private vocabulary goes.** Standard vocabularies replace `lg:`. The few terms no
   standard covers, such as a calibrated confidence, stay in one small, documented namespace
   of the engine's own.

## Alternatives considered

- **Model-first induction,** with the other signals only checking the draft. Faster to a first
  version. Rejected: each proposal would rest on weaker evidence, and the measured accuracy of
  model-drafted taxonomies does not support leading with them.
- **No model at all.** Fully reproducible, and nothing leaves the customer's environment.
  Rejected as the default: it misses classes that are only implied and names things poorly.
  It remains what a deployment gets when it makes no egress decision.
- **Validating constraints but dropping none.** Nothing is hidden. Rejected: the queue fills
  with rules that mostly do not hold, and reviewer attention is the scarce resource.
- **No validation in v1.** Simpler, and needs no data access. Rejected: a reviewer would
  approve rules without knowing whether the data obeys them.
- **A smaller export, SKOS and SHACL only.** Rejected: that is a glossary with rules, not yet
  an ontology, and it carries no provenance.
- **Adding an alignment format or an open interchange file in v1.** Each is useful where a
  customer aligns to an external vocabulary or feeds another tool. Deferred: one interchange
  format's ontology part is an unreleased draft, and neither is needed to prove the engine.
- **RDF 1.2 for statement-level annotations.** Deferred until it is a final recommendation;
  named graphs carry the annotations until then.
- **Readable identifiers.** Easier to read in files and queries. Rejected: a rename either
  breaks every reference or leaves an identifier that contradicts its label.
- **Identifiers derived from Unity Catalog object ids.** A tight link to the catalog.
  Rejected: Concepts that come from documents have no such id, and a re-created table gets a
  new one. The link to the catalog is the Binding (ADR-0069).

## Consequences

- The knowledge-graph export changes its vocabulary and its identifiers. That is a contract
  change with a `SCHEMA_VERSION` bump, and existing consumers of the Turtle file are affected.
- Each exported format needs a validator in the test suite, so that a file the engine writes
  is known to conform before it is called conformant.
- Constraint validation needs read access to the data and therefore the same in-workspace
  execution, filter and mask labelling as matching (ADR-0071).
- The engine's own namespace is small and has to be documented and versioned.
