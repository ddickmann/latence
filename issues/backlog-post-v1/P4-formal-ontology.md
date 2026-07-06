# P4 — Formal ontology plugin (preserved)

**Parent**: post-v1 · **Plugin · Hardening bar**

## What to build
An optional plugin that induces a formal ontology over the corpus KG and materializes it as SHACL shapes + OWL TBox/ABox (TTL). Ships the ontology-induction IP the architecture pass wrongly dismissed — the inventory rates it 4–4.5/5 with weeks of reinvention cost.

## Ports (algorithmic assets — PORT-AS-IS)
- **ConceptInducer** (`ontology/concept_inducer.py`, 620 LOC): agglomerative embedding clustering, Union-Find concept merge (>0.98 cos) with parent-child fixup, relation signatures.
- **Taxonomy** (`ontology/taxonomy.py`, 400 LOC): 3-signal hierarchy (distributional inclusion + embedding hypernym + lexical; Hearst'92/Roller'18).
- **AxiomInducer** (`ontology/axiom_inducer.py`, 600 LOC): OWL2 disjointness/cardinality/domain-range/property-chains/restrictions.
- SHACL/TBox exporters.

## Acceptance criteria
- [ ] Induces concepts, a concept hierarchy, and OWL axioms from the canonical KG.
- [ ] Exports valid SHACL + OWL/TTL that load in a standard reasoner/validator.
- [ ] Ontology-quality metrics (axiom precision, hierarchy coherence, schema coverage) reported.
- [ ] Runs as an opt-in plugin; not on the v1 critical path.
- [ ] Hardening bar met.

## Blocked by
v1 spine stable through S9. Independent of P1/P2/P3.
