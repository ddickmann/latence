# Grilling: ontology induction and the output standards

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 05, 10, 21

## Question

Decide:

- how the schema level is induced: classes, hierarchy, properties with domain and range,
  cardinality, constraints;
- that every axiom is a proposal with Evidence and an actor;
- whether shapes are validated against data before acceptance;
- which standards are emitted: OWL profile, SKOS, SHACL, PROV-O, RDF 1.2, Ossie, metric view
  YAML;
- the IRI policy;
- what replaces the current private `lg:` TTL vocabulary.

## Answer

Decided by the owner on 2026-11-10, one question at a time. Recorded as
[ADR-0073](../../../docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md).

1. **Induction: several explainable signals vote, and a model is one voice.** In order of
   weight: what the warehouse declares (types, keys, NOT NULL, value lists); statistics on the
   data (a class whose members are always a subset of another's is proposed as its subclass);
   names (the lexical head of a term); and the hosted model, which proposes and names
   candidates from metadata. Every proposal lists its signals, and an expert approves it.
   Every axiom is therefore a proposal with Evidence and an Actor (ADR-0069).
2. **Validation: every proposed constraint is run against the data first,** inside the
   workspace. The expert sees the rule with its support and can open the violations. Weakly
   supported rules are dropped before they reach the queue. Accepting a rule never changes
   data; its violations become review items.
3. **Standards in v1:** OWL 2 held to the QL profile; SHACL 1.1; SKOS with language tags
   (English first); PROV-O; and metric-view YAML for measure proposals. Ossie and RDF 1.2 wait
   until they are final. SSSOM is not in v1.
4. **Naming:** opaque, stable IRIs under a per-customer prefix, with a neutral default for
   standalone use. Readable names live in labels.
5. **The private `lg:` vocabulary is replaced** by the standard vocabularies. The few terms no
   standard covers stay in one small, documented namespace of the engine's own.

Cardinality and domain/range are constraints and follow answers 1 and 2: declared ones are
taken from the warehouse, mined ones are proposed with their support.

## From the map (moved 12 Nov)

[ADR-0073 — The schema level is induced by voting signals, validated on data, and exported in W3C standards](../../docs/adr/0073-the-schema-level-is-induced-by-voting-signals-validated-on-data-and-exported-in-w3c-standards.md) and [Ontology induction and the output standards](issues/14-ontology-induction-and-standards.md) — declared constraints, data statistics and names vote, with the hosted model as one voice. Every proposed constraint is run against the data before review, and weak ones are dropped. v1 exports OWL 2 QL, SHACL 1.1, SKOS, PROV-O and metric-view YAML. IRIs are opaque under a per-customer prefix; the private `lg:` vocabulary goes.
