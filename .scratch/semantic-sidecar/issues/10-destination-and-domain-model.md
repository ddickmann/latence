# Grilling: the destination and the domain model

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 05, 06

## Question

Pin the destination and the language before any design.

- **Terms:** what are a *Concept*, a *Binding* (concept ↔ UC table/column/measure), an
  *Assertion* (a harvested statement from any source), a *Locator* (a text span, UC object,
  query, lineage edge or value sketch), and an *Actor* (engine, model or human)?
- **Levels:** where is the line between the ontology's schema level (classes, hierarchy,
  properties, constraints) and its instance level (canonical entities)? Does v1 need both?
- **Done:** refine the map's draft criteria and make each one measurable.
- **What is this not:** what stays with Metis or Unity Catalog.

The result goes into `CONTEXT.md` (the glossary) and an ADR if the trade-off is real.

## Answer

Decided by the owner on 2026-11-09, one question at a time. Recorded as
[ADR-0069](../../../docs/adr/0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md).

1. **Levels.** v1 covers both levels and treats them differently. Instance-level merges and links
   may be accepted by the engine inside the calibrated accept band. Schema-level elements
   (classes, hierarchy, properties, constraints) are only ever proposals that an expert approves.
2. **Concept and Entity.** A *Concept* is a schema-level element: a class, a property or
   relation, or a measure or dimension. A measure is a Concept bound to a UC metric-view measure,
   not a separate kind. An *Entity* is one canonical real-world thing at the instance level, made
   by merging records and mentions.
3. **Ledger.** Two append-only record kinds. An *Assertion* is one statement harvested from one
   source; it always carries a *Locator* and the *Actor* that produced it. A *Decision* is an
   accept, reject, merge, split or edit made on top of assertions, with its Actor and the evidence
   it used. Nothing in the ontology exists without at least one Assertion, and the current
   ontology is a replay of the ledger. A correction is a new Decision, never an edit.
4. **Authority.** An *Actor* is the engine, a model (name and version) or a human.
   - The engine may accept instance-level merges and links, only inside the accept band.
   - A model's verdict accepts nothing by itself: it is a feature of the score or a suggestion.
   - Schema-level changes, splits of an existing Entity and any write-back to Unity Catalog need
     a human.
   - A human Decision outranks the engine and is never overwritten by a later run.
5. **Bindings.** A *Binding* ties the semantic layer to a Unity Catalog object. It is keyed on UC
   object ids and carries a role (*defines*, *identifies*, *measures*). v1 has both kinds:
   - a *Concept binding* ties a Concept to a table, column or metric-view measure;
   - an *Entity binding* ties an Entity to the rows that represent it (object id plus
     primary-key value). Together these form the cross-system crosswalk, a delivered output.
6. **What this is not.** The engine never defines a table or a metric itself. It binds to what
   UC defines, adds what UC lacks (synonyms, descriptions, the class hierarchy, relations between
   Concepts, the Entity crosswalk) and sends anything that would change UC upstream as a proposal
   for a human to apply. **Drafting new metric-view measures as proposals is in scope for v1**
   (owner; the recommendation was to leave it out). This overlaps with Metis, so *Boundary with
   Metis* has to draw that line.
7. **Done for v1**, one measurable check per criterion:
   1. *Trace.* 100% of Concepts, Entities, Bindings, merges and axioms link to at least one
      Assertion and an Actor. The same inputs run twice give byte-identical output.
   2. *Curation survives.* A test set of human Decisions is 100% intact after a full rebuild and
      after a delta run.
   3. *UC-native.* Zero elements redefine something UC defines. Exports validate against their
      standards (OWL 2 QL, SHACL, SKOS).
   4. *One seam.* It runs standalone end to end, and Metis reads it through exactly one declared
      interface.
   5. *Text path.* Strictly better than the recorded baseline on extraction F1, linking F1 and
      documents per second. No fixed margin.
   6. *No retrieval code remains.*
   7. *Resolution quality.* Auto-accept precision is at least 0.95 with its confidence stated,
      on the warehouse side and on the text side. Review rate and expert minutes are reported
      and gate nothing.

The glossary entries go into `CONTEXT.md` with the first slice that implements them, so the
glossary never describes code that does not exist.
