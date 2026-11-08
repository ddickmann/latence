# Grilling: the destination and the domain model

Type: grilling
Mode: HITL
Status: open
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
