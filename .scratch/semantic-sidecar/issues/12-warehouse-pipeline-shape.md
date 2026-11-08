# Grilling: the shape of the warehouse pipeline

Type: grilling
Mode: HITL
Status: open
Blocked by: 01, 03, 10

## Question

How does warehouse metadata flow through the engine?

- Which harvesters exist in v1: UC metadata, metric views, lineage, query history, Genie
  agents, glossaries, BI models.
- Do they become Source and Stage kinds in the existing Pipeline (ADR-0003), or a parallel
  pipeline?
- Which matching signals are used, and in what order (names, comments, values and sketches,
  lineage, co-usage)?
- What may be read from the data (egress and sensitivity), and how are row filters and masks
  respected?
- How do warehouse assertions and text mentions meet in one resolution phase?
