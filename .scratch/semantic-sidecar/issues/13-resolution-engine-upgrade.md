# Grilling: the upgrade of the resolution engine

Type: grilling
Mode: HITL
Status: open
Blocked by: 04, 10

## Question

Decide the resolution engine for both modalities:

- a learned or Fellegi-Sunter matcher versus the string cascade;
- calibration and precision-targeted thresholds;
- an LLM in the uncertain band, and its egress rules;
- persisted must-link and cannot-link constraints from human decisions, replayed every run
  and under deltas (ADR-0018/0029/0061);
- an actor on every decision (`MergeDecision` has none today);
- splitting homonyms;
- scale: in-memory versus Spark;
- what happens to `disambiguation.cascade`, `.embedding` and `.glinker`, and to
  `type_consolidation`.
