# Research: relation resolution and canonicalization at scale

Type: research
Mode: AFK
Status: open
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
