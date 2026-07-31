# 12 — Task: M1 — EU segmentation + hyperedge builder

Type: task
Status: open
Blocked by: 03, 04

**CHECK IN with Dennis before starting — first long-running implementation on this map.**

## Question

Implement spec 1.2–1.5 per the decisions of 03/04: sentence-split, window extension driven by relation argument spans, minimal-cover EUs merged/split to the 40–120-token target, weak EUs, hyperedge records with (possibly uncalibrated, per 09) gamma, arity cap with relation-group splitting, dedup/merge with support counter. Tests per spec M1: every relation's arguments inside its EU span; no EU > 180 tokens; merge idempotency. Repo DoD bar applies.
