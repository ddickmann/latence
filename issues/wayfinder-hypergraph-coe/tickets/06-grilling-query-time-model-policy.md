# 06 — Grilling: query-time model + entity_kb policy

Type: grilling
Status: open
Blocked by: 01

## Question

Online entry (spec 2.1) needs GLiNER2 NER + glinker linking on the query string, and GLiNER2's zero-shot relation-type extraction for w_relq (spec 3.2) — today only rerank loads a model at query time. (GLiREL is NOT involved: the repo's fused GLiNER2 extraction covers all its spec roles — see ticket 02 addendum + the spec's adaptation note — so one model family may serve NER + w_relq in a single query-time load; grill whether that consolidation holds.) Decide: which models load in `latence-retrieval` (a query-signal-gen seam per ADR-0049?); how the entity_kb/alias table reaches the stateless query path; latency budget; the degraded path when models are absent (V_vec-only entry — how much does the frontier claim survive?). Outcome: seam design + ADR note.
