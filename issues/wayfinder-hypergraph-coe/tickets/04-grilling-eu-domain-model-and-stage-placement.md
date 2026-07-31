# 04 — Grilling: Evidence Unit domain model + stage placement

Type: grilling
Status: open
Blocked by: 01

## Question

Domain-model the Evidence Unit / hyperedge against CONTEXT.md — especially its relation to the existing **Evidence** term (edge→source justification) and to Chunk (storage granularity vs reasoning granularity, spec 1.2). Where does EU segmentation + hyperedge construction live: a new corpus-level Stage, or an extension of Graph Assembly? Settle: the EU record contract (edge_id, entity_ids, relations, tau_text, provenance, gamma, token_cost, support, has_relations), segmentation rules ownership (40–120-token targets, merge/split), arity cap m_max, dedup/merge semantics (same entity set + same relation signature only), weak EUs. Outcome: CONTEXT.md terms + ADR + record schema.
