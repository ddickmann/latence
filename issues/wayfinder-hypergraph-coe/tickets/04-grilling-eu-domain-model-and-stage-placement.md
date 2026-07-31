# 04 — Grilling: Evidence Unit domain model + stage placement

Type: grilling
Status: resolved
Blocked by: 01

## Question

Domain-model the Evidence Unit / hyperedge against CONTEXT.md — especially its relation to the existing **Evidence** term (edge→source justification) and to Chunk (storage granularity vs reasoning granularity, spec 1.2). Where does EU segmentation + hyperedge construction live: a new corpus-level Stage, or an extension of Graph Assembly? Settle: the EU record contract (edge_id, entity_ids, relations, tau_text, provenance, gamma, token_cost, support, has_relations), segmentation rules ownership (40–120-token targets, merge/split), arity cap m_max, dedup/merge semantics (same entity set + same relation signature only), weak EUs. Outcome: CONTEXT.md terms + ADR + record schema.

## Answer

**Recorded as [ADR-0057](../../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)**; CONTEXT.md updated with the `Hyperedge` term and the amended `Evidence` / `Graph Assembly` entries.

1. **The EU is NOT the existing `Evidence`, and is not named "Evidence Unit".** The two are
   structurally opposite: `Evidence` answers *why did we believe this edge* (audit, raw content
   **forbidden** in `snippet`, binary, pointer-only); the new unit answers *what text do we hand the
   model* (retrieval, raw content **is the point**, m-ary, carries `token_cost`). CONTEXT already
   polices one collision here (`Evidence` vs `Provenance`); a third term in that neighbourhood would
   be a domain-language defect. **Domain term: `Hyperedge`**, `_Avoid_: Evidence Unit, EU, fact, claim`.
2. **It is a third `GraphScope`** (`NODE`/`EDGE`/`HYPEREDGE`) with a `GraphHyperedge` payload on the
   existing `GraphRecord`, following the ADR-0025/0026/0027 scoped-record pattern — inheriting
   Provenance threading, checkpoint/resume, content-addressed ids, the S11 affected-set key, and
   Export for free. `SCHEMA_VERSION` bumps.
3. **Graph Assembly emits it — no new Stage.** It needs canonical `entity_ids`, so it can only exist
   post-Disambiguation, exactly where Graph Assembly already sits: the same input re-projected m-ary.
4. **`tau_text` follows the corpus redaction variant**, enforced at the Hyperedge writer. Otherwise
   the hypergraph is a PII bypass around ADR-0044's universal floor — the ADR-0042 truncation-leak
   defect class through a new door. Span stays in original coordinates for Provenance/Purge; where
   `extract_on: masked` is off, the writer needs an offset alignment (copy the ADR-0031 `offset_map`
   pattern, don't invent one).
5. **Mechanical parameters adopted from the spec as defaults** (not separately grilled): `m_max=12`
   with relation-group splitting; 40–120 token target, merge below 25, split above 180; dedup on
   identical entity-set **AND** relation signature with a `support` counter.
6. **Weak Hyperedges emitted by default behind a config toggle** — they carry propagation mass but
   are expected to outnumber relation-bearing units several-to-one; the eval (ticket 17) decides the
   default's fate.
7. **Upstream prerequisite surfaced by [ticket 01](01-research-spec-vs-repo-audit.md):** per-mention
   **link confidence must be persisted** through Disambiguation — glinker computes it and it is
   dropped before any durable record, leaving `gamma` and weighted incidence with no data source.
   Folded into M1 ([ticket 12](12-task-m1-eu-segmentation-hyperedge-builder.md)) scope.
