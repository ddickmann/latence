# 06 — Grilling: query-time model + entity_kb policy

Type: grilling
Status: resolved
Blocked by: 01

## Question

Online entry (spec 2.1) needs GLiNER2 NER + glinker linking on the query string, and GLiNER2's zero-shot relation-type extraction for w_relq (spec 3.2) — today only rerank loads a model at query time. (GLiREL is NOT involved: the repo's fused GLiNER2 extraction covers all its spec roles — see ticket 02 addendum + the spec's adaptation note — so one model family may serve NER + w_relq in a single query-time load; grill whether that consolidation holds.) Decide: which models load in `latence-retrieval` (a query-signal-gen seam per ADR-0049?); how the entity_kb/alias table reaches the stateless query path; latency budget; the degraded path when models are absent (V_vec-only entry — how much does the frontier claim survive?). Outcome: seam design + ADR note.

## Answer

**Recorded as [ADR-0059](../../../docs/adr/0059-query-side-entry-is-tiered-zero-model-default-with-gliner2-strictly-opt-in.md).**

Binding constraint found while grilling: **`latence-retrieval` depends on `pydantic` alone** —
deliberately not on `latence-core` or any pipeline package. A query-side model must therefore be an
**extra wrapping the upstream library directly**, never a dependency on `latence-extract-gliner2`
(which would drag the whole pipeline into the downstream library).

**Three tiers, each independently useful:**

- **Tier 0 (default, no new models)** — `V0` from ANN over `entity_vecs` using **the query embedding
  the caller already computed** for their own first-stage search. No extra, no added inference.
- **Tier 1 (zero models)** — query n-grams against the alias table in the per-call KV (ADR-0056):
  a real `V_llm` anchor set from **point lookups**. Expected to carry most linking value on
  enterprise corpora where entities are named exactly.
- **Tier 2 (opt-in extra)** — GLiNER2 query NER **and** `w_relq` relation-type detection from **one
  model load**; named checkpoint, no default, lazy, on the `latence.retrieval` seam (ADR-0049),
  following `Reranker`'s ADR-0012 discipline.

**`entity_kb` = the alias table inside the per-call KV.** No resident KB, no service.

**Degradation is a gradient, not a cliff** — multi-hop reach comes from propagation over the
hypergraph, not from how entry entities were found, so tier 0 still surfaces evidence pure ANN never
would. What degrades: restart mass spreads over ANN-recalled entities rather than concentrating on
entities the query provably names; `w_relq` is absent and its weight redistributes.

**Latency, stated plainly:** ADR-0056's ~2.3 ms is the *structural* path; tier 2 adds GLiNER2
inference — tens of ms on CPU, an order of magnitude more than everything else combined. That cost
is why it is opt-in.

Consequence for the eval ([ticket 17](17-task-m6-eval-gate.md)): report the curve **per tier**, or
the headline number silently assumes tier 2 and misrepresents the default install.
