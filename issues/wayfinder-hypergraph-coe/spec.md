# Hypergraph Evidence Store + Knapsack Chain-of-Evidence (CoE)

> **Repo adaptation note (2026-07-31).** The original draft named **GLiREL** as the relation
> extractor. This repo does not use GLiREL and does not need it: relation extraction is the
> GLiNER2 unified fused provider (`fastino/gliner2-multi-v1`, Apache-2.0 weights+code; predecessor
> `relation.gliner_relex` likewise Apache-2.0), whose relation output natively carries per-relation
> confidence AND head/tail argument char offsets (resolved to true original offsets, ADR-0031;
> persisted as `RelationMention.head_mention_id`/`tail_mention_id` + `confidence`). All GLiREL
> references below have been replaced accordingly; the query-side relation-type signal (w_relq)
> uses GLiNER2 zero-shot relation/classification on the query string. This also voids the
> GLiREL licence blocker recorded in wayfinder ticket 02.

**Purpose.** Upgrade the existing retrieval framework (GLiNER2 NER + relation extraction, glinker entity disambiguation, metadata management, existing knapsack solver) with (1) hyperedge-oriented graph construction and (2) a query-aware chain-of-evidence selector that maximizes utility under a token budget. Adapted from HyCE-RAG (arXiv:2607.22597), but **with zero LLM calls in indexing or the online path** — all extraction stays on the GLiNER-family models. Groundedness/contradiction handling is explicitly **out of scope** for this iteration.

**Hard design constraints**

1. Engine-agnostic: the vector engine (Weaviate/Qdrant/pgvector/FAISS/whatever) serves **only** ANN entry retrieval. Everything structural lives in a sidecar store (Parquet/DuckDB + scipy sparse + a KV store). Swapping the vector engine must never touch graph or solver code.
2. No LLM in offline construction or online retrieval. LLM appears only at final answer generation (outside this spec).
3. The index is built **for the solver**: every quantity the online selector needs (token costs, confidences, co-membership, degrees) is precomputed at index time. The online path is pure sparse linear algebra + the knapsack solver.

---

## Part 1 — Hyperedge-Oriented Graph Construction (Offline)

### 1.1 Core representational shift

Today the pipeline emits **binary triples** `(head, relation, tail)` per chunk. Binary edges fragment evidence: a sentence stating "Company A acquired Company B in 2019 for €X, pending approval by Regulator C" becomes 3–4 disconnected pairwise edges, and the joint fact is lost. The shift:

> **Hyperedge = one Evidence Unit (EU): a minimal text span, the full set of canonical entities occurring in it, the set of directed relations extracted from it, and provenance back to the source chunk/document.**

The EU is the atomic thing the solver later selects. Entities are vertices; hyperedges connect m ≥ 2 vertices simultaneously.

### 1.2 Evidence Unit segmentation

Do **not** equate EU with chunk. Chunks are storage/retrieval granularity; EUs are reasoning granularity.

Segmentation rules (deterministic, no LLM):

1. Sentence-split each chunk (use a proper splitter, e.g. `pysbd`/spaCy senter — abbreviation-safe for DE/EN).
2. Run GLiNER2 NER + relation extraction per sentence window (window = 1 sentence, extend to 2–3 sentences **only** when a relation's arguments span the boundary — the GLiNER2 relation head/tail argument offsets tell you this).
3. An EU is the **minimal contiguous sentence window covering all arguments of at least one extracted relation**, merged with any overlapping EU in the same chunk (union of spans, union of entities/relations).
4. Sentences with entities but no relations: emit a *weak EU* (entities only, `has_relations=false`) — these still participate in propagation but get a utility discount later.
5. **Size targets (matters for the knapsack):** aim for EUs of **40–120 tokens**. Below ~25 tokens, merge into the neighbor EU sharing the most entities. Above ~180 tokens, split at the sentence boundary that minimizes cut relations. A tight, roughly unimodal cost distribution is what makes the budgeted selection later behave well — huge variance in item sizes degrades knapsack packing quality.

### 1.3 Building hyperedges from GLiNER2 fused output

Per EU `u`:

```
entities_raw  = GLiNER2.ner(u.text)                      # mentions + type + score
relations     = GLiNER2.extract_relations(u.text, ...)    # head/tail: text, confidence, char offsets
entities_can  = glinker.link(entities_raw, entity_kb)    # mention -> canonical entity_id + link score
```

Hyperedge record:

```
e = {
  edge_id:        hash(doc_id, span_start, span_end),
  entity_ids:     sorted unique canonical ids (m >= 2 required; else weak EU or drop),
  relations:      [(head_id, rel_type, tail_id, conf_rel)],
  tau_text:       EU span text (verbatim),
  provenance:     {doc_id, chunk_id, char_start, char_end, section, timestamp, source_meta},
  gamma:          extraction confidence (see 1.5),
  token_cost:     len(tokenizer.encode(tau_text)),        # THE knapsack weight — precomputed
  support:        1  (incremented on merge, see 1.4),
  has_relations:  bool
}
```

Cap hyperedge arity at `m_max = 12`; if an EU exceeds it (tables, dense lists), split by relation groups (entities co-occurring in the same relation stay together).

### 1.4 Canonicalization and merging (glinker does the heavy lifting)

- glinker resolves every mention to a global `entity_id`; keep the alias table `alias -> entity_id` it produces/uses. Persist per-mention link confidence.
- Entity record: `{entity_id, canonical_name, type, aliases[], description}` where `description` is composed **deterministically**: canonical name + type + top-k most frequent relation contexts (e.g., "ZF Friedrichshafen — ORG — supplier_of: BMW, acquired: WABCO"). This composed string is what gets embedded (1.6) — it makes the entity index query-matchable without any LLM summarization.
- **Hyperedge dedup/merge:** two hyperedges are equivalent iff same `entity_ids` set AND same relation signature `{(head, rel_type, tail)}`. On merge: keep the highest-γ instance's `tau_text` as primary, append other provenances, `support += 1`. `support` becomes a frequency prior in scoring — corpus-repeated facts are more reliable.
- Do **not** merge across differing relation signatures even with identical entity sets (those are distinct facts about the same group).

### 1.5 Confidence calibration (replaces the paper's LLM extraction confidence)

The paper's γ(e) comes from an LLM. You get it cheaper and better-behaved:

```
gamma(e) = w1 * mean(link_conf of entities)        # glinker
         + w2 * mean(conf_rel of relations)         # GLiNER2 relation confidence
         + w3 * min(ner_scores)                     # weakest-mention penalty
```

with `w = (0.3, 0.5, 0.2)` as starting point. **Calibrate** the raw GLiNER2 scores first (isotonic regression against a small hand-labeled sample per domain, ~200 EUs) so γ is comparable across entity types. Weak EUs (no relations): `gamma *= 0.5`.

### 1.6 Storage layout (engine-agnostic sidecar)

```
store/
  entities.parquet        # entity_id, name, type, aliases, description, degree
  hyperedges.parquet      # edge_id, entity_ids[], relations[], gamma, token_cost,
                          # support, has_relations, provenance, tau_ref
  eu_text.lmdb            # edge_id -> tau_text  (or duckdb blob table)
  incidence.npz           # scipy CSR: B (|V| x |E|), values in {0,1}
  entity_postings.pkl     # entity_id -> [edge_ids sorted by gamma desc]   (inverted lists)
  edge_adjacency.npz      # scipy CSR: A_E (|E| x |E|), A_E[i,j] = |shared entities|  (see 1.7)
  degrees.npz             # d_v, d_e vectors
  aliases.duckdb          # alias -> entity_id
```

Vector engine holds exactly two collections, nothing else:

- `entity_vecs`: embedding of `entity.description` → payload `{entity_id}`
- `eu_vecs`: embedding of `tau_text` → payload `{edge_id}` (used for the textual-relevance term and as fallback entry path)

Any ANN engine works because the payload is only an ID; all joins happen in the sidecar.

### 1.7 Precomputation the solver depends on

At index time compute and persist:

1. **`token_cost` per hyperedge** with the *same tokenizer as the target generation model*. This is the knapsack weight; never compute it online.
2. **Edge–edge co-membership `A_E = Bᵀ·B`** (sparse, zero the diagonal), i.e., how many entities two hyperedges share. This gives O(1) lookup of connectivity and redundancy during greedy selection — the single most important precomputation for solver speed.
3. Degree vectors `d_v = B·1`, `d_e = Bᵀ·1`.
4. Per-entity postings sorted by γ (fast candidate generation with early termination).
5. Optional: per-entity truncated PPR sketches over the global hypergraph (top-200 neighbors) if corpora are large and you want to skip online propagation for latency-critical paths; otherwise online propagation on the query-local subgraph is cheap enough (Part 2).

**Incremental updates:** appends only touch `entity_postings`, `B` (grow columns), `A_E` rows for new edges, and the two vector collections. No global recompute. Deletions: tombstone edge_ids, filter at query time, compact weekly.

---

## Part 2 — Online: Query-Aware Subgraph + Confidence Propagation

No LLM anywhere in this section. Query understanding = GLiNER2 on the query string.

### 2.1 Entry entities

```
mentions   = GLiNER2.ner(query)                          # same model as indexing
V_llm      = glinker.link(mentions, entity_kb)           # canonical ids, high precision
V_vec      = ANN_topK(entity_vecs, embed(query), K=10)   # recall branch
V0         = dedup(V_llm ∪ V_vec)
```

Restart/relevance vector per entity `v` in the candidate universe:

```
r_hat(v) = alpha * sim_norm(query, v) + (1 - alpha) * 1[v ∈ V_llm]
r        = r_hat / (||r_hat||_1 + eps)          # alpha = 0.6 start
```

Note the indicator uses `V_llm` (linked, precise) not the union — the paper uses the full entry set; restricting to the linked branch keeps the restart mass anchored on entities the query *actually names*.

### 2.2 Candidate hyperedge universe

```
E_c = union of entity_postings[v] for v in V0          # 1-hop
E_c = E_c ∪ {top edges by A_E adjacency to E_c}        # optional 2-hop, cap |E_c| <= 2000
V_q = union of entity_ids over E_c
B_q = incidence submatrix (V_q x E_c)                  # slice from global CSR
```

### 2.3 Confidence propagation (random walk with restart on the bipartite incidence)

```
S      = B_q @ inv(D_e) @ B_q.T @ inv(D_v)             # column-stochastic, sparse
c(0)   = r
c(t+1) = lam * r + (1 - lam) * S @ c(t)                # lam = 0.2, K = 4..6 steps
rho(e) = mean over incident vertices of c*(v)          # hyperedge confidence
```

For `|V_q| < ~500` you may use the closed form `c* = lam·(I − (1−lam)S)⁻¹·r` (sparse solve); iteration is simpler and equally fine. Drop isolated vertices / empty edges before building `S` (degrees must be > 0).

This is the "reasoning beyond cosine" mechanism: entities with zero direct query similarity gain confidence through shared hyperedges with anchored entities, and `rho(e)` ranks multi-hop-relevant evidence that HNSW alone would never surface.

---

## Part 3 — Knapsack Chain-of-Evidence

### 3.1 Framing

The existing solver stays the optimization core. What changes is the **item definition and the value function**:

- **Item** = hyperedge (Evidence Unit), not chunk.
- **Weight** = `token_cost(e)` (precomputed, Part 1.7).
- **Budget** `T` = context-token budget for evidence.
- **Value** = *query-conditional marginal utility* `v(e | A)` where `A` is the already-selected set — because plain additive knapsack values cannot express redundancy or chain formation. This marginal structure is exactly the "iteratively compose utility" behavior wanted, implemented today with a heuristic utility and later swappable for the learned token-level classifier (3.5).

### 3.2 Utility function

Base (context-free) value:

```
v0(e) = w_prop * rho(e)                       # propagated confidence   (0.35)
      + w_rel  * sim_norm(query, tau(e))      # EU-text vs query, from eu_vecs (0.20)
      + w_ext  * gamma(e)                     # extraction confidence   (0.15)
      + w_ent  * |entities(e) ∩ V_llm| / |entities(e)|   # entry overlap (0.15)
      + w_sup  * log1p(support(e)) / log1p(s_max)        # frequency prior (0.10)
      + w_relq * mean(conf_rel(r) for r matching query relation types)   # (0.05)
```

`w_relq`: run GLiNER2's zero-shot relation-type extraction on the query itself (cheap) to detect which relation types the query asks about; hyperedges carrying those relation types get the bonus. Weak EUs (`has_relations=false`): `v0 *= 0.6`.

Marginal value given selected set `A`:

```
v(e | A) = v0(e)
         * (1 - red(e, A))                    # redundancy discount
         + w_conn * conn(e, A)                # chain-formation bonus (w_conn = 0.15)

red(e, A)  = max over a in A of  A_E[e, a] / |entities(e)|        # entity-overlap ratio, O(1) via A_E
             (raise to relation-level: if relation signature ⊂ signatures(A), red = 1.0 → hard skip)
conn(e, A) = 1 if e shares >= 1 entity with A ∪ V0 else 0
             (0 for the first item; V0 counts so the chain starts anchored at the query)
```

Properties: monotone, and the redundancy-discount form is submodular-ish by construction — diminishing returns as `A` grows. That licenses the greedy solver below with a near-(1 − 1/e) quality expectation in practice.

### 3.3 Solver: budgeted greedy with lazy evaluation, on top of the existing knapsack

Recommended composition — keep the existing knapsack as the **final packer**, add a marginal-utility loop around it:

```
ALGORITHM CoE-Select(E_c, rho, T):
  A = []; used = 0
  H = max-heap of (v0(e)/token_cost(e))            # density-ordered, lazy
  while used < T and H nonempty:
    e = pop(H)
    m = v(e | A) / token_cost(e)                    # recompute marginal density
    if m < peek(H).cached_density:                  # lazy greedy: stale? push back
        push(H, e with cached_density = m); continue
    if used + token_cost(e) <= T and v(e | A) > theta_min:
        A.append(e); used += token_cost(e)
  # final refinement with the existing solver:
  #   run knapsack over A ∪ {next 20 rejected candidates} with FROZEN values v(e | A_final)
  #   to repair greedy's boundary mistakes (swap a large low-density item for two small ones)
  return knapsack_refine(A, candidates, T)
```

- `theta_min` (~0.15 · max v0): stop adding filler once marginal value collapses; do not spend the budget just because it exists.
- The two-phase design (greedy for marginals → existing knapsack for boundary repair with frozen values) is what lets the DP/ILP solver be reused unchanged: it only ever sees additive values.
- Complexity: candidate gen O(|E_c| log |E_c|), each marginal O(|A|) with `A_E` lookups; typical query: 2000 candidates, 15–40 selected, sub-10ms after propagation.

### 3.4 Chain assembly and ordering (post-selection)

Selection gives a *set*; generation wants *chains*.

```
1. Build G_sel: nodes = selected hyperedges, edge (e_i, e_j) iff A_E[e_i, e_j] >= 1.
2. Connected components of G_sel = one chain each.
3. Within a component: order by hop distance of each hyperedge's closest entity to V0
   (BFS layers from V0 over the component), tie-break by rho desc.
4. Emit structured context per chain:

   ## Chain k (anchor: <entity names ∩ V0>)
   [step 1] (rho=.82, gamma=.91, src=doc_17#c4) <tau_text>
            entities: A, B, C | relations: A --acquired--> B, ...
   [step 2] ...
```

Chains, not shuffled chunks, are the generation input. Components not connected to any V0 entity (possible after 2-hop expansion) go last under a "peripheral evidence" header or are dropped if budget-tight.

### 3.5 Hook for the future token-level utility classifier

Keep `v0` and `v(e|A)` behind a single interface:

```
class UtilityModel(Protocol):
    def base(self, query_feats, edge_feats) -> float
    def marginal(self, query_feats, edge_feats, selected_state) -> float
```

The heuristic above is `HeuristicUtility`. From day one, **log a selection trace per query**: `{query, V0, E_c features, rho, per-step (e, v(e|A), accepted/rejected), final A, downstream answer id}`. These traces are the weak-supervision dataset for the learned classifier (features are already token-attributable via `tau_text` offsets from GLiNER2 — the labels can later come from answer-attribution / leave-one-out ablation). No pipeline change needed when the classifier lands; it just replaces the heuristic behind the interface.

---

## Part 4 — Why the index is shaped this way (summary of solver-driven decisions)

| Index decision | Solver requirement it serves |
|---|---|
| `token_cost` precomputed with the generator's tokenizer | knapsack weight must be exact and free at query time |
| EU size targets 40–120 tokens, merge/split rules | homogeneous item weights → better packing, meaningful marginals |
| `A_E` edge co-membership matrix | O(1) redundancy + connectivity in every marginal evaluation |
| hyperedge dedup with `support` counter | removes trivially redundant items *before* the solver; support becomes a prior |
| per-entity postings sorted by γ | candidate generation with early termination |
| calibrated γ | values comparable across domains/entity types → stable weight tuning |
| entity `description` composed from relations | entity ANN entry works without LLM summaries |
| vector engine stores IDs only | engine swap never touches solver/graph code |

## Part 5 — Implementation order & tests

1. **M1 — EU segmentation + hyperedge builder** (Part 1.2–1.4). Test: on a fixture corpus, every relation's arguments are inside its EU span; no EU > 180 tokens; merge idempotency.
2. **M2 — Sidecar store + precomputation** (1.6–1.7). Test: `B`, `A_E`, degrees consistent (`A_E == Bᵀ·B − diag`); incremental append equals full rebuild.
3. **M3 — Entry + propagation** (Part 2). Test: `S` column-stochastic; `c(t)` converges (‖c(t+1)−c(t)‖₁ < 1e−6 within K≤10 for λ=0.2); a synthetic 2-hop corpus ranks the bridging hyperedge above a lexically-similar distractor.
4. **M4 — CoE selector** (Part 3). Test: budget never exceeded; hard-redundant edge (identical relation signature) never selected twice; greedy+refine ≥ greedy alone on value; determinism given fixed seed.
5. **M5 — Chain assembly + trace logging** (3.4–3.5).
6. **Eval:** MuSiQue / HotpotQA supporting-fact recall@budget vs. current chunk-topk baseline; report tokens-in-context vs. supporting-fact coverage curves. Target: strictly dominate the baseline curve.

Suggested module layout:

```
retrieval/
  segmentation/eu_builder.py
  graph/canonicalize.py        # glinker integration, merge
  graph/store.py               # parquet/lmdb/CSR accessors
  graph/precompute.py          # A_E, degrees, postings
  online/entry.py
  online/propagate.py
  online/utility.py            # HeuristicUtility + UtilityModel protocol
  online/select.py             # CoE-Select + knapsack_refine adapter
  online/chains.py
  eval/supporting_facts.py
```

---

## Part 6 — Diffusion-kernel notes and known limitations (addendum)

### 6.1 Weighted incidence (do this — replaces binary B)

The propagation in Part 2.3 uses binary incidence, i.e., the standard clique/star
expansion of the hypergraph. That treats every entity in a hyperedge as an equal
confidence distributor and discards role information. Upgrade:

```
B[v, e] = link_conf(v, e) * role_weight(v, e)

role_weight = 1.0  if v is head or tail of any relation in R(e)
              0.6  if v is a peripheral mention (entity present, no relation role)
```

Both factors are known at index time (glinker + GLiNER2 head/tail argument offsets) — persist
the weighted B alongside degrees computed from it. Normalization in S is unchanged
(row/column sums over weighted entries). This is edge-dependent vertex weighting
(Chitra & Raphael style) and preserves hyperedge semantics that uniform expansion
destroys, at zero extra online cost.

### 6.2 Pluggable diffusion kernel

The RWR update is one instance of diffusion on the incidence topology. Put the
kernel behind an interface so it can be A/B tested without touching callers:

```
class Propagator(Protocol):
    def propagate(self, B_q, r, params) -> c_star

RWRPropagator          # default: c(t+1) = lam*r + (1-lam)*S*c(t), K steps
HeatKernelPropagator   # c* = expm_multiply(-t * L, r), L = I - S; Poisson-weighted
                       # walk lengths instead of geometric — smoother multi-hop decay
```

Expect marginal ranking differences at query-subgraph scale; keep RWR as default
and let eval decide.

### 6.3 Known limitation: associative vs. sequentially constrained multi-hop

Diffusion (any kernel) answers "what is structurally close to the query anchors" —
associative multi-hop. It does NOT execute ordered constraint chains of the form
"the supplier of the OEM that acquired X in 2019" where hop 2 depends on resolving
hop 1 first. Partial mitigation without LLM steps:

- Chain assembly (3.4) already orders by hop distance along shared entities; extend
  it to prefer paths that follow *directed* relation edges head->tail when the query's
  detected relation types (the w_relq signal) imply direction.
- Optional typed-traversal pass: if the query yields exactly one linked anchor and
  one detected relation type, do a deterministic 1-step relation-typed expansion from
  the anchor BEFORE propagation and add the reached entities to V0 with weight 0.5.

Anything beyond that (true iterative retrieve-resolve-retrieve) requires a reasoning
step in the loop and is deliberately out of scope for this iteration. Document it;
don't fake it with deeper diffusion.
