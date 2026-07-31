# Wayfinder map — Hypergraph Evidence Store + Knapsack Chain-of-Evidence

Label: `wayfinder:map`

## Destination

The framework's retrieval + data-generation layer reaches the frontier defined by [spec.md](spec.md):
hyperedge Evidence Units emitted by the pipeline as engine-agnostic sidecar **files**, query-aware
confidence propagation, and the knapsack chain-of-evidence selector shipped in `latence-retrieval` —
**proven** by MuSiQue/HotpotQA supporting-fact recall@budget curves strictly dominating the current
chunk-topk baseline AND an enterprise-shaped corpus eval. Plus a data-gen SOTA sweep whose contents
its own scoping ticket decides. Zero LLM calls in indexing or the online path.

## Notes

- **This map carries execution** (Dennis's charting decision): the M1–M6 `task` tickets implement the
  spec, not just decide. **Hard rule: check in with Dennis before ticket 12 (M1) — the first
  long-running implementation — starts.** Decisions tickets need no check-in.
- Tracker: repo-native layout — tickets in `tickets/NN-slug.md` (matching `issues/phase-4-*/tickets/`
  precedent, not `.scratch/`). Ticket headers: `Type:`, `Status:` (open/claimed/resolved),
  `Blocked by:`. Frontier = open + unblocked + unclaimed, lowest number first.
- Domain: consult `CONTEXT.md` (Evidence, Pack, Expand, Capability/Provider, Corpus Version) and
  ADRs 0017, 0048–0053 before any decision ticket. Grilling tickets use /grilling +
  /domain-modeling; task tickets meet the repo's DoD bars (`issues/DEFINITION-OF-DONE.md`) with
  /tdd + codebase-design.
- Non-negotiables from the spec: engine-agnostic (vector engine = ANN entry only, IDs in payloads);
  no LLM in offline construction or online retrieval; the index is built *for the solver*.
- Standing prefs: no fake green — real models, real datasets, real deps for any eval claim;
  diagnose tight (max 2 live validation runs per fix cycle); research tickets carry licence/IP/
  regulation verification per the research-diligence directive.
- Research subagents resolve their ticket file only; their Decisions-so-far pointers are synced by
  the next map session (concurrency safety).

## Decisions so far

<!-- one line per closed ticket: gist + link -->

- [Grilling: query-time model + entity_kb policy](tickets/06-grilling-query-time-model-policy.md) — **tiered entry, zero-model default** → [ADR-0059](../../docs/adr/0059-query-side-entry-is-tiered-zero-model-default-with-gliner2-strictly-opt-in.md). Tier 0 = ANN on `entity_vecs` with the caller's own query embedding; tier 1 = alias point-lookups in the KV (still zero models); tier 2 = GLiNER2 NER + `w_relq` from **one load**, strictly opt-in. Binding constraint: `latence-retrieval` is pydantic-only, so query models come as an **extra wrapping the upstream library**, never a dep on `latence-extract-gliner2`. **Multi-hop survives tier 0** (reach comes from propagation, not entry) — what degrades is anchor precision. Tier 2 costs tens of ms CPU vs the 2.3 ms structural path; eval must report **per tier**.

- [Grilling: token_cost tokenizer policy](tickets/05-grilling-token-cost-tokenizer-policy.md) — **approximate index-time weight + exact query-time verification** → [ADR-0058](../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md). Optional operator-named tokenizer (heuristic default, **nothing pinned**, ADR-0012 intact) with the **tokenizer identity stored beside the value**; `token_cost` is a ranking weight, and the budget guarantee comes from re-counting the *selected* 15–40 items with the generation tokenizer (~2–3 ms) — counting all 2000 candidates would cost ~160 ms. Mismatch → mandatory verification, never a scale factor. **Model swap needs no re-index.**

- [Grilling: EU domain model + stage placement](tickets/04-grilling-eu-domain-model-and-stage-placement.md) — **the unit is a `Hyperedge`, never "Evidence Unit"** (would be a third collision with `Evidence`/`Provenance`; `Evidence` is pointer-only and bars raw content, a Hyperedge is raw content) → [ADR-0057](../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md). Modelled as a **third `GraphScope` on `GraphRecord`**, emitted by **Graph Assembly (no new Stage)**; **`tau_text` follows the corpus redaction variant** (else the hypergraph is a PII bypass around ADR-0044); weak Hyperedges default-on behind a toggle; glinker link-confidence persistence folded into M1.

- [Grilling: where the hypergraph sidecar sits under the stateless boundary](tickets/03-grilling-sidecar-statefulness-ruling.md) — **A′: strict per-call over a point-lookup KV** → [ADR-0056](../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md). ADR-0048 **not** amended (statelessness stays structural); sidecar = pipeline files + derived rebuildable KV; **global `A_E` and incidence CSR deleted from the design**; postings carry γ inline; vector engine keeps two ID-payload collections. M2/M3 tickets rewritten accordingly; KV-under-concurrency spun out as ticket 19.

- [Prototype: residency simulation](tickets/18-prototype-residency-simulation.md) — measured, synthetic, 10k/100k/1M EUs: global `A_E` is **infeasible** (2.9B nnz @100k, 232B/2.6TiB @1M) **and slower than recomputing locally** where it fits → spec 1.7's key precomputation is refuted; residency changes **nothing** about what gets selected (100/100 identical, Jaccard 1.0) so it is a pure latency/ops call; strict-per-call over a **point-lookup KV is within 0.56 ms of fully resident and flat in corpus size** (2.26 vs 1.70 ms @1M) while strict-over-parquet-scan degrades to 69 ms. Feeds ticket 03.

- [Research: spec-vs-repo evidence audit](tickets/01-research-spec-vs-repo-audit.md) — glinker link confidence is **dropped** before persistence (γ/weighted-B have no data source today); everything hyperedge-shaped is new; sentence split is a naive regex (new dep needed); relation argument offsets recoverable via join to EntityMention spans (no schema change); existing Packer reusable as frozen-value packer **but** it's the same greedy family, so the spec's "boundary repair" premise is void; aliases/descriptions die inside glinker; parked branches have zero overlap; eval lands on latence-benchmark's pinned-dataset seam; nodes/edges parquet + DuckDB substrate partially reusable.
- [Research: external landscape, datasets, dependency licences](tickets/02-research-external-landscape-and-licences.md) — HyCE-RAG paper is real but its LLM-judge eval is incomparable (anchor on canonical metrics + HippoRAG-2 recall@5); no zero-LLM hypergraph-RAG line published → genuinely differentiated; MuSiQue CC BY 4.0 / HotpotQA CC BY-SA 4.0 (share-alike on derived artifacts); deps all green — the GLiREL licence blocker is **MOOT** (repo uses GLiNER2 fused extraction, Apache-2.0 weights+code, which covers every spec role GLiREL had; see ticket 02 addendum + spec.md adaptation note); GDPR delta none, but Purge must extend to the sidecar (parquet purge = rewrite/compaction).

## Not yet specified

- Weighted incidence B + pluggable Propagator kernel A/B (spec Part 6.1–6.2) — graduates after M3
  (ticket 14) and the eval harness exist to arbitrate. Note ADR-0056: weighting applies to the
  query-local `B_q` built per call, not to a stored global matrix.
- Whether the spec's `knapsack_refine` boundary-repair phase survives — ticket 01 found the existing
  Packer is the same greedy family, not DP/ILP, so "repair" may be a no-op. Graduates into ticket 15
  (M4) scope, or its own decision, once M4 is picked up.
- Learned token-level utility classifier trained on selection traces (spec 3.5) — after traces
  accumulate from real use; the UtilityModel seam ships in M4.
- Contents of the data-gen SOTA sweep — until its scoping ticket (11) resolves; may spin off its
  own map.
- Enterprise-shaped gold-set construction mechanics (which corpus, labeling method) — until eval
  design (08) resolves.
- Retrieval MCP surface for chains — exposing CoE output via the Retrieval MCP tools; after M5.
- Operational policy for incremental sidecar appends vs weekly compaction — after the deltas/purge
  ticket (10).

## Out of scope

- Groundedness / contradiction handling — explicitly excluded by the spec for this iteration.
- True iterative retrieve-resolve-retrieve multi-hop (reasoning step in the loop) — spec 6.3:
  document the associative-diffusion limitation, don't fake it.
- Any LLM call in indexing or online retrieval — LLM appears only at final answer generation,
  outside this effort.
- Building or operating a retrieval engine — ADR-0048's stateless boundary stands; we upgrade the
  tooling around the adopter's engine.
