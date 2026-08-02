# Wayfinder map — Hypergraph Evidence Store + Knapsack Chain-of-Evidence

Label: `wayfinder:map`

## Destination

The framework's retrieval + data-generation layer reaches the frontier defined by [spec.md](spec.md):
hyperedge Evidence Units emitted by the pipeline as engine-agnostic sidecar **files**, query-aware
confidence propagation, and the knapsack chain-of-evidence selector shipped in `latence-retrieval` —
**proven** by MuSiQue/HotpotQA supporting-fact recall@budget curves dominating the current
chunk-topk baseline (AUC + endpoints, ticket 08) AND a manually-gold-labeled enterprise corpus
eval. Zero LLM calls in indexing or the online path. (The data-gen SOTA sweep left this
destination — ticket 11: hyperedges inherit every upstream improvement by construction, so
generic pipeline quality is a spin-off map charted after M6 from measured weak spots.)

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
- **STANDING RULE (Dennis, 2026-07-31): a capability is only dropped or weakened when DATA says so.**
  Parking a frontier feature out of caution is counter-innovative. Two consequences: (1) every
  sacrifice in an ADR must cite a measurement, and one that cannot is re-opened; (2) the shipped
  **default** should be the strong configuration with a documented cheap mode, not the reverse.
  The budget this is traded against is **20–50 ms** for the structural query path — established from
  the adopter's own first-stage ANN (10–50 ms) and the cross-encoder rerank this repo already ships
  (50–200 ms), against a measured 2.3 ms structural path (ADR-0056). Roughly an order of magnitude
  of unspent headroom; spend it.
- Standing prefs: no fake green — real models, real datasets, real deps for any eval claim;
  diagnose tight (max 2 live validation runs per fix cycle); research tickets carry licence/IP/
  regulation verification per the research-diligence directive.
- Research subagents resolve their ticket file only; their Decisions-so-far pointers are synced by
  the next map session (concurrency safety).

## Decisions so far

<!-- one line per closed ticket: gist + link -->

- **[OFFICIAL GATE PASSED — both full corpora](tickets/17-task-m6-eval-gate.md)** (2026-08-02): MuSiQue-full test **AUC +15.1%** (+10.4 pts @16k), HotpotQA-full test **AUC +6.6%** (+9.7 pts @16k), endpoints hold, worst 0.0%, single pre-registered test scorings. **Tier 2 wins both datasets**; exact≈heuristic costs (ADR-0058 confirmed at scale); weak = tail recall; **calibration recall-neutral** (per-regime isotonic — the judged sample's real yield is the upstream extraction-quality evidence file). THE MAP'S DESTINATION IS REACHED IN FULL.

- [Task: M6 — the eval gate](tickets/17-task-m6-eval-gate.md) — **SUBSET GATE PASSED on both datasets** (2026-08-02): dev-tuned budget-conditional policy (baseline@1k, CoE@≥2k), test + enterprise scored unchanged — MuSiQue AUC **0.508 vs 0.478** (+11.2 pts @16k), enterprise AUC **0.820 vs 0.707** (+15.6 pts @4k+), endpoints hold, worst point 0.0%. Both scorings disclosed (flat share FAILED the 1k endpoint and is on the record). Remaining for the official numbers: the full-corpus pod run with per-tier curves + the four A/Bs + recall@5.

- [Task: enterprise gold set + calibration data](tickets/21-task-enterprise-gold-set.md) — **complete + validated** (harness verified; data LLM-generated at Dennis's direction with provenance recorded, pending his spot-check): 40-doc corpus with 80 planted cross-doc chains, byte-deterministic export (226 hyperedges), **100 gold questions (validate exit 0)**, **226 calibration judgments (224/1/1)**. The generation runs **found two real builder defects** (stream-copy support ×3 with PII-primary risk; duplicate relation triples) — fixed at source with regression tests, suite 4730→**4732P/0F**, before the data was accepted.

- [Task: M5 — chain assembly + traces](tickets/16-task-m5-chains-traces.md) — **BUILT + green** (subagent, verified after its suite-gate stall): sparse chain components, BFS-from-anchor ordering, spec-6.3 directed preference behind a `relations_lookup` seam, peripheral evidence last, byte-deterministic emission, redaction-safe τ by construction; selection **trace hook** (ids/counts only — no raw text) seeding the future learned utility. Suite 4694→**4730P/0F** (incl. the labeling harness's 22).

- [Task: M4 — chain-of-evidence selector](tickets/15-task-m4-coe-selector.md) — **BUILT + green** (subagent, verified): UtilityModel/HeuristicUtility + lazy greedy with theta_min, **incremental sparse redundancy** (the measured requirement), **ADR-0058 exact verification pass live** (mandatory on counter mismatch, trim not scale); **the M3-deferred ranking claim now pinned — the 2-hop chain beats the maximally-advantaged lexical distractor AT SELECTION**; `knapsack_refine` **dropped with recorded rationale** (same greedy family = no-op; frozen-value DP optimises the wrong objective); dedup invariant makes the hard-redundancy case unrepresentable. Suite 4679→**4694P/0F**. [M4-RESULT.md](M4-RESULT.md).

- [Task: KV under concurrent load](tickets/19-task-kv-concurrency-validation.md) — **measured, fine as-is**: per-call open = 0.04–0.06 ms p50, flat c=1→16 and 50k→500k HEs; c=16 tail is CPU saturation in reads, not open contention — ADR-0056's contingency fix would save nothing and stays unbuilt. Cold-cache honestly unmeasured (needs sudo). ADR-0056's consequences updated.
- [Grilling: licence placement](tickets/07-grilling-licence-placement.md) — **all hypergraph/CoE modules stay PolyForm-NC** (Dennis); Apache set stays exactly the seven generic primitives, publish-readiness pin untouched; revisitable after the M6 gate as a fresh decision.
- [Grilling: calibration ownership](tickets/09-grilling-calibration-ownership.md) — **shipped tooling, adopter labels** (Dennis): `latence calibrate` fits isotonic from ~200 labeled Hyperedges → calibration sidecar, absent ⇒ today's uncalibrated γ (default unchanged); the sample rides ticket 21's labeling harness; calibrated-vs-not is an M6 A/B; builds after M5.
- [Grilling: deltas/Purge integration](tickets/10-grilling-deltas-purge-integration.md) — **correct by inheritance, proven by test**: corpus-level re-runs regenerate hyperedges/parquet/KV per Corpus Version; no query-time tombstones needed; Purge expunges `tau_text` (new e2e pins it, incl. no collateral erasure); churn stats + incremental maintenance = measured-need residuals.
- [Grilling: data-gen SOTA sweep scope](tickets/11-grilling-datagen-sota-sweep-scope.md) — **premise dissolved by Dennis's question**: Hyperedges re-project the same extracted records, so every upstream improvement flows in by construction; generic pipeline-quality work = spin-off map after M6, charted from measured eval weak spots. Destination narrowed accordingly.

- [Task: M3 — entry + confidence propagation](tickets/14-task-m3-entry-propagation.md) — **BUILT + green** in `latence-retrieval` (new module, nothing existing touched): the **ticket-20-verified order** (expand hop-priority/hub-capped → `s_hyp` prune → RWR on small `B_q` → vertex gate + coverage), **zero-dep pure Python** (viable because prune precedes propagation; package stays pydantic-only); ADR-0059 tiers 0/1 live against the KV alias table; independent format-compatible `HyperedgeKV` reader with **interop pinned** against core's writer; `Propagator` seam (heat-kernel stays fog). Two math-backed spec deviations pinned in tests: the 1e-6@K≤10 convergence claim is impossible worst-case (geometric bound + head-rank-stability instead), and ρ-beats-anchor-touching-distractor is structurally false (the true claim — 2-hop entity gains confidence from zero — is pinned; selection-level ranking is M4's). Suite 4668→**4679P/0F**.

- [Grilling: eval design + enterprise gold set](tickets/08-grilling-eval-design.md) — MuSiQue + HotpotQA supporting-fact **recall@budget curves (1k–16k)** + recall@5 for HippoRAG-2 comparability, vs our own dense+Packer chunk-topk baseline; **gate = AUC + endpoint dominance on both datasets, no point >2% worse** (Dennis); **enterprise eval = MANUAL ~100-question gold set** over the real messy corpus (Dennis labels; harness must make it efficient → ticket 21, now blocking 17); **compute = subset-first, one full pod run at gate time**; per-tier / heuristic-vs-exact / weak-on-off A/Bs all reported.

- [Task: M2 — sidecar store + precomputation](tickets/13-task-m2-sidecar-store-precompute.md) — **BUILT + green**: the ADR-0056 KV as a **zero-dep byte-reproducible single-file format** (mmap binary search, per-call open/close; format-is-the-contract, LMDB stays an optional adapter), shipped by the KG Export beside the parquet; postings γ-sorted with γ inline; **alias keyspace live** (ADR-0059 tier-1 works today, canonical names; glinker's richer aliases still fog); spec-1.4 entity descriptions + degree stamped on nodes (the zero-LLM embedding input); **ADR-0058 index side complete** (`token_counter: "tiktoken:<enc>"`, lazy, loud on missing, identity stored). No `A_E`, no CSR. Suite 4661→**4668P/0F**. Default path byte-identical, e2e-proven.

- [Task: M1 — Hyperedge segmentation + builder](tickets/12-task-m1-eu-segmentation-hyperedge-builder.md) — **BUILT + green** on `hypergraph-m1`: contracts v20 (`GraphHyperedge`, third `GraphScope`, link-confidence persistence), zero-dep DE/EN sentence splitter, the full builder (segmentation, size targets, arity split, dedup+support, γ, **redaction-variant `tau_text` enforced at the writer**), opt-in Runner wiring (`wants_document_streams`, byte-identical when off), `graph-hyperedges.parquet` export + Quality-Report section. Suite: baseline 4644P/1F → **4661P/0F**, ruff + strict mypy clean, 16 new tests incl. the PII-bypass regression and two real-Runner e2e paths. **The hypergraph now exists.**

- [Prototype: frontier headroom](tickets/20-prototype-frontier-headroom.md) — initial negative result **overturned by paper verification**: my pipeline propagated over the raw expansion; HyCE-RAG **prunes to TopK_h by `s_hyp` BEFORE propagating** and gates vertices post-propagation (Eq 16–33). Paper-faithful order = **2× selection recall at 2.7× lower latency** (hits the candidate ceiling under realistic bridge sims); the old harness was also information-theoretically unfair (bridges got minimum textual signal). Multi-hop thesis NOT at risk — consistent with the repo's already-validated KG multi-hop. Still standing: 2-hop expansion load-bearing; hop-priority truncation; incremental sparse overlap (dense `|E_c|²` is the latency cap); PPR sketches bounded (82 MiB@200k); weighted incidence ~12%, not free. M3/M4 rewritten with the verified order.

- [Grilling: EU domain model + stage placement](tickets/04-grilling-eu-domain-model-and-stage-placement.md) — **the unit is a `Hyperedge`, never "Evidence Unit"** (would be a third collision with `Evidence`/`Provenance`; `Evidence` is pointer-only and bars raw content, a Hyperedge is raw content) → [ADR-0057](../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md). Modelled as a **third `GraphScope` on `GraphRecord`**, emitted by **Graph Assembly (no new Stage)**; **`tau_text` follows the corpus redaction variant** (else the hypergraph is a PII bypass around ADR-0044); weak Hyperedges default-on behind a toggle; glinker link-confidence persistence folded into M1.

- [Grilling: where the hypergraph sidecar sits under the stateless boundary](tickets/03-grilling-sidecar-statefulness-ruling.md) — **A′: strict per-call over a point-lookup KV** → [ADR-0056](../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md). ADR-0048 **not** amended (statelessness stays structural); sidecar = pipeline files + derived rebuildable KV; **global `A_E` and incidence CSR deleted from the design**; postings carry γ inline; vector engine keeps two ID-payload collections. M2/M3 tickets rewritten accordingly; KV-under-concurrency spun out as ticket 19.

- [Prototype: residency simulation](tickets/18-prototype-residency-simulation.md) — measured, synthetic, 10k/100k/1M EUs: global `A_E` is **infeasible** (2.9B nnz @100k, 232B/2.6TiB @1M) **and slower than recomputing locally** where it fits → spec 1.7's key precomputation is refuted; residency changes **nothing** about what gets selected (100/100 identical, Jaccard 1.0) so it is a pure latency/ops call; strict-per-call over a **point-lookup KV is within 0.56 ms of fully resident and flat in corpus size** (2.26 vs 1.70 ms @1M) while strict-over-parquet-scan degrades to 69 ms. Feeds ticket 03.

- [Research: spec-vs-repo evidence audit](tickets/01-research-spec-vs-repo-audit.md) — glinker link confidence is **dropped** before persistence (γ/weighted-B have no data source today); everything hyperedge-shaped is new; sentence split is a naive regex (new dep needed); relation argument offsets recoverable via join to EntityMention spans (no schema change); existing Packer reusable as frozen-value packer **but** it's the same greedy family, so the spec's "boundary repair" premise is void; aliases/descriptions die inside glinker; parked branches have zero overlap; eval lands on latence-benchmark's pinned-dataset seam; nodes/edges parquet + DuckDB substrate partially reusable.
- [Research: external landscape, datasets, dependency licences](tickets/02-research-external-landscape-and-licences.md) — HyCE-RAG paper is real but its LLM-judge eval is incomparable (anchor on canonical metrics + HippoRAG-2 recall@5); no zero-LLM hypergraph-RAG line published → genuinely differentiated; MuSiQue CC BY 4.0 / HotpotQA CC BY-SA 4.0 (share-alike on derived artifacts); deps all green — the GLiREL licence blocker is **MOOT** (repo uses GLiNER2 fused extraction, Apache-2.0 weights+code, which covers every spec role GLiREL had; see ticket 02 addendum + spec.md adaptation note); GDPR delta none, but Purge must extend to the sidecar (parquet purge = rewrite/compaction).

## Not yet specified

- **Extraction quality is the next frontier, now measured** (official calibration sample,
  M6-CALIBRATION-OFFICIAL.md): GLiNER2@0.5 on wiki text fabricates relations at scale — 58% of
  relation-bearing Hyperedges carry ≥1 unasserted triple (`capital of` 1/11 genuine), the γ high
  tail is the LEAST reliable, and γ≤0.5 ⟺ weak by construction (calibration must be per-regime —
  staged as `calibration-map-per-regime.json`). This is the opening evidence file for the
  post-M6 data-gen spin-off map (ticket 11's designed destination) — the retrieval gates pass
  DESPITE extraction noise, so extraction gains compound on top.

- **The calibration A/B needs a noisier corpus** (ticket-21 calibration finding): γ takes only
  two values on the pattern-extractor/planted corpus (0.79 and 0.45), so isotonic calibration has
  nothing to bend — the M6 calibrated-vs-uncalibrated A/B must run on the MuSiQue/HotpotQA side
  (GLiNER2 extraction, continuous confidences), not the enterprise corpus.

- **Hyperedge yield is a chunker knob** (ticket-21 finding 3: the same corpus gave 40 hyperedges
  at chunk `max_tokens: 220` vs 226 at 90 — sentence windows rarely cross chunk boundaries, so
  chunk size gates segmentation). The M6 eval should sweep chunk size as a reported dimension;
  if the coupling is confirmed harmful, the fix is windowing across chunk boundaries inside the
  builder — a measured-need change, not built on suspicion.

- Weighted incidence B + pluggable Propagator kernel A/B (spec Part 6.1–6.2) — graduates after M3
  (ticket 14) and the eval harness exist to arbitrate. Note ADR-0056: weighting applies to the
  query-local `B_q` built per call, not to a stored global matrix.
- Learned token-level utility classifier trained on selection traces (spec 3.5) — after traces
  accumulate from real use; the UtilityModel seam ships in M4.
- Retrieval MCP surface for chains — exposing CoE output via the Retrieval MCP tools; after M5.
- Operational policy for incremental sidecar appends vs weekly compaction — after the deltas/purge
  ticket (10).

## Out of scope

- **The data-gen SOTA sweep** — [ticket 11](tickets/11-grilling-datagen-sota-sweep-scope.md)
  dissolved the premise: Hyperedges re-project the same extracted records, so upstream
  improvements flow in by construction. Generic pipeline-quality work becomes a spin-off map
  charted after the M6 gate, from the eval's measured weak spots.
  **CHARTED 2026-08-02**: [issues/wayfinder-datagen-quality/map.md](../wayfinder-datagen-quality/map.md)
  — 11 tickets, one per measured error pattern / carried finding, opening evidence =
  [M6-CALIBRATION-OFFICIAL.md](M6-CALIBRATION-OFFICIAL.md).

- Groundedness / contradiction handling — explicitly excluded by the spec for this iteration.
- True iterative retrieve-resolve-retrieve multi-hop (reasoning step in the loop) — spec 6.3:
  document the associative-diffusion limitation, don't fake it.
- Any LLM call in indexing or online retrieval — LLM appears only at final answer generation,
  outside this effort.
- Building or operating a retrieval engine — ADR-0048's stateless boundary stands; we upgrade the
  tooling around the adopter's engine.
