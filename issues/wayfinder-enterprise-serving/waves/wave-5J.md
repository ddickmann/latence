# Wave 5-J — the rescue pipeline gets one home: `retrieve()` + `select()`, and the bench copy dies

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 5-J). Three commits, in order:
the split (shipped, byte-identical), the A/B oracle (new, run on the full gold sets), the
deletion (bench). Nothing was deleted before its oracle was green.

The duplication: the measured S5b policy existed **twice** — `latence_retrieval/rescue.py` and
`benchmark/s5/rescue.py` — as ~600 near-line-identical lines (RELATION_CUES, the four gate
triggers + the navigational veto, `_append_and_dedupe`, `_unit_scaled`, `_graph_seeds`,
`_coe_select`, every constant). Wave 2-F had already closed the candidate-contract gap with the
`decorate=` seam. Exactly one interface gap was left: **the bench retrieves once per query and
selects at every `BUDGET_CURVE` point, while the shipped `retrieve()` fused retrieval and
selection.**

## 1. The split (commit `63c1993`, `packages/latence-retrieval/src/latence_retrieval/rescue.py`)

| | |
|---|---|
| `retrieve(bundle) -> RescueRetrieval` | the expensive half (two first stages, expander rescore, traversal, hypergraph propagation). The frozen handoff carries `base`, `union`, `gate`, `rescued`, `signals` — the shipped twin of the bench's `Retrieved`, including the `HypergraphSignals` the `coe` selector spends its budget on |
| `select(text, retrieval, *, selector=None, budget=None) -> RescueResult` | the cheap half, a pure function of that handoff. Overrides apply to the call only; an overriding selector falls back to **its own** measured default (10k packed / 16k coe); `coe` requires the hypergraph lane at select time exactly as at construction |
| `run(bundle) -> RescueResult` | the one-shot production call — literally `select(bundle.text, retrieve(bundle))` |

Both halves stateless (ADR-0048): the configured `Packer` is still built once at construction and
reused whenever the budget is the configured one; an overriding budget builds an
identically-configured packer for that call.

**Byte-identity, pinned not asserted.** Seven cases — all three selectors, default and explicit
budgets, a gate that fired and one vetoed by `mode-off`, and the decorated composition whose
precomputed `redundancy` rows ride the union across the split — were captured on the pre-split
code as canonical-JSON sha256 of the **whole `RescueResult`** and must be reproduced by *both*
`run(bundle)` and `select(text, retrieve(bundle))` (`TestTheSplit`, 14 parametrised assertions).
The pre-seam full-literal pin from Wave 2-F still passes unchanged. A counting-spy test proves
four budgets cost exactly one fetch per first stage.

Callers updated: `benchmark/serving/parity.py` (`retrieve` → `run`), the graph-rescue guide (new
*Budget A/Bs without re-retrieval* section), CHANGELOG.

## 2. The A/B oracle (commit `1f230f2`, `../tickets/05J/ab_rescue.py`)

Both pipelines over the **same** `benchmark.s5.legs.Runner` — same dense matrix, same BM25 index,
same `GraphRetriever`, same hyperedge store — so the only variable is which pipeline code drove
them. Per query, compared with `==` on full values (never on a recall summary):

- the whole **union**: `(id, score, metadata)` per candidate — the `rescued` stamp, the four
  denormalised KG columns **and** the precomputed `redundancy` map are inside the comparison;
- the **base** as `(id, score)`;
- every field of the **`GateDecision`**;
- the **hypergraph signals**: ρ (ids, order, floats), fields, anchors, vertex confidences, both
  stage counters;
- **selection for all three legs at every one of the seven `BUDGET_CURVE` points** (the
  selector-less ranking leg once): ranked ids, `tokens_packed`, `budget` — **15 selections per
  query per side**.

The shipped side was wired with `ExportDecorator` (the bench's `_kg_metadata` + `_decorate` behind
the Wave 2-F `CandidateDecorator` seam, delegating to `KgColumnDecorator` so the redundancy
numbers are the packer's own by construction) and `HyperedgeChunks`; every other knob was left at
its shipped default, because each of those defaults **is** the bench constant — which is itself
part of what the run proves.

The three hazards from the retrieval exploration were each made observable, not assumed away:
`_coe_select`'s budget threading (every curve point, both selectors), the module-level
`HeuristicTokenCounter` vs the bench's per-call one (`tokens_packed` compared, not recomputed),
and redundancy-metadata population (the union comparison carries the full map).

### Results — bench `RescuePipeline` vs shipped `GraphRescueRetrieval`

Query vectors from the existing `benchmark/serving/cache` checkpoints (no re-embedding);
`PYTHONHASHSEED=0`; full usable gold sets, no sampling.

| dataset | queries (full usable gold) | comparisons per query | diffs | wall clock |
|---|---|---|---|---|
| multihop_rag | **2,255** / 2,255 | 5 retrieval-level + 15 selections | **0** | 3,004 s |
| wiki2multihop | **12,576** / 12,576 (2 shards × 6,288) | 5 retrieval-level + 15 selections | **0** | 4,426 s + 4,435 s |
| vidoseek | **1,142** / 1,142 | 5 retrieval-level + 15 selections | **0** | 1,458 s |

**15,973 queries × 20 comparisons = 319,460 comparisons, zero diffs.** Per-run JSON in
`../tickets/05J/results/`. The `vidoseek` run was executed from a detached worktree at `1f230f2`
(both pipelines present) after the deletion had landed locally — same code on both sides of the
comparison as the other two datasets. `ohr_bench` was skipped: it is the OCR-noise control and
the two multi-hop sets plus vidoseek already exercise every lane the rescue family has.

**One honest asymmetry, measured rather than argued.** The base comparison is `(id, score)`, not
metadata: the shipped seam applies ONE decorator, so on a gate-fired query the base carries the
precomputed `redundancy` row ahead of the expander, where the bench computed it only for the
union. It is inert — the expander reads KG columns and the union decoration overwrites the key —
and the union comparison, which carries the full pairwise map, is what proves that across all
15,973 queries. The cost is real though: on a gate-fired query the redundancy map is now built
twice (base and union). Left as-is deliberately; a per-step decorator would be a second seam to
buy back an O(pool²) Jaccard the packer would otherwise recompute per budget anyway.

## 3. The deletion (commit `667d7dd`, `benchmark/s5/rescue.py`, `run_matrix.py`, s5 suite)

`benchmark/s5/rescue.py`: **814 → 260 lines.** What remains is exactly the four things a
*benchmark* adds on top of the component:

- **leg dispatch** — `LEG_SELECTORS`, `selector_for(leg)`, `headline_budget(leg)` (the "not a
  graph-rescue leg" refusal keeps its message);
- **the budget curve + its budgets** — `BUDGET_CURVE`, `SELECT_BUDGET`, `COE_SELECT_BUDGET`
  (which are also the shipped defaults) and, named only because ticket-01's latency harness
  reproduces the seed harvest outside the pipeline, `SEED_DOCS_K` / `SEED_NODE_CAP` — now pinned
  by a test against the constructed component's own defaults;
- **the gate report** — `GateStats` (unchanged, now fed the shipped `GateDecision`);
- **the caller's side of ADR-0048** — `ExportDecorator`, `HyperedgeChunks`, `build_pipeline`,
  `bundle_for`.

Deleted: `RescuePipeline`, `Retrieved`, `RescueOutcome`, `GateDecision`, `decide_gate`,
`top_k_agreement`, `_append_and_dedupe`, `_unit_scaled`, `_packer`, `_is_rescue`, `_token_set`,
`_jaccard`, `_tokens`, `RELATION_CUES`, `RESCUE_CAP`, `SELECT_POOL_K`, `RANK_SCORE_K`, the four
`EXPAND_*` weights, `PACK_CENTRALITY_WEIGHT`, `ENTITY_HEAVY_SEEDS`, `AGREEMENT_K`,
`LOW_AGREEMENT`, `HIGH_AGREEMENT`, `LOW_CONFIDENCE`, `NAVIGATIONAL_MAX_WORDS`.

`run_matrix._rescue_cell` now consumes the split API directly: one `retrieve` per query, a
`select` per curve point, latency measured as *the retrieval paid once + this budget's own
selection*.

**The cell itself is pinned too**, not just the pipeline: the three rescue cells were produced
twice on the real `multihop_rag` export (`--limit 8`, `PYTHONHASHSEED=0`) — once from a detached
worktree at `1f230f2` (the bench copy) and once from this branch — and the written JSONs are
**identical modulo the `latency_ms_*` keys** (which are wall clock and cannot repeat across runs):
same `metrics`, same seven-point `budget_curve`, same `gate` block, same `select_budget`, same
`tokens_packed_mean/max` (9,884.5 / 9,979 at the 10k knapsack; 15,921.6 / 15,999 at the 16k CoE).
That covers the cell wiring the per-query A/B does not touch.

### The s5 suite, repointed (496 → 561 lines, 18 tests)

- architecture invariants (base preserved under a graph lane that knows nothing, rescue appended
  and deduped, the selector's budget, the real-export end-to-end) now run the **shipped** pipeline
  over the same bench fixtures through a two-line `_Bench` wrapper (`build_pipeline` +
  `bundle_for`) — same claims, real engines;
- `_append_and_dedupe` / `_unit_scaled` / `top_k_agreement` imported from the shipped module (same
  functions, same assertions; `k` is now explicit);
- the four pure gate-policy tests were **deleted**, not lost: each has a 1:1 twin in
  `packages/latence-retrieval/tests/test_rescue.py::TestGateDecisions` (relation cue, known-item,
  navigational veto, disagreeing first stages). Duplicating them here would have pinned the same
  function twice through a private reach-through;
- `GateStats` is now fed `GateDecision` values directly — what is under test is the cell's report
  (hit rate, veto count, reason histogram), not the policy that produced the decisions;
- the redundancy pins run on `ExportDecorator` and the packer's own config.

**Two deliberate behaviour changes, neither reachable from a cell.** A typo'd `graph_mode` is now
refused at *construction* instead of at the first query. `graph_rescue_coe` with **no** hyperedge
store is now a loud refusal instead of silently returning the base — every rescue leg lists
`graph-hyperedges.kv` in its `required_files`, so a matrix cell can never be in that state, and
"CoE bought nothing" vs "CoE could not run" are different facts.

### Post-deletion re-run of the harness

With the bench copy gone, the harness pins the equation the deletion *relies* on, on the same
real data: the **split path** (one retrieval, then a selection at every operating point — exactly
what `_rescue_cell` does) against a **reference path** (a pipeline *configured* at that operating
point and run end-to-end for that query), at five points: both headlines, the ranking leg, and
the curve's extremes 2,000 / 32,000 — because a budget that is not the pipeline's configured
default is where a threading defect would hide.

| dataset | queries | diffs | wall clock |
|---|---|---|---|
| multihop_rag | 446 (a `--limit 500` sample) | **0** | 1,616 s |
| wiki2multihop | 300 (a `--limit 300` sample) | **0** | 628 s |

Samples, not the full sets, and stated as such: this run costs six full retrievals per query
(one split + five configured references) where the deletion gate above cost two, and the
full-set evidence is the pre-deletion table.

## Gates

| gate | result |
|---|---|
| `packages/latence-retrieval/tests` | **820 passed, 4 skipped** (796/4 before Wave 5-J; +24 = `TestTheSplit`) |
| `benchmark/s5/tests` + `benchmark/serving` | **266 passed** |
| `packages/latence-benchmark/tests` + publish-readiness + public-release gate | **537 passed, 3 skipped** |
| `ruff check packages benchmark/s5 benchmark/serving issues/wayfinder-enterprise-serving` | clean |
| `mypy --strict packages/latence-retrieval/src` | clean (31 files) |
| Apache seven-file pin | unchanged (`test_publish_readiness` green) |
| pre-split byte-identity | 7 cases × 2 entry points, canonical-JSON sha256, green |
| A/B oracle | 296,620 comparisons over 14,831 real queries, **0 diffs** |
| cell-level oracle | 3 rescue cells, pre vs post, identical modulo `latency_ms_*` |

## Provenance note for the in-flight pod sweep

The S5b sweep running on the pod is pinned to its own checkout and is **not** affected by this
branch. Its `graph_rescue*` cells were produced by the **pre-split** code (at or before
`d1631b6`); every cell produced from `667d7dd` onwards comes from the shipped component through
`build_pipeline`. The A/B above is the evidence that the two produce identical rankings,
selections and gate decisions on the full gold sets — 319,460 comparisons, 0 diffs — so cells
from either side of the commit range are comparable. The provenance line is recorded here so
nobody has to infer it later, and the commits were pushed rather than held: the pod's checkout is
pinned, so pushing cannot disturb the sweep in flight.
