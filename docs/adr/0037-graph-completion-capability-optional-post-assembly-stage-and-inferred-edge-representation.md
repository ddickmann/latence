# Graph Completion is an OPTIONAL post-Assembly Capability; a predicted edge is an inferred-marked EDGE-scope GraphRecord counted separately from asserted edges

**Status: accepted** — W2-linkpred Slice 1/2 (seam) + **Slice 2/2 (learned scorer — BUILT**, see the
closing note). Establishes the link-prediction seam, the runner wiring, the inferred-edge contract,
and a pure-Python **reference completer**; the learned ULTRA/PyKEEN scorer (Slice 2) is now built
behind this same seam as `graph_completion.ultra` (`latence-linkpred-ultra`). Builds on ADR-0004 (Capability protocols +
Provider plugins), ADR-0007 (CPU-first reference providers), ADR-0016 (thin core, heavy deps
isolated), ADR-0017 (Export writes files, never a live DB), ADR-0027 (Graph Assembly's scoped
GraphRecord + the `is_predicted` predicted-vs-extracted split), ADR-0033 (streaming, memory-bounded
Runner), ADR-0036 (Provider ecosystem: profile, conformance, bake-off).

## Context

ADR-0005 deferred **link prediction** from v1 (sequencing + dependency surface, not asset quality);
the preserved **Link prediction stack**
(`link_prediction/{scorer,ultra_scorer,candidate_generator,calibrator}.py`, rated 5/5 in the
asset review and taken as-is) is:
**ULTRA** (ICLR'24, inductive/zero-shot) + PyKEEN behind one scorer abstraction, type-aware tiered
2-hop candidate generation with a recall guarantee, temperature scaling + precision-targeted
thresholds. ADR-0027 already anticipated the split: every S9 edge carries `is_predicted=False`, "so
a future predicted edge is never silently mixed with an extracted one".

The crux is **enterprise honesty**: a predicted edge is a *model belief*, not something asserted in
a source document. Merging predicted edges into the asserted knowledge graph — or fabricating source
mentions/offsets for them — would let a hallucinated relation masquerade as extracted-from-text
fact. That is exactly the failure this framework exists to prevent. So the representation must make a
predicted edge **unmistakable**, keep it **out of the asserted stream**, and count it
**separately**.

## Decisions

### 1. A new `GraphCompleter` Capability — an OPTIONAL corpus-level Stage after Graph Assembly

Link prediction is its own Capability (ADR-0004), mirroring `GraphAssembler`: corpus-level,
`Iterable[GraphRecord]` in / `Iterator[GraphRecord]` out, deterministic. It **consumes the assembled
graph** (NODE + EDGE GraphRecords — nodes give the entity vocab + types + source docs, edges give
the observed relations) and yields **only ADDITIONAL EDGE-scope GraphRecords for predicted edges**.
It never re-emits the input and never mutates an asserted edge. It `depends_on` the Graph Assembly
Stage and is **genuinely OPTIONAL** — a stack with no `graph_completion` Stage is byte-identical to
today (proven by an e2e test that diffs the graph_assembly checkpoint + the report's asserted-edge
count with and without the Stage). This places prediction *after* the honest extracted graph is
built, so the asserted graph never depends on the predictor.

### 2. A predicted edge is an EDGE-scope GraphRecord marked `inferred` — NOT a new GraphScope

We reuse `scope=EDGE` (not a new `GraphScope.PREDICTED`) and mark the edge
`properties["inferred"] = True`. **Rationale:** a new GraphScope would ripple through every consumer
of the scoped carrier — Export (Parquet/TTL/GraphML writers), the delta engine's affected-set +
GraphPatcher, `stack validate`, the Quality Report — for no representational gain. The `inferred`
marker plus a **separate count** is the minimal honest representation: a predicted edge IS a graph
edge (same endpoints-are-nodes contract, same Evidence field), it is simply *believed* rather than
*asserted*. Every predicted edge additionally carries `properties["scorer"]`
(`"reference.adjacency"` here, `"ultra"` in Slice 2), `properties["score"]` (raw model score),
`properties["rank"]`, and `properties["calibrated"]` (bool — `False` for the reference's raw
path-support fraction; Slice 2's temperature-scaled ULTRA sets it `True`). This is the ADR-0027
`is_predicted` split promoted from an attribute to its own Stage's output.

Deterministic ids keep it distinct from any asserted edge: `edge_id = sha256(corpus_id,
source_node_id, label, target_node_id, "predicted")` (the `"predicted"` discriminator guarantees it
never collides with an asserted `edge_id`, which mixes `"edge" + relation_id`); `relation_id =
f"predicted:{scorer}:{edge_id}"` — clearly NOT a source `NormalizedRelation`, and long enough to
satisfy the contract's `min_length`.

### 3. Evidence is honest and model-derived — NOT fabricated mention offsets

CONTEXT separates **Provenance** (source lineage) from **Evidence** (justification for an
inference). A predicted edge has **no textual mention** — asserting one would be a lie. So its
Evidence carries `mention_ids = []` (empty, deliberately), `document_ids` = the **union of the head
+ tail nodes' `source_document_ids`** (the real documents the endpoints came from — honest, not
invented), `snippet` = a human justification (`"Predicted edge (not asserted in text);
{scorer} score {s:.3f} ..."`), and `confidence` = the calibrated score. The Quality Report's
`evidence_coverage` (fraction of edges whose Evidence names a mention) is computed over **asserted
edges only**, so an inferred edge with empty `mention_ids` cannot drag it down.

### 4. Collisions with asserted edges are DROPPED, not emitted

A predicted edge whose `(source, label, target)` matches an **asserted** edge is already asserted —
it is dropped (logged, not emitted). The reference engine never proposes an observed pair in the
first place; the Provider double-guards against the asserted-edge set. This keeps prediction strictly
*additive of new knowledge*.

### 5. The Quality Report counts inferred edges SEPARATELY — never merged into asserted counts

`GraphQuality` gains `inferred_edge_count` (+ `inferred_edges_by_scorer`) alongside `edge_count`.
The report gathers asserted GraphRecords from the **Graph Assembly** Stage and predicted GraphRecords
from the **Graph Completion** Stage as two distinct streams, so `edges_total` / `edges_by_label` /
`evidence_coverage` count asserted edges only and an inferred edge can never inflate them. A run
without a Graph Completion Stage has `inferred_edge_count == 0`. This is the machine-checkable
statement of the honesty crux.

### 6. Reference completer (this slice): a deterministic, bounded 2-hop closure — no torch

Exactly as Graph Assembly shipped a pure reference builder before any learned Provider, this slice
ships `graph_completion.reference` in `latence-core` (ADR-0007 CPU-first, ADR-0016 no extra deps):
a deterministic, explainable **2-hop symmetric/transitive closure** over the observed relations —
if `A partner_of B` and `B partner_of C` are observed and `partner_of` is transitive, propose
`A partner_of C`; if `p` is symmetric, propose the reverse. The score is a **path-support fraction**
(distinct 2-hop paths / normalisation, clamped `[0,1]`); only proposals at or above a config
threshold are kept; **fan-out is bounded** per source node (the ported relation quadratic-guard
discipline) so a hub node cannot explode the candidate set. It is NOT ULTRA — it is the honest
reference that proves the contract + wiring + determinism and gives Slice 2's learned scorer a
drop-in bake-off baseline to beat. Byte-identical across runs (predicted edges yielded in edge-id
order).

### 7. Slice 2 (NOT this slice): the learned ULTRA/PyKEEN scorer behind the same seam

Slice 2 (`latence-linkpred-ultra`, an isolated package per ADR-0016) ports that preserved link
prediction stack behind this same `GraphCompleter` seam: **ULTRA** (inductive/zero-shot) + PyKEEN
scorer, the type-aware tiered 2-hop candidate generator (recall guarantee), the temperature-scaling
calibrator (`calibrated=True`), and MRR/Hits@k evaluation surfaced in the bake-off. **Licenses,
verified 2026-07-09 (ADR-0012, weights + code checked SEPARATELY):** ULTRA **code** is **MIT**
(`github.com/DeepGraphLearning/ULTRA/blob/main/LICENSE`); ULTRA **released weights** are **MIT**
(HF `mgalkin/ultra_50g` / `ultra_4g` model cards, `license: mit`); **PyKEEN** is **MIT**
(`github.com/pykeen/pykeen/blob/master/LICENSE`). All three permissive — no restricted-license
opt-in needed. Slice 2 records the citation + verification date on its ProviderProfile.

## Consequences

- **Positive:** a predicted edge is unmistakable (`inferred` marker), never merged into the asserted
  graph, honestly evidenced (empty `mention_ids`, no fabricated offsets), and counted separately —
  the enterprise-honesty crux is machine-checkable. The seam is proven end to end offline by a
  pure-Python reference, and Slice 2's learned scorer drops in with zero contract change. The
  optional Stage keeps every existing stack byte-identical.
- **Negative / cost:** reusing `scope=EDGE` means every EDGE consumer must respect
  `properties["inferred"]` to keep the two classes apart (Export ships both, distinguished only by
  the marker; a downstream loader that wants asserted-only filters on it). We accept this over a new
  GraphScope's ripple. The reference completer's recall is intentionally weak (a 2-hop heuristic, not
  a learned scorer) — it exists to prove the seam, and Slice 2 is where recall/precision are
  earned and benchmarked.
- **Reviewer checkpoints (met):** inferred edges are UNMISTAKABLE and never merged (separate count +
  `inferred` marker + honest empty-mention Evidence); the Stage is genuinely OPTIONAL (byte-identical
  without it); determinism byte-identical over two runs, bounded fan-out, collision-drop logged; the
  reference completer is a real explainable heuristic, not a stub.

## Slice 2 landed — the learned ULTRA/PyKEEN scorer (W2-linkpred Slice 2/2)

Decision 7 is now BUILT in the isolated `latence-linkpred-ultra` package (ADR-0016), registered as
`graph_completion.ultra` on this exact seam — a drop-in for `graph_completion.reference` with the
inferred-edge contract reused verbatim (`properties.inferred`, `scorer="ultra"`, calibrated
`score`, `rank`, honest empty-mention Evidence, deterministic `edge_id`, asserted-collision drop,
separate `inferred_edge_count`). What Slice 2 adds behind the seam:

- **Scorer abstraction** — an inductive **ULTRA** scorer (ICLR'24, zero-shot from a pretrained
  checkpoint; default `ultra_3g`, a config path — **weights NOT vendored**) with a **PyKEEN**
  transductive-KGE **graceful fallback**; if NEITHER is available a **typed `ProviderError`**, never
  a silent no-op (a stack may opt a trailing stage into `skip_if_unavailable` skip-with-flag).
- **Type-aware tiered 2-hop candidate generation** — bounded per-relation fan-out with a **rare-
  relation full-recall guarantee**; every cap that drops candidates is **logged** (no silent
  truncation).
- **Calibrator** — temperature scaling + a **precision-targeted threshold**: only edges whose
  *calibrated probability* clears a configurable target precision are emitted, and the emitted
  `confidence` IS that calibrated probability (`calibrated=True`) — a raw scorer logit is never
  emitted as confidence.
- **KG-quality metrics** — held-out **MRR / Hits@k** with genuine **leak detection** (the held-out
  edges *and their inverses* are removed from the observed graph before scoring; a leak raises
  rather than fabricating a metric).

**Anti-false-green (non-negotiable):** the offline provider is FAITHFUL to the REAL ULTRA API
(`Ultra(rel_model_cfg, entity_model_cfg)`; `model(data, batch) -> [B, num_nodes]`; the `tasks`
helpers `build_relation_graph` / `all_negative` / `compute_ranking` at their real shapes). An
un-faithful shim FAILS the anchor tests: `tests/test_ultra_scorer_faithful.py` (guard #1 — a
torch-FREE AST anchor over the vendored `models.py` / `tasks.py` API surface, so it runs in the
CPU/offline merge gate and turns RED on any vendored-API drift) and `tests/test_ultra_scorer_drive.py`
(guard #2 — drives `UltraScorer` end-to-end against a shape-faithful fake wherever torch is
installed). The heavy deps (torch / torch-geometric / torch-scatter / easydict / pykeen) are isolated
and lazily imported so provider discovery stays torch-free; the real MRR/Hits@k are re-validated on
the CUDA pod (`scripts/validate-linkpred-on-gpu.md`), never fabricated. **Licenses (ADR-0012,
verified 2026-07-09, weights AND code separately):** ULTRA code MIT, ULTRA weights MIT, PyKEEN MIT —
all permissive; the vendored ULTRA inference subset is attributed in the package NOTICE /
THIRD-PARTY-LICENSES. Wired into `stacks/gpu-sota.yaml` + `stacks/gpu-sota-pod.yaml` as an OPTIONAL
trailing Stage and `matrix/graph_completion.yaml` (reference vs ultra bake-off).
