# W2-linkpred (Slice 1/2) — Graph-Completion Capability seam + reference completer + ADR

**The post-Assembly link-prediction subsystem, seam first (learned ULTRA provider is Slice 2).** Adds
an OPTIONAL `graph_completion` Stage after Graph Assembly that predicts missing KG edges. This slice
establishes the Capability, the runner wiring, the inferred-edge contract, an ADR, and a **reference
pure-Python completer** (no heavy dep) that proves the seam — exactly how GraphAssembler shipped a
reference provider before the learned ones. Slice 2 (`latence-linkpred-ultra`) adds the real ULTRA +
PyKEEN scorer behind this seam.

## New Capability: `GraphCompleter` (capability.py)
Mirror `GraphAssembler` (corpus-level, deterministic, `Iterator` in/out):
```python
class GraphCompleter(Protocol):
    def complete(self, records: Iterable[GraphRecord]) -> Iterator[GraphRecord]:
        """Consume the assembled graph (NODE + EDGE GraphRecords) and yield ADDITIONAL
        EDGE-scope GraphRecords for PREDICTED edges — never re-emitting the input, never
        mutating asserted edges. Deterministic (predicted edges in edge-id order)."""
```
- Corpus-level: consumes the whole run's `GraphRecord`s (nodes give the entity vocab + types +
  source docs; edges give the observed relations). Emits ONLY new predicted EDGE records.
- Deterministic + seeded → byte-identical (Baseline bar): predicted edges sorted by edge-id.

## Inferred-edge contract (the enterprise-honesty crux — never silently merged)
A predicted edge is a `GraphEdge` at `scope=EDGE`, but UNMISTAKABLY marked inferred:
- `properties["inferred"] = True` (REQUIRED on every predicted edge), plus
  `properties["scorer"]` (e.g. `"reference.adjacency"` here, `"ultra"` in Slice 2),
  `properties["score"]` (raw model score), `properties["rank"]`, `properties["calibrated"]` (bool).
- `edge_id` = deterministic sha256 over `(corpus_id, source_node_id, label, target_node_id,
  "predicted")` — distinct from any asserted edge; endpoints must be distinct (contract already
  enforces). `relation_id` = `f"predicted:{scorer}:{edge_id}"` (satisfies min_length; clearly NOT a
  source `NormalizedRelation`).
- **Evidence is honest and model-derived, NOT fabricated mention offsets:** `mention_ids = []`
  (there is no textual mention — asserting one would be a lie), `document_ids` = union of the head +
  tail nodes' `source_document_ids`, `snippet` = a human justification
  (`f"Predicted edge (not asserted in text); {scorer} score {s:.3f}"`), `confidence` = the calibrated
  score. A predicted edge that collides (same endpoints+label) with an asserted edge is DROPPED (it
  is already asserted) — logged, not emitted.
- **Never merged into the asserted stream:** the completer emits its own records; a consumer/exporter
  filters `properties["inferred"]`. The Quality Report counts asserted vs inferred edges SEPARATELY
  (add `inferred_edge_count` alongside `edge_count`); an inferred edge must never inflate the asserted
  edge count. (Keep `scope=EDGE` — a new GraphScope would ripple through export/delta/validate; the
  `inferred` marker + separate count is the minimal honest representation. Justify this in the ADR.)

## Runner wiring (optional stage, post-Assembly)
- Add a `graph_completion` capability branch in `_dispatch_stage` (mirror the `graph_assembly`
  branch ~runner.py:1238): isinstance-check `GraphCompleter`, gather the run's `GraphRecord`s (same
  streaming/`_require_*` discipline as assembly consumes DisambiguationRecords), call
  `provider.complete(graph_records)`, and thread the emitted predicted edges downstream to Export
  (they flow into the KG export alongside asserted edges, distinguished by `properties.inferred`).
- The stage is OPTIONAL: a stack with no `graph_completion` stage behaves exactly as today (prove
  with a test). A run WITH it appends predicted edges. It depends_on the graph-assembly stage.

## Reference completer (this slice — pure-Python, no heavy dep, in latence-core)
`graph_completion.reference` (an `AdapterBase`/registered provider, `compute="cpu"`): a deterministic,
explainable heuristic so the seam is exercised end-to-end without torch. Suggested rule:
**2-hop symmetric/transitive closure over observed relations** — e.g. if `A partner_of B` and
`B partner_of C` observed, propose `A partner_of C` (configurable per-relation transitivity/symmetry);
score = a simple path-support fraction; only propose above a config threshold; cap fan-out (bounded,
like the relation quadratic guard). This is NOT ULTRA — it's the honest reference that proves the
contract + wiring + determinism, and gives Slice 2 a drop-in target to beat in the bake-off.

## Definition of Done
- `GraphCompleter` Protocol in capability.py; `graph_completion.reference` registered, AdapterBase,
  deterministic, emits inferred-marked predicted EDGE records with honest Evidence; collisions with
  asserted edges dropped; fan-out bounded.
- Runner `graph_completion` stage wired, OPTIONAL (a stack without it is unchanged — tested); predicted
  edges reach Export marked `inferred`; Quality Report gains a SEPARATE `inferred_edge_count`.
- **ADR-00XX** (next number): the Graph-Completion Capability, the post-Assembly optional-stage
  placement, and the inferred-edge representation decision (EDGE scope + `inferred` marker + separate
  count + model-derived Evidence with empty mention_ids; why not a new GraphScope). Note Slice 2 =
  learned ULTRA/PyKEEN scorer + calibrator + MRR/Hits@k, MIT-licensed (ULTRA MIT code+weights,
  PyKEEN MIT — both verified 2026-07-09).
- Conformance: the reference completer passes the applicable conformance cases (roster +1); a stack
  with the optional stage passes `latence stack validate`. `bash scripts/verify-local.sh` green.
- Determinism byte-identical over two runs; no asserted-edge count inflated by inferred edges (tested).

## Reviewer should check
- Inferred edges are UNMISTAKABLE and never merged into asserted edges (separate count; `inferred`
  marker; honest Evidence with empty mention_ids — no fabricated offsets). This is the enterprise crux.
- The `graph_completion` stage is genuinely OPTIONAL (a stack without it is byte-identical to today).
- Determinism (predicted edges in edge-id order; seeded); bounded fan-out; collision-drop logged.
- The reference completer is a real, explainable heuristic (not a stub) that exercises the whole seam,
  giving Slice 2's ULTRA provider a drop-in bake-off baseline.
