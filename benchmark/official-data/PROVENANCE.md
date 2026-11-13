# benchmark/official-data — provenance

What lives here, and where each artifact came from. Anything LLM-generated is labeled as
such at the point of use, never silently mixed with measured or human-authored data.

**These files are dated records.** The three helper scripts of this folder were removed on
2026-11-10, together with the retrieval layer
(`docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md`).
None of them could run: `make_calibration_sample.py` read `graph-hyperedges.parquet`, a file
the current pipeline does not write, and the other two read the sample file it produced.
Neither input is tracked. Git history has the scripts. What remains under `musique-full/` is
the judgment file described below, a per-regime calibration map
(`calibration-map-per-regime.json`) and the dev-split scores with and without that map
(`calib-ab-dev.json`), written by `benchmark/official/calib_ab.py` before it was removed.

## Calibration judgments (official)

- **File:** `musique-full/calibration-judgments.jsonl` (250 lines, ticket-21 harness format:
  `{"hyperedge_id", "judgment", "note"?}`, sorted keys).
- **Provenance: LLM-generated — Claude Opus, 2026-08-02. NOT human labels. Pending human
  spot-check.** An isotonic curve fitted from these inherits whatever the model got wrong.
- **Sample:** deterministic stratified draw over the OFFICIAL MuSiQue-full export
  (`musique-full/export/graph-hyperedges.parquet`, 60,054 Hyperedges). All Hyperedges are
  sorted by `(gamma, hyperedge_id)` and every `60054 // 250 = 240`-th row is taken
  (indices 0, 240, 480 … 59,760). No cherry-picking and no random seed. The removed
  `make_calibration_sample.py` drew it; the export it read is not in the tree.
  γ spans 0.2500 – 0.9325 across the sample.
- **Judging basis:** each Hyperedge is judged against **its own `tau_text`** — the paragraph
  span IS the evidence (this corpus has no planted-fact chain file). `entity_node_ids` are
  joined to canonical names and types via `graph-nodes.parquet`, and `relations` to
  head/label/tail triples. Sibling Hyperedges over the same chunk are shown to the judge so
  that an arity-split singleton is not mistaken for a recall miss.
- **Rubric** (`correct` / `partial` / `incorrect`) and the two structural conventions it
  rests on are reproduced under *Rubric* below, from the removed generator
  `make_calibration_judgments.py`. The generator held the label and its reason per row and
  refused to emit a non-`correct` label without a note; its table matched the committed
  judgment file row for row when it was removed.
- **Distribution:** 135 correct (54.0%) / 80 partial (32.0%) / 35 incorrect (14.0%).
- **Analysis:** the γ-quintile and structural breakdown was produced by the removed
  `analyze_calibration.py`. The write-up it fed is not in this repository.

### Rubric

Copied from the docstring of the removed `make_calibration_judgments.py`. The `tau_text` is
the evidence; this corpus has no gold chain file.

* `correct`: every member is an entity the span genuinely mentions and reasonably typed, and
  every relation triple is asserted by the span in the right direction.
* `partial`: the core extraction is right but one element of several is wrong (a bogus or
  mis-typed member among good ones, one unasserted relation among asserted ones), or a
  *central* entity of the span is missing altogether.
* `incorrect`: nothing survives. The sole member is not an entity or is a broken span, or
  every relation the Hyperedge asserts is unasserted or false.

Two structural facts of the export shaped the judgments:

1. **A span with more than `max_arity` (12) entities and no relations was split into
   singleton Hyperedges.** So an arity-1 Hyperedge over a 15-name cast table is a deliberate
   split, not a recall miss; its siblings carry the rest. "Materially incomplete" was charged
   only when an entity central to the span was missing from the whole chunk.
2. **Redundant relation labels over one asserted pair were not counted as errors** when both
   labels are semantically compatible (`created_by` + `author_of`, `located_in` + `part_of`).
   A label was charged only when it asserts something the span does not: `capital_of`,
   `spouse_of`, `employer` for non-employment, `founded_by` for non-founding, `country_of`
   onto a demonym.
