# benchmark/official-data — provenance

What lives here, and where each artifact came from. Anything LLM-generated is labeled as
such at the point of use, never silently mixed with measured or human-authored data.

## Calibration judgments (official)

- **File:** `musique-full/calibration-judgments.jsonl` (250 lines, ticket-21 harness format:
  `{"hyperedge_id", "judgment", "note"?}`, sorted keys).
- **Provenance: LLM-generated — Claude Opus, 2026-08-02. NOT human labels. Pending human
  spot-check.** An isotonic curve fitted from these inherits whatever the model got wrong.
- **Sample:** deterministic stratified draw over the OFFICIAL MuSiQue-full export
  (`musique-full/export/graph-hyperedges.parquet`, 60,054 Hyperedges). All Hyperedges are
  sorted by `(gamma, hyperedge_id)` and every `60054 // 250 = 240`-th row is taken
  (indices 0, 240, 480 … 59,760). No cherry-picking, no random seed, byte-reproducible:
  `python3 benchmark/official-data/make_calibration_sample.py musique-full 250`.
  γ spans 0.2500 – 0.9325 across the sample.
- **Judging basis:** each Hyperedge is judged against **its own `tau_text`** — the paragraph
  span IS the evidence (this corpus has no planted-fact chain file). `entity_node_ids` are
  joined to canonical names and types via `graph-nodes.parquet`, and `relations` to
  head/label/tail triples. Sibling Hyperedges over the same chunk are shown to the judge so
  that an arity-split singleton is not mistaken for a recall miss.
- **Rubric** (`correct` / `partial` / `incorrect`) and the two structural conventions it
  rests on are documented in the generator, `make_calibration_judgments.py`, which carries
  the label AND its reason per row and refuses to emit a non-`correct` label without a note.
- **Distribution:** 135 correct (54.0%) / 80 partial (32.0%) / 35 incorrect (14.0%).
- **Analysis:** `issues/wayfinder-hypergraph-coe/M6-CALIBRATION-OFFICIAL.md`
  (γ-quintile trend, failure patterns). Reproduce with
  `python3 benchmark/official-data/analyze_calibration.py musique-full`.

### Helper scripts (this directory, no new dependencies — `pyarrow` only)

| script | what it does |
|---|---|
| `make_calibration_sample.py` | draws the deterministic stratified sample → `calibration-sample.jsonl` |
| `make_calibration_judgments.py` | holds the judgment table (label + reason) → `calibration-judgments.jsonl` |
| `analyze_calibration.py` | γ-quintile and structural breakdown of the judgments |
