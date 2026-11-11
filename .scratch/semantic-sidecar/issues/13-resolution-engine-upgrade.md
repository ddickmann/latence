# Grilling: the upgrade of the resolution engine

Type: grilling
Mode: HITL
Status: open
Blocked by:

## Question

Decide the resolution engine for both modalities:

- a learned or Fellegi-Sunter matcher versus the string cascade;
- calibration and precision-targeted thresholds;
- an LLM in the uncertain band, and its egress rules;
- persisted must-link and cannot-link constraints from human decisions, replayed every run
  and under deltas (ADR-0018/0029/0061);
- an actor on every decision (`MergeDecision` has none today);
- splitting homonyms;
- scale: in-memory versus Spark;
- what happens to `disambiguation.cascade`, `.embedding` and `.glinker`, and to
  `type_consolidation`.

Start from the candidate design in [design/resolution-at-scale.md](../design/resolution-at-scale.md) and the
feasibility spike's numbers.

## Input from the feasibility spike (2026-11-11)

The spike reported go ([its answer](20-resolution-feasibility-spike.md#answer), numbers in
[`../spike/REPORT.md`](../spike/REPORT.md)). What it hands to this decision:

- **Matcher:** Fellegi-Sunter (Splink on DuckDB) with isotonic calibration and Learn-then-Test
  bands clears 0.95 precision on every estate measured; the string cascade reached pairwise F1
  0.540 where the matcher reached 0.913.
- **Second stage:** GLiDE 0.4B over the undecided pairs cuts the review band by 56% at 10⁵
  records; the reranker adds a further 14%. Zero-shot Decide adds nothing.
- **Labels:** uniform labels over candidate pairs do not scale to 10⁶; labels drawn from the
  undecided pairs do (7,500 labels did what 50,000 uniform ones did).
- **The reject line loses recall:** about a quarter of the true pairs at 10⁶ are below it.
- **Scale in memory:** 10⁶ records run in 92 s at 5.2 GB; Spark is unmeasured.
- **Budget:** 4 hours for 10⁶ records on one 24 GB GPU (owner).

**From two real company registers** (same day, after the go): the matcher ranks true links
poorly there (AUC 0.938) although 95.8% of them carry the same name. To decide here: how an
exact name weighs against a differing address; a veto for names one short token apart (fund
series, numbered partnerships); a per-source rule that a register holds no duplicates; and
score-aware labelling, since 2,000 uniform labels held 26 true links. With these open, the
model alone lifts the machine from nothing to 36.5% recall at 0.995 precision.
