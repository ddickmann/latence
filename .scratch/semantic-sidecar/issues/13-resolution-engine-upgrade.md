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

**Answered in the spike the same day** by a scorer fitted on reviewer labels
(`spike/pairfeat.py`, `spike/learned.py`, `run_a.py --learned`): the labels say what each
piece of evidence is worth in the estate at hand, so none of the rules above is written by
hand. Same 5,000 labels, no GPU: 97% recall at 99.7% precision on the registers, more recall
than the best model arm on both synthetic estates, and the same recall at higher precision on
the person records. What this hands to the upgrade:

- **The first stage stays, as one piece of evidence and as the blocker.** Its probability is
  a feature of the fitted scorer; it no longer sets the lines alone.
- **Training labels are spread over patterns of evidence;** test labels are drawn on each
  side of even odds. Uniform labels are not used.
- **Lines are tried at quantiles of the scores,** the first accept step on a fifth of the
  test labels, and no pair is accepted that the scorer rates below even odds.
- **Scores are calibrated across folds,** because the cluster check reads them as
  probabilities.
- **Not carried over:** rank-among-candidates evidence (it cost precision where an
  organisation has several records).
- **Open for this decision:** evidence per kind of record is hand-written (organisations,
  persons); whether a neural model still earns its place on the small review band that is
  left; the evidence step runs in one thread.

**Decided by the owner (2026-11-11): the scorer is fitted per domain.** A scorer carried from
one estate to another finds 26% to 52% of the links where the estate's own labels find 87% to
97% ([report](../spike/REPORT.md#how-it-carries-to-new-data)). Fitting it is deployment work.
What that asks of the engine:

- **Labels are an input of every deployment.** About 600 give the automatic merges on clean
  names, about 2,500 on noisy ones; a certified reject line needs 300 more on its side. The
  number does not grow with the estate.
- **The labels have to come from the review screen.** Today they are simulated from a known
  truth. Reading a reviewer's answers back as labels is not built.
- **A source can be declared free of duplicates** (a master list with a key): its own pairs
  are ruled out, no cluster holds two of its records, and the scorer sees how clearly a
  record fits one master record. Needed as soon as two masters carry the same name.
- **Linking to a master list is its own mode:** only pairs between the sources are compared.
- **Blocking passes depend on the shape of the names;** a pass over a shared word in the same
  town was needed for names that reuse a few words. The passes should be chosen per estate
  from what they propose, not fixed.
- **An absent field must not stop a run** (the first stage stopped without postcodes).
