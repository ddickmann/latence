# Research: the state of the art in entity resolution with calibration, constraints and scale

Type: research
Mode: AFK
Status: resolved
Blocked by: 

## Question

What should the resolution engine become? Today it is a 5- or 6-step string cascade with
Union-Find, hand-set thresholds (0.84 fuzzy, 0.93 embedding), and an audit log of merges.

Cover:

- learned pairwise matchers and cross-encoders;
- LLMs used only in the uncertain band;
- probability calibration and precision-targeted thresholds;
- Fellegi-Sunter and Splink, including running on Spark/Databricks;
- clustering under must-link and cannot-link constraints (correlation clustering,
  constrained Union-Find);
- active learning from reviewer decisions;
- splitting homonyms;
- incremental and streaming ER that stays consistent under deltas;
- blocking at warehouse scale;
- ER evaluation (pairwise, B-cubed, cluster metrics).

Also note what ADR-0046 learned: retrieval embedders over-merge short names, and no clean
threshold exists.

Resolves with a report at `research/04-sota-entity-resolution.md`.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/04-sota-entity-resolution.md](../research/04-sota-entity-resolution.md). arXiv, ACL Anthology, Hugging Face and the Splink docs were blocked, so paper claims come from abstracts and snippets, and unchecked figures are marked UNVERIFIED.

- **Current limits.**
  - Mentions are grouped by normalised surface before any matching, so a homonym can never be split.
  - Thresholds are hand-set and uncalibrated, and the first rule that fires decides.
  - Union-Find merges transitively with no check.
  - There is no cannot-link and no store of reviewer decisions.
  - Canonical ids drift when a new surface joins a cluster.
- **Embeddings retrieve; they don't decide.** This matches ADR-0046's over-merge finding. A controlled 2026 study (1,215 runs) found cross-encoders beat bi-encoders. Embeddings only generate candidate pairs.
- **Calibrated decisions.**
  - Replace the fixed 0.84 and 0.93 with calibrated probabilities (Platt or isotonic).
  - Choose thresholds with Learn-then-Test, which guarantees a target precision.
  - This gives three bands: accept, review, reject. A link with too few labels stays unknown.
- **LLMs only in the review band.** They use select-from-a-shortlist (ComEM; CaRL-EM, ACL 2026). Answers are cached and audited, and calls are gated by the egress decision.
- **Splink** (MIT) fits warehouse metadata: it trains without labels and runs on Spark. 5.0.0 (2026-09-28) adds incremental scoring. No cannot-link support in its clustering was found, and no Databricks-specific docs.
- **Clustering.**
  - Human must-links are merged first.
  - Cannot-links and the existing type and distinguishing-token guards act as hard vetoes.
  - Every merge between two clusters is re-checked, and one strong negative blocks it.
  - Correlation clustering (Pivot) is optional.
- **Human decisions** are stored on stable member keys and replayed first on every run. They also serve as active-learning labels. Homonym splits are proposed, never applied silently.
- **Deltas.** Keep ADR-0061 (a delta equals a full rebuild). Add a pair-score cache, re-cluster only touched components, and assign stable ids by overlap with the previous run.
- **Blocking and evaluation.** Use several blocking passes (token keys, TF-IDF, embedding search, warehouse keys) and report coverage for each. Report pairwise, B-cubed and GMD/VI metrics; core has no ER metric today.
- **Licences.** Zingg (AGPL), er-evaluation (AGPL, reported) and LinkTransformer (GPL) are reference only. dedupe (MIT) is a reference design.
- **For other tickets:**
  - the granularity of text resolution, Splink as a dependency or in core, and the cross-encoder base and its labels (*the upgrade of the resolution engine*);
  - the label count required before automatic acceptance (*the review, certification and versioning lifecycle*);
  - LLM egress scope (*the text-path upgrade*, *the upgrade of the resolution engine*);
  - linking text to UC objects (*the boundary with Metis*, *the warehouse pipeline*);
  - byte-identical replay with a GPU scorer (*the destination and the domain model*).
