# Research: the state of the art in entity resolution with calibration, constraints and scale

Type: research
Mode: AFK
Status: open
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
