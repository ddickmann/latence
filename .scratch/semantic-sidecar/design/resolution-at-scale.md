# Design note: entity and relation resolution at corpus and Unity Catalog scale

Status: **candidate design for ticket 13** (the resolution engine grilling). It becomes an ADR only
after that grilling and the feasibility spike (ticket 20) report. Written 2026-11-08.

Sources: [research 03](../research/03-sota-schema-matching.md),
[research 04](../research/04-sota-entity-resolution.md),
[research 05](../research/05-ontology-learning-and-standards.md). Relation resolution is detailed
in [research 21](../research/21-relation-resolution-at-scale.md). Claims the reports mark UNVERIFIED stay UNVERIFIED here.

## Why this comes first

Entity and relation resolution is the hard problem of every ontology and knowledge graph,
whether on Databricks or in a text pipeline. If it fails, everything downstream inherits its
errors:

- concepts fragment into duplicates;
- distinct entities collapse into one;
- edges attach to the wrong node;
- the ontology is induced from noise.

So resolution is the critical path. **No contract or harvester build starts before the
feasibility spike (ticket 20) reports go or no-go.**

## The operating principle: AI resolves, experts decide what matters

This is a partnership between machine and human, not automation with a human afterthought:

- **The machine prepares, resolves and processes as much as it can,** and knows when it is unsure.
  Calibrated probabilities with a precision guarantee split every decision into three bands:
  - **accept:** resolved by the machine, sampled for QA;
  - **reject:** dropped, sampled for QA;
  - **review:** goes to a person, already prepared.
- **The expert gets a very convenient workbench** to resolve the review band, and to QA everything
  else against enterprise standards. Every decision they make is kept as a constraint, so it
  survives rebuilds. It also trains the machine, so the review band shrinks over time.
- **The measure of success is both sides:** precision of what the machine accepts, and the expert
  effort per resolved fact. Precision alone is not enough.

## 1. Resolve against anchors first; cluster only what is left

Resolution is first a linking problem, and only for what remains a clustering problem.

- **Anchors** form the knowledge base:
  - UC master and reference tables: customers, suppliers, products, code lists;
  - approved concepts;
  - UC objects.
- **Linking:** a text mention or warehouse record is linked to an anchor by retrieval plus ranking.
  The cost is O(n·k), where k is the number of candidates.
- **NIL clustering:** only mentions that match no anchor are clustered among themselves. This is the
  entity-linking design from the TAC-KBP tradition.
- **Why it scales:** an O(n²) clustering problem with no guarantees becomes a bounded ranking
  problem.
- **Why it is safe:** "UC is the source of truth" holds by construction. An anchor is only ever
  linked to, never redefined.

## 2. Three stages, each with its own job and metric

| Stage | Job | How | Metric |
|---|---|---|---|
| Blocking / candidates | Recall | Several passes, unioned: token, phonetic and initials keys; top-k TF-IDF; ANN over embeddings (the embedder's only remaining job, ADR-0068); for UC objects, catalog/schema keys, column type and lineage neighbours. Salted Spark joins on Databricks. | Pair completeness and reduction ratio, per pass |
| Pairwise scoring | Precision | Today's rungs become **features**. The type check and the ADR-0046 distinguishing-token guards become **vetoes**. Scorer by modality: Fellegi-Sunter / Splink (MIT; EM without labels, term-frequency weights, Spark, incremental in v5) for structured records; a fine-tuned multilingual cross-encoder for text; an LLM picking from a shortlist only in the review band, cached and egress-gated. | Calibrated probability per (modality, type) |
| Clustering | Consistency | Constrained. Human must-links first; cannot-links and vetoes are hard constraints; accepted edges are applied in descending probability. Two clusters merge only after a cluster-level check, and one strong negative blocks the merge. No unchecked transitive closure. Optional Pivot correlation clustering. Homonym splits are *proposed* for review. | B-cubed, pairwise, generalized merge distance |

## 3. Calibrated decisions with a stated guarantee

- **Calibration:** Platt or isotonic, per (modality, type), fitted on reviewed pairs.
- **Thresholds:** Learn-then-Test sets the accept and reject thresholds with a guarantee, for
  example "auto-accept precision ≥ 0.95 with 95% confidence" (the owner's v1 bar, 2026-11-08).
- **No labels, no accept band:** a (modality, type) without enough labels gets no accept band.
  Unknown stays unknown, which keeps the no-silent-over-merge rule.
- **Why the old cutoffs go:** this replaces the hand-set 0.82, 0.84 and 0.93. ADR-0046 measured why
  they cannot work: genuinely distinct names score 0.85–0.89, overlapping correct merges at
  0.86–0.95.

## 4. Collective (relational) evidence

- Entities and relations are resolved alternately, for a bounded number of rounds.
- Shared neighbours become features: the same employer, address or co-mentioned entities, or the
  same lineage neighbours.
- This separates two "Müller"s, or two `customer_id` columns, where names alone cannot.

## 5. Relation resolution (detail: [research 21](../research/21-relation-resolution-at-scale.md), stages R0–R8)

- **Schema first.** Canonical predicates come from the ontology and from UC. Declared and verified
  foreign keys and joins are gold relations.
- **Open phrases** are mapped Extract-Define-Canonicalize style: retrieve canonical relations by
  their definitions, then rerank or select.
- **Domain and range types are vetoes.** Inverse and symmetric relations are detected;
  `RelationNormalizer` already swaps inverses.
- **One edge per fact.** Each edge is aggregated over its supporting mentions, with a calibrated
  confidence and a support count. A violated functional property is flagged as a contradiction for
  review.
- **Unmapped relations become proposals** for new properties, never silent new labels.
- **Unit of work:** each distinct (raw label, head class, tail class) triple is canonicalised, not
  each mention. That is thousands of items rather than millions, so a model in the review band is
  affordable.
- **Today's gap:** one edge is emitted per mention, and the fuzzy label fold is unsafe
  (`not_located_in` folds into `located_in`). Both are fixed by aggregation and by the definition matcher.

## 6. The expert workbench (ticket 22)

What a reviewer needs to resolve and QA quickly, with confidence:

- **A queue ranked by impact × uncertainty.** Edges that bridge large clusters, or that would break
  a cannot-link, come first. The queue is diverse across types and sources, and grouped by
  **cluster**, not by pair, so one action can approve, split or merge a whole group.
- **Prepared evidence cards, side by side:**
  - mentions shown in their source context, with the spans highlighted;
  - UC object cards: comment, tags, lineage, profile statistics, never raw personal values;
  - the features that drove the score, as a match-weight waterfall;
  - the calibrated probability, and **why the machine was unsure**.
- **The AI's pre-drafted suggestion and its rationale.** The reviewer confirms or overrides, and
  disagreements are recorded.
- **Keyboard-first, single-key decisions:** accept, reject, split, merge, not-sure, skip.
- **Generalise a decision into a rule:** "apply to every pair where …". The rule is a reviewable
  proposal that turns one decision into many.
- **A graph view of a cluster**, with bridge edges highlighted, for split decisions.
- **Conflicts surface as queue items.** New evidence that contradicts a human decision is never
  resolved silently.
- **QA on enterprise standards:**
  - sampled audits of auto-accepted and auto-rejected decisions, reported as precision estimates
    with confidence intervals;
  - four-eyes approval for critical types;
  - certification of a version;
  - reviewer agreement;
  - throughput: decisions per hour, auto-resolution rate, review-band size over time.
- **Where it runs:**
  - standalone, for example as a Databricks App;
  - inside Metis's review routes when running as a sidecar (tickets 16 and 17).

  Splink's comparison viewer and cluster studio (MIT) and Metis's review routes are references.

## 7. Human decisions are constraints

- **A versioned decisions store.** It holds must-link and cannot-link facts and approved proposals,
  keyed on stable member keys (UC ids, mention content keys), each with actor, time and reason.
- **Replayed first** on every run; it survives rebuilds and deltas.
- **It feeds calibration and active learning.**

## 8. Incremental without drift

- ADR-0061 stays: a delta equals a full rebuild.
- Add a pair-score cache, keyed by the content of both sides, the scorer version and the calibrator
  version.
- Re-cluster only the touched components.
- Keep entity ids stable by maximum overlap with the previous run, and record id lineage.

## 9. Two execution scales

- **Metadata level:** 10⁴–10⁶ UC objects × about 10³ concepts. Single-node Python/DuckDB, with
  aggregates and sketches computed in the warehouse by SQL.
- **Instance level:** 10⁷–10⁸ records or mentions. Spark on Databricks: Splink, connected components
  over a Delta candidate table, and salting for skewed blocks.
- **The T-Box stays small.** Ontology induction (research 05) works on resolved,
  evidence-aggregated clusters, never on raw mentions. Shapes are validated in Spark SQL.

## What stays hard (the spike measures each)

- **Homonyms with no context.**
- **Cold start:** until reviewers have labelled pairs, most links sit in the review band. Expert
  effort is therefore the first number that matters.
- **LLM egress and cost.**
- **German compound names.** Deferred: v1 is English only (owner, 2026-11-08), and German follows
  once English passes.
- **Gold data.**

## Where Latence stands, and what carries over

- **Today:**
  - normalised surfaces are clustered with first-rung-wins and fixed thresholds;
  - there is no calibration, no cannot-link and no ER metric;
  - canonical ids churn;
  - glinker takes 73% of the dogfood runtime.
- **Carries over:** the audit trail including near-misses, Evidence, determinism, affected-set
  deltas, the guards (as vetoes), the exact and alias rules, and GLinker as one candidate source.
- **Rewritten:** the resolution core itself.
