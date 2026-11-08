# Research 04: the state of the art in entity resolution with calibration, constraints and scale

Ticket: [issues/04-sota-entity-resolution.md](../issues/04-sota-entity-resolution.md). Date: 2026-11-08.

**Method note.** Web search worked, but the network policy blocked arxiv.org, aclanthology.org,
huggingface.co and the Splink docs site. Paper claims below therefore come from abstracts, search
snippets, GitHub pages and proceedings listings, not full texts. Any figure I could not check
against a primary page is marked UNVERIFIED.

## Summary

- **Keep the cascade, but change what it is.** Exact and alias stay as rules that decide. The
  type check and the ADR-0046 distinguishing-token guard stay as vetoes. Acronym, substring,
  fuzzy and embedding similarity stop deciding and become *features* of one scored candidate
  pair.
- **Score pairs, calibrate them, and pick thresholds by guaranteed precision.** Each candidate
  pair gets a calibrated match probability. Two thresholds then give three bands: accept, review
  and reject. The thresholds are chosen against a target precision, with a finite-sample
  guarantee (Learn-then-Test). They are not hand-set.
- **Use the scorers that fit each modality.** A Fellegi-Sunter/Splink-style scorer fits warehouse
  metadata: it is interpretable, can train without labels, and runs in Spark. A fine-tuned
  cross-encoder fits text mentions. The current 0.84 and 0.93 constants go away.
- **Call an LLM only for the review band.** Use a "select from a shortlist" prompt, cache the
  answer and audit it. The LLM is never the default path.
- **Cluster under constraints, not by transitive closure.** Human must-links are contracted
  first. Cannot-links and vetoes act as hard constraints. Every merge between two clusters is
  checked first: one strong piece of negative evidence blocks it.
- **Persist human decisions as constraints keyed on stable member identities.** The keys are
  mention content keys and Unity Catalog object ids, never cluster ids. The same decisions feed
  active learning.
- **The biggest structural limit today is that the engine clusters normalised surfaces, not
  mentions.** Two different people who share a name can never be split. v1 must resolve at the
  mention or record level.

## The current engine and its limits

`latence_core.disambiguation.EntityResolver` (ADR-0026) groups mentions by `normalize_text`
(NFKC + casefold). It then walks every pair of distinct surfaces through exact → alias → acronym
→ substring → "embedding" (here a lexical Jaccard stand-in). The first rung that fires decides
the pair. A rung's confidence is checked against `min_confidence=0.82`, and the result is folded
into a Union-Find whose representative is the lexicographically smallest surface. Every
decision, applied or not, is logged as a `MergeAudit`.

The GPU Provider (ADR-0046) adds a corpus-internal KB with GLinker neural linking and a fuzzy
`difflib` rung (≥0.84, same label only), guarded by numeric-token equality and a
leading-short-code check. Its Granite-STS embedding rung is off by default, with a ≥0.93 cutoff
when enabled. Type labels are canonicalised beforehand by the same kind of cascade (ADR-0054).
Deltas recompute the corpus-level phase exactly over the union of reused and fresh records
(ADR-0061). The committed store is scoped to the affected set, computed as a closure over
blocking keys (ADR-0018/0029).

ADR-0046 records the lesson the ticket asks about. Granite is a retrieval embedder. On short
entity names, its cosine similarities for genuinely different names (0.85–0.89) overlap those of
correct merges (0.86–0.95), so no clean threshold exists. Turning the rung on produced 49 false
merges, with every distinct person collapsed into one. The guard examples (`K² Ventures UG` vs
`D² Ventures UG` at 0.93) show the same thing for character-level similarity: one similarity
score cannot carry the decision.

The limits that matter for v1:

1. **Surface-level clustering.** All mentions with one normalised surface are a single node before
   any rung runs, so a homonym cannot be split and context never enters the decision.
2. **The confidences are not probabilities.** Alias 0.95 and acronym 0.9 are constants, while
   substring and fuzzy scores are ratios. All of them are compared against one `min_confidence`,
   and none is calibrated against labels.
3. **The first rung that fires decides.** Evidence is never combined. The only negative evidence
   is the guards.
4. **Transitive closure.** Union-Find merges chains: A~B and B~C put A and C together, and
   nothing checks the clusters being joined.
5. **No human constraint channel.** The `aliases` config is the only must-link. There is no
   cannot-link, and nothing persists reviewer decisions. A grep of `packages/` for must-link or
   cannot-link found nothing relevant.
6. **Canonical id churn.** The id comes from the lexicographically smallest surface. A new,
   smaller surface that joins a cluster therefore renames the entity.
7. **Quadratic pair scan.** The fuzzy cascade scans every pair. The `exact_only` path was cut down
   for exactly this reason: about 1M surfaces meant about 5e11 pairs.
8. **Text only.** `MentionInput` carries text, label and confidence. There are no structured
   attributes for warehouse objects.
9. **No ER metric in the harness.** I found no pairwise or B-cubed implementation under
   `latence_core`, although ADR-0026 names pairwise merge precision as the number to beat.

## 1. Learned pairwise matchers and cross-encoders

Ditto, a cross-encoder over serialised pairs, is still the standard supervised baseline. The
most useful recent evidence is a controlled study of 1,215 fine-tuning runs: three architectures,
Qwen3 at 0.6B, 4B and 8B, and nine datasets
([Zhang et al., arXiv 2607.24688](https://arxiv.org/abs/2607.24688);
[code](https://github.com/Jantory/llm-trained-matcher), licence not stated). Its findings:

- Cross-encoders keep a consistent edge over bi-encoders because they encode the pair jointly.
  Scale narrows the gap but does not close it.
- Generative matchers do not universally beat cross-encoders.
- For bi-encoders, embedding-oriented checkpoints give the best start.

This is the published form of ADR-0046's lesson: an independent embedding is a candidate
generator, not a judge.

AnyMatch, a small model for zero-shot matching, reports average quality within 4.4% of GPT-4
MatchGPT at much lower cost ([arXiv 2409.04073](https://arxiv.org/abs/2409.04073)). Fine-tuning
LLMs helps small models, but it hurts transfer across domains
([Steiner et al., arXiv 2409.08185](https://arxiv.org/abs/2409.08185)).

LinkTransformer is GPL ([arXiv 2309.00789](https://arxiv.org/html/2309.00789v2)), so it is a
reference only. For a German and English cross-encoder base, a multilingual reranker such as
`BAAI/bge-reranker-v2-m3` is plausible, but its Apache-2.0 licence is UNVERIFIED because the
model card could not be fetched.

## 2. LLMs only in the uncertain band

- **Prompt strategy.** Wang et al. compare three LLM strategies. *Match* judges one pair,
  *compare* weighs two candidates against each other, and *select* picks the best of a
  shortlist. Select is the most cost-effective at under half the cost of match, but it suffers
  from position bias. Their compound framework ComEM adds up to 4% F1 at lower cost
  ([arXiv 2405.16884](https://arxiv.org/abs/2405.16884)).
- **Learned routing.** CaRL-EM (ACL 2026) learns a policy over Match, Compare, Select and Decide
  and over model sizes. It reports a 78% cost reduction with higher F1, and about 89% of
  supervised quality at 94% less cost (UNVERIFIED: these figures are from the preprint)
  ([ACL 2026.acl-long.1258](https://aclanthology.org/2026.acl-long.1258/)).
- **Cascades with a band.** A product-linking cascade gives the clearest precedent. A calibrated
  cross-encoder auto-accepts above HIGH and auto-rejects below LOW, and only the middle band goes
  to an expensive agent. The width of that band is the cost lever, since an agent call is said to
  cost about 7,000× a cross-encoder call (UNVERIFIED: from a summary)
  ([arXiv 2608.25037](https://arxiv.org/pdf/2608.25037)).
- **Escalation with a guarantee.** "Trust or Escalate" (ICLR 2025) escalates from a cheap judge to
  an expensive one with a guaranteed level of agreement with humans
  ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/08dabd5345b37fffcbe335bd578b15a0-Paper-Conference.pdf)).
- **LLM clustering.** LLM-CER (SIGMOD 2026) has the LLM cluster small record sets directly. It
  reports up to 5× fewer calls (UNVERIFIED) ([arXiv 2506.02509](https://arxiv.org/abs/2506.02509)).

Peeters, Steiner and Bizer (EDBT 2025) find that the best LLMs need few or no examples and are
more robust to unseen entities. They also find that the best prompt depends on the model and
dataset
([PDF](https://www.sowi.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)).

**For this engine:** use the select strategy over the cluster shortlist, with shuffled candidate
order to counter position bias, and only inside the review band. Sending data to the model needs
the map's explicit egress decision. For replay, key each answer on the pair content hash, the
model id and the prompt hash.

## 3. Probability calibration and precision-targeted thresholds

Raw cross-encoder scores are badly calibrated, and fixed cutoffs drift when a checkpoint changes.
Fellegi-Sunter weights are log-likelihood ratios and are only as good as their independence
assumption.

The standard fixes are Platt scaling (two parameters, good with little data) and isotonic
regression (more flexible, but it overfits small sets)
([Niculescu-Mizil & Caruana 2005](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)).

To pick a threshold with a precision guarantee, use Learn-then-Test. It treats each candidate
threshold as a hypothesis test and uses fixed-sequence testing to choose one, guaranteeing
P(precision ≥ 1−α) ≥ 1−δ ([Angelopoulos et al., arXiv 2110.01052](https://arxiv.org/abs/2110.01052)).
MAPIE implements it for precision ([docs](https://mapie.readthedocs.io/en/latest/theory/risk-control/);
licence BSD-3-Clause, UNVERIFIED).

This replaces "0.84 felt right" with a statement like "auto-accept at ≥0.98 precision with 95%
confidence on N reviewed pairs". Calibrate per modality and per entity type, because names, legal
citations and table columns behave very differently.

## 4. Fellegi-Sunter and Splink, including Spark and Databricks

Splink is MIT-licensed ([repo](https://github.com/moj-analytical-services/splink)) and runs on
DuckDB, Spark and Postgres. It claims about 1M records in about a minute on a laptop, and 100M+
records on Spark. Its strengths for this engine:

- m/u probabilities estimated by EM without labels;
- term-frequency adjustments, so a common surname weighs less than a rare one
  ([docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html));
- salted blocking rules to handle skew on Spark
  ([docs](https://moj-analytical-services.github.io/splink/topic_guides/performance/salting.html));
- waterfall-style explanations of each match weight.

Splink 5.0.0 was released on 2026-09-28 and is not fully backwards compatible. It adds
`predict_within()` and `predict_between()`, which score only the new comparisons when records
are added, and removes `find_matches_to_new_records`
([announcement](https://moj-analytical-services.github.io/splink/blog/2026/09/28/splink-500-released.html);
the exact semantics come from a maintainer's gist and are UNVERIFIED).

Its limits:

- **Conditional independence.** The model assumes field agreements are independent given match
  status, which is often false
  ([Resnick 2017](https://www.norc.org/content/dam/norc-org/documents/standard-projects-pdf/Resnick_2017_JSM.pdf)).
- **Connected-components clustering.** Clustering at a threshold is connected components, with
  no cannot-link support that I could find.
- **Weight ceiling.** Weights above about 53 cannot be told apart when clustering
  ([issue](https://github.com/moj-analytical-services/splink/issues/3002)).
- **igraph licence.** The `is_bridge` graph metric needs igraph (GPL-2.0; UNVERIFIED that Splink
  loads it only optionally).

On Databricks, Splink's Spark backend is the realistic path. I found no Databricks-specific
documentation. Zingg also runs on Spark and has active learning, but it is AGPL-3.0
([repo](https://github.com/zinggAI/zingg)), which conflicts with an Apache-2.0 engine, so it is a
reference only.

## 5. Clustering under must-link and cannot-link constraints

- **Correlation clustering.** It fits ER because it does not need the number of entities. It is
  NP-hard. The Pivot algorithm gives an expected 3-approximation and has parallel and MPC
  variants ([survey](https://arxiv.org/pdf/2208.12636)).
- **Hard constraints.** Constrained correlation clustering treats must-link and cannot-link as
  hard: only clusterings that satisfy all of them are valid
  ([arXiv 2608.05700](https://arxiv.org/pdf/2608.05700)).
- **Richer feedback.** Interactive work suggests cluster-level feedback can need far fewer inputs
  than pairwise feedback ([ICML 2022](https://mlanthology.org/icml/2022/angell2022icml-interactive/)).
- **Practitioner lessons.** A practitioner report from a self-serve ER pipeline (arXiv 2607.26298)
  has three directly relevant lessons:
  1. no single matcher wins everywhere, so the pipeline needs a bake-off;
  2. precision needs rule-based vetoes and recall needs more varied candidate retrieval, not a
     shared threshold;
  3. transitive merges are where a false positive spreads, so every merge between groups needs
     re-verification, where any negative evidence blocks the merge.

  A third-party review flags inconsistencies in its ablation (UNVERIFIED)
  ([abstract](https://arxiv.org/abs/2607.26298)).

**The practical algorithm, and my recommendation:**

1. Contract all must-links, from humans, aliases and exact matches, into super-nodes with
   Union-Find. A must-link that conflicts with a cannot-link is a reported error, not a guess.
2. Sort the remaining accepted-band edges by descending calibrated probability, and break ties
   deterministically.
3. Union two clusters only if no cannot-link or veto crosses them, *and* a cluster-level check
   passes. For example: no cross pair below the reject threshold, and a minimum or average
   linkage above a bar.
4. Optionally refine with Pivot inside each component.

This is Kruskal-style constrained agglomeration. It stays deterministic and auditable, because
each union records its edge and its check.

## 6. Active learning from reviewer decisions

Classic uncertainty sampling falls apart at small label budgets
([arXiv 1909.03585](https://arxiv.org/pdf/1909.03585)). Two methods address that:

- Battleship combines uncertainty with diversity across the latent space for low-resource
  matching ([arXiv 2311.15685](https://arxiv.org/abs/2311.15685)).
- Uncertainty Herding is built to work at any budget
  ([arXiv 2412.20644](https://arxiv.org/abs/2412.20644)).

dedupe is MIT-licensed ([repo](https://github.com/dedupeio/dedupe)) and learns weights and
blocking predicates from reviewer-labelled pairs. It is a reference design.

**For this engine:** the review queue is the uncertain band. Rank it by uncertainty × cluster
impact: does the edge bridge two large clusters, would accepting it trigger a cannot-link, how
large is the affected cluster. Add diversity across entity types and modalities.

Each decision does two things at once:

1. it becomes a persisted hard constraint;
2. it becomes a labelled pair for recalibration and fine-tuning.

## 7. Splitting homonyms

Author-name disambiguation is the main source of methods:

- splitting graph hubs that join many clusters, and node split/merge
  ([Shin et al. 2014](https://www.researchgate.net/publication/262937993_Author_name_disambiguation_using_a_graph_model_with_node_splitting_and_merging_based_on_bibliographic_information));
- splitting by community detection over co-occurrence ([DISC](https://journals.sagepub.com/doi/10.1177/0165551518761011)).

Splink's graph metrics (bridges, density, centralisation) are the off-the-shelf diagnostics for
over-linked clusters ([docs](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/graph_metrics.html)).

**What v1 needs.** First, mention-level resolution, so a split can be expressed at all. Second, a
post-clustering pass that flags suspect clusters: bridge edges, low density, conflicting types,
co-occurring values that cannot belong together (two birthdates, two UC catalogs with conflicting
lineage), and context that splits into separate groups. The pass re-clusters the flagged cluster
with correlation clustering and emits a *proposed split* for review rather than applying it
silently. A same-surface link becomes a default must-link that a human cannot-link can override.

## 8. Incremental and streaming ER that stays consistent under deltas

- **Incremental record linkage.** It merges, splits and moves clusters when records are inserted,
  deleted or changed, aiming at batch-equivalent quality
  ([Gruenheid, Dong & Srivastava, PVLDB 2014](http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf)).
- **Insert-order repair.** FAMER's incremental repair reduces dependence on insert order, but
  some entities still end up in non-optimal clusters
  ([ESWC 2020](https://dbs.uni-leipzig.de/files/research/publications/2020-6/pdf/FAMER_Incremental_eswc2020.pdf)).
- **Progressive ER over dynamic data.** PIER targets that case
  ([EDBT 2023](https://openproceedings.org/2023/conf/edbt/paper-180.pdf)).

Latence's ADR-0061 choice, exact recompute over the union every delta, is stronger than any of
these: incremental equals rebuild by construction. The danger is cost, now that a learned scorer
is in the loop.

The fix keeps exactness:

- **Cache pair scores.** Key them by (content of both sides, scorer version, calibrator version),
  so a delta scores only new pairs, as Splink 5's `predict_between` does.
- **Re-cluster only affected components.** Recompute only the connected components of the
  candidate graph that the delta touches. This is the existing blocking-closure idea, applied
  to the candidate graph.
- **Replay human constraints first** on every run.
- **Make entity ids stable.** After each run, map new clusters to old ids by maximum member
  overlap, with deterministic tie-breaks. Retire or alias ids on merge and split, and record the
  lineage as audit. A canonical id must not depend on which surface sorts first.

## 9. Blocking at warehouse scale

Sparkly does top-k TF-IDF blocking with Lucene on Spark, configures itself automatically, and
reports beating 8 state-of-the-art blockers. It is a strong baseline (licence UNVERIFIED)
([PVLDB 16](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)). pyJedAI (Apache-2.0,
[repo](https://github.com/AI-team-UoA/pyJedAI)) bundles blocking, meta-blocking and matching
methods.

**For this engine:**

- **Run several blocking passes and union the candidates:** the current token keys, phonetic and
  initials keys, top-k TF-IDF, and ANN over embeddings. Embedding blocking is where the retrieval
  embedder stays useful, as ADR-0068 allows.
- **Block warehouse objects natively:** by catalog and schema, by matching column types, and by
  lineage neighbours.
- **Measure every pass:** report pair completeness and reduction ratio for each.
- **On Databricks, keep candidates as a Delta table.** Compute it with Spark SQL joins plus
  salting, so blocking scales without Python loops.

## 10. ER evaluation

- **Pairwise P/R/F1.** Easy to compute, but dominated by large clusters, and over-optimistic on
  large databases.
- **B-cubed P/R/F1.** Computed per record, so each record counts equally; also reported per
  cluster.
- **Cluster metrics.** Exact cluster match, and generalized merge distance with variation of
  information as a special case. GMD counts splits and merges as edit operations, and its
  rankings can disagree with pairwise ones
  ([Menestrina et al., PVLDB 2010](https://api.crossref.org/works/10.14778%2F1920841.1920871)).
- **Estimator bias.** Common estimators are biased, and F1 on benchmarks may not represent
  population performance ([Binette & Reiter, arXiv 2404.05622](https://arxiv.org/abs/2404.05622)).

Their `er-evaluation` package is reportedly AGPL-3.0, so it is a reference only and the metrics
would be reimplemented ([JOSS](https://joss.theoj.org/papers/10.21105/joss.05619.pdf)).

**For this engine:** the Quality Report should state pairwise, B-cubed and GMD/VI against a
labelled gold set of clusters, split by modality and entity type. It should also report the
review-band rate and the realised precision of auto-accepts on audited samples.

**Warehouse metadata note.** Matching columns and tables to concepts is schema matching. Magneto
(PVLDB 2025) combines small and large language models for it at low cost
([arXiv 2412.08194](https://arxiv.org/abs/2412.08194)). The VIDA-NYU bdi-kit toolkit is
Apache-2.0 ([repo](https://github.com/VIDA-NYU/bdi-kit)); whether it ships Magneto is
UNVERIFIED.

## Licences at a glance

| Tool | Licence | Use |
|---|---|---|
| Splink | MIT | dependency candidate for the Spark and DuckDB scorer |
| dedupe | MIT | reference for active learning |
| pyJedAI | Apache-2.0 | reference or dependency for blocking |
| bdi-kit (schema matching) | Apache-2.0 | reference for the warehouse modality |
| MAPIE (Learn-then-Test) | BSD-3-Clause, UNVERIFIED | calibration and thresholds |
| Zingg | AGPL-3.0 | reference only |
| er-evaluation | AGPL-3.0 (reported) | reimplement the metrics |
| LinkTransformer | GPL | reference only |
| igraph (Splink `is_bridge`) | GPL-2.0, UNVERIFIED | avoid, or use networkx (BSD) |

## Recommended architecture: resolution engine v1

The engine runs as a pipeline of stages:

1. **Inputs, as records.** Each record is either a text mention (surface, type, context window,
   document key) or a warehouse object (UC object id, name, comment, data type, tags, lineage,
   usage). Each record carries a stable member key. This generalises `MentionInput`.
2. **Normalise.** Keep NFKC + casefold, unchanged. Compute exact and alias keys, which become
   must-link edges.
3. **Block.** Run several passes and union the results (§9). On Databricks the candidate set is a
   Delta table. Report pair completeness and reduction ratio for each pass.
4. **Extract features.** The old rungs become comparators:
   - exact and alias;
   - acronym;
   - whole-token substring ratio;
   - `difflib` ratio;
   - Jaro-Winkler;
   - numeric-token equality and leading-code difference, the ADR-0046 guard, now an explicit
     veto;
   - type compatibility, also a veto;
   - embedding cosine, a feature only;
   - context overlap.

   Warehouse objects add comparators for name tokens, comment similarity, type compatibility and
   lineage.
5. **Score.** A swappable Provider computes a log-odds score:
   - (a) a Fellegi-Sunter scorer, trained by EM with term-frequency adjustments, which is the
     default for warehouse objects and runs in pure Python/DuckDB or Splink on Spark;
   - (b) a fine-tuned multilingual cross-encoder on GPU, for text mentions;
   - (c) a deterministic rule scorer that reproduces today's behaviour, as the CPU baseline.
6. **Calibrate.** Apply Platt or isotonic calibration per (modality, type) on reviewed pairs. Set
   the accept and reject thresholds by Learn-then-Test at a declared precision target. If a
   (modality, type) has too few labels, it has no accept band: everything goes to review or
   stays unresolved, because an unknown link stays unknown.
7. **Adjudicate the review band** with optional steps:
   - the LLM select strategy, which needs the egress decision and caches its answers;
   - the human queue.

   Both write audit entries naming the actor.
8. **Cluster under constraints** (§5): contract must-links, then constrained agglomeration in
   descending probability with a verified-merge check, then an optional Pivot refinement.
9. **Split homonyms** (§7). The result is proposals, not silent splits.
10. **Assign identities.** Map clusters to stable ids by overlap with the previous run, and record
    id lineage.
11. **Audit.** Every edge decision carries its features, raw score, calibrated probability,
    band, actor (rule, model, LLM or human), scorer and calibrator versions, and Evidence. Every
    union carries its cluster check.

**Human decisions store.** A versioned table of must-link and cannot-link facts between member
keys, with actor, timestamp and reason. It is replayed first on every run, survives rebuilds and
deltas, and feeds the calibrator and the active-learning queue. Any conflict between a decision
and a new run is surfaced, never resolved silently.

**Keep:** NFKC normalisation; the exact and alias rules; the type and distinguishing-token
guards, now as vetoes; the audit discipline, including near-misses; determinism; ADR-0061's
rule that incremental equals rebuild; affected-set scoping; and the GLinker KB linker as one
candidate source.

**Retire:** first-rung-wins; the constant per-rung confidences; the single `min_confidence`; and
unchecked transitive closure.

## Open questions

1. **Granularity of the text path.** Should text resolve every mention individually, or keep
   "same surface in one document" as a default must-link? The second is cheaper; the first is
   needed for homonyms inside one document.
2. **Labels before any labels exist.** What is the minimum labelled set per (modality, type) to
   open an accept band? Should the first release run with no accept band, except exact and
   alias, until reviewers have labelled enough?
3. **Scorer dependency.** Use Splink as a dependency (MIT, but it brings a heavy SQL-engine
   stack), or reimplement Fellegi-Sunter in core and use Splink only in the Databricks runner?
4. **Cross-encoder base and training data.** Which Apache-2.0 multilingual base works for German
   and English, and where do the labels come from (reviewer decisions, or synthetic pairs from
   aliases)? The licence of the bge-reranker weights is UNVERIFIED.
5. **LLM band egress.** What may leave the boundary: surface strings only, or context windows
   and column comments? This needs the map's egress decision.
6. **Cross-modality links.** Does a text mention linking to a UC column go through the same
   clusterer, or is it a separate linking step with the UC object as a fixed anchor? The second
   matches "UC is the source of truth".
7. **Determinism when the scorer is non-deterministic.** Is caching pair scores enough to meet
   "a run replays byte-identically" when the GPU scorer is non-deterministic, or must replays
   read the score cache only?
