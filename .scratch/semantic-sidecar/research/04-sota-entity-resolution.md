# Research 04: the state of the art in entity resolution with calibration, constraints and scale

Ticket: [issues/04-sota-entity-resolution.md](../issues/04-sota-entity-resolution.md). Date: 2026-11-08.

**Method note.** The network policy blocked arxiv.org, aclanthology.org, huggingface.co and the
Splink docs site. Paper claims therefore come from abstracts, search snippets, GitHub pages and
proceedings listings. Figures I could not check against a primary page are marked UNVERIFIED.

## Summary

- **Keep the cascade, but change its role.** Exact and alias stay as rules that decide. The type
  check and the distinguishing-token guard stay as vetoes. Acronym, substring, fuzzy and embedding
  similarity become *features* of one scored candidate pair.
- **Calibrate the score.** Each pair gets a calibrated match probability. Two thresholds, chosen
  for a precision target with a finite-sample guarantee (Learn-then-Test), give three bands:
  accept, review, reject. The fixed 0.84 and 0.93 cutoffs go away.
- **Score by modality.** Warehouse metadata uses a Fellegi-Sunter/Splink scorer, which is
  interpretable, can train without labels and runs on Spark. Text mentions use a fine-tuned
  cross-encoder.
- **LLM only in the review band.** Use a "select from a shortlist" prompt, and cache and audit
  every answer.
- **Cluster under constraints, not by transitive closure.** Human must-links are contracted
  first, and cannot-links and vetoes are hard constraints. Before two clusters merge, the merge is
  checked: one piece of strong negative evidence blocks it.
- **Store human decisions as constraints** keyed on stable member identities: mention content
  keys and UC object ids, never cluster ids. The same decisions are training labels for active
  learning.
- **The biggest structural limit today: the engine clusters normalised surfaces, not mentions.**
  Two different entities that share a name can never be split.

## The current engine and its limits

`EntityResolver` (ADR-0026) groups mentions by `normalize_text` (NFKC + casefold). It then walks
every pair of distinct surfaces through five rungs: exact → alias → acronym → substring →
"embedding". The embedding rung is a lexical Jaccard stand-in. The first rung that fires decides.
Its constant or ratio confidence is compared with `min_confidence=0.82`. Merges go into a
Union-Find whose representative is the lexicographically smallest surface. Every decision,
applied or not, is recorded in a `MergeAudit`.

The GPU Provider (ADR-0046) adds:

- GLinker neural linking against a KB built from the corpus;
- a `difflib` fuzzy rung (≥0.84, same label only), guarded by numeric-token equality and a
  leading-short-code check;
- a Granite-STS embedding rung, off by default (≥0.93 when enabled).

The surrounding pipeline:

- type labels are canonicalised first, by a similar cascade (ADR-0054);
- a delta recomputes the corpus-level phase exactly over the union of reused and fresh records
  (ADR-0061);
- the committed store is scoped to the affected set, a closure over blocking keys
  (ADR-0018/0029).

**What ADR-0046 learned.** Granite is a retrieval embedder. On short entity names, its cosine for
genuinely distinct names (0.85–0.89) overlaps the cosine of correct merges (0.86–0.95). Turning
the rung on caused 49 false merges, including every distinct person collapsing into one, and no
clean threshold exists. Character similarity fails the same way: `K² Ventures UG` vs
`D² Ventures UG` scores 0.93, higher than a correct OCR merge at 0.84. A single similarity score
cannot carry the decision.

**Limits that matter for v1:**

1. **Surface-level clustering.** A homonym cannot be split, and context never enters the
   decision.
2. **The confidences are not probabilities.** Per-rung constants (alias 0.95, acronym 0.9) and
   ratios are compared against one `min_confidence`. None of them is calibrated.
3. **The first rung that fires decides.** Evidence is never combined. Only the guards act as
   negative evidence.
4. **Unchecked transitive closure.** A~B and B~C put A and C together, and the clusters being
   joined are never checked.
5. **No cannot-link and no persisted reviewer decisions.** The `aliases` config is the only
   must-link channel.
6. **Canonical id churn.** The id comes from the lexicographically smallest surface, so a new
   surface that sorts first renames the entity.
7. **Quadratic pair scan.** The fuzzy path compares every pair; at about 1M surfaces that is
   about 5e11 pairs.
8. **Text only.** There are no structured attributes for warehouse objects.
9. **No ER metric.** I found no pairwise or B-cubed metric under `latence_core`, although
   ADR-0026 names pairwise merge precision as the bar.

## 1. Learned pairwise matchers and cross-encoders

Ditto is still the supervised baseline. A controlled study ran 1,215 fine-tuning runs: three
architectures, Qwen3 at 0.6B, 4B and 8B, and nine datasets
([arXiv 2607.24688](https://arxiv.org/abs/2607.24688); [code](https://github.com/Jantory/llm-trained-matcher)).
It found:

- cross-encoders keep a consistent edge over bi-encoders, because they encode the pair jointly;
- scale narrows that gap but does not close it;
- generative matchers do not universally win.

This is the published form of ADR-0046's lesson: an independent embedding generates candidates;
it should not decide matches.

Other results:

- AnyMatch, a small zero-shot model, comes within 4.4% of GPT-4 MatchGPT at much lower cost
  ([arXiv 2409.04073](https://arxiv.org/abs/2409.04073)).
- Fine-tuning LLMs helps small models but hurts transfer across domains
  ([arXiv 2409.08185](https://arxiv.org/abs/2409.08185)).
- LinkTransformer is GPL ([arXiv 2309.00789](https://arxiv.org/html/2309.00789v2)), so it is a
  reference only.
- A plausible German and English cross-encoder base is `BAAI/bge-reranker-v2-m3` (Apache-2.0,
  UNVERIFIED).

## 2. LLMs only in the uncertain band

There are three LLM matching strategies:

- **match** judges one pair;
- **compare** weighs two candidates against each other;
- **select** picks the best record from a shortlist.

The findings:

- **Select is cheapest.** It costs under half as much as match, but suffers from position bias.
  ComEM adds up to 4% F1 at lower cost ([arXiv 2405.16884](https://arxiv.org/abs/2405.16884)).
- **A learned policy can route work.** CaRL-EM (ACL 2026) learns which strategy and which model
  size to use. It reports 78% lower cost at higher F1 (UNVERIFIED, preprint figures)
  ([ACL](https://aclanthology.org/2026.acl-long.1258/)).
- **Band cascades work in practice.** A product-linking cascade auto-accepts and auto-rejects with
  a calibrated cross-encoder and sends only the middle band to an agent. An agent call reportedly
  costs about 7,000× a cross-encoder call (UNVERIFIED), so the band's width is the cost lever
  ([arXiv 2608.25037](https://arxiv.org/pdf/2608.25037)).
- **Escalation can carry a guarantee.** "Trust or Escalate" (ICLR 2025) escalates from cheap to
  expensive judges with guaranteed agreement with humans
  ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/08dabd5345b37fffcbe335bd578b15a0-Paper-Conference.pdf)).
- **LLMs can cluster directly.** LLM-CER (SIGMOD 2026) clusters small record sets in one call
  ([arXiv 2506.02509](https://arxiv.org/abs/2506.02509)).
- **Prompts are not portable.** The best LLMs need few examples and generalise to unseen
  entities, but each model and dataset needs its own prompt
  ([EDBT 2025](https://www.sowi.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)).

**For this engine:** use select over a shortlist in shuffled order, only inside the review band,
and only after an explicit egress decision. For replay, key each answer on the pair content hash,
the model id and the prompt hash.

## 3. Probability calibration and precision-targeted thresholds

Raw cross-encoder scores are badly calibrated and drift when the checkpoint changes.
Fellegi-Sunter weights are only as good as their independence assumption.

- **Calibration.** Platt scaling has two parameters and works with little data. Isotonic
  regression is more flexible but overfits small sets
  ([Niculescu-Mizil & Caruana](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)).
- **Thresholds with a guarantee.** Learn-then-Test tests candidate thresholds as hypotheses in a
  fixed sequence. The chosen threshold guarantees P(precision ≥ 1−α) ≥ 1−δ
  ([arXiv 2110.01052](https://arxiv.org/abs/2110.01052)). MAPIE implements it for precision
  ([docs](https://mapie.readthedocs.io/en/latest/theory/risk-control/)).

This turns "0.84 felt right" into "auto-accept at ≥0.98 precision with 95% confidence on N
reviewed pairs". Calibrate per modality and per entity type.

## 4. Fellegi-Sunter and Splink, including Spark and Databricks

Splink ([repo](https://github.com/moj-analytical-services/splink), MIT) runs on DuckDB, Spark
and Postgres. It claims about 1M records per minute on a laptop, and 100M+ records on Spark.

**Strengths:**

- EM estimation of the m and u probabilities without labels;
- term-frequency adjustments, so a common name weighs less than a rare one
  ([docs](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html));
- Spark salting for skewed blocks;
- explainable match weights.

**Splink 5.0.0** (2026-09-28) is not fully backwards compatible. It adds `predict_within()` and
`predict_between()`, which score only the new comparisons when records arrive
([announcement](https://moj-analytical-services.github.io/splink/blog/2026/09/28/splink-500-released.html);
the exact semantics are UNVERIFIED).

**Limits:**

- field agreements are assumed independent given match status, which is often false
  ([Resnick](https://www.norc.org/content/dam/norc-org/documents/standard-projects-pdf/Resnick_2017_JSM.pdf));
- clustering is connected components at a threshold, and I found no cannot-link support;
- weights above about 53 cannot be told apart when clustering
  ([issue](https://github.com/moj-analytical-services/splink/issues/3002)).

**On Databricks:** the Spark backend is the realistic path; I found no Databricks-specific
documentation. Zingg runs on Spark and supports active learning, but it is AGPL-3.0
([repo](https://github.com/zinggAI/zingg)), so it is a reference only.

## 5. Clustering under must-link and cannot-link

- **Correlation clustering** needs no entity count but is NP-hard. Pivot gives an expected
  3-approximation and has parallel and MPC variants
  ([survey](https://arxiv.org/pdf/2208.12636)).
- **Constrained correlation clustering** accepts only clusterings that satisfy every hard
  must-link and cannot-link ([arXiv 2608.05700](https://arxiv.org/pdf/2608.05700)).
- **A practitioner report** from a self-serve pipeline has three lessons
  ([arXiv 2607.26298](https://arxiv.org/abs/2607.26298); a third-party review flags
  inconsistencies in its ablation, UNVERIFIED):
  1. no single matcher dominates;
  2. precision needs rule vetoes and recall needs more varied candidates, not a shared threshold;
  3. transitive merges spread false positives, so every merge between groups is re-verified, and
     any negative evidence blocks it.

**Recommended algorithm:**

1. Contract the must-links (human, alias, exact) with Union-Find. A must-link that contradicts a
   cannot-link is reported as an error.
2. Take the accepted-band edges in descending calibrated probability, with deterministic tie
   breaks.
3. Join two clusters only if no cannot-link or veto crosses them and a cluster-level check
   passes: no cross pair in the reject band, and a minimum-linkage bar.
4. Optionally refine each component with Pivot.

Each union records its edge and its check.

## 6. Active learning from reviewer decisions

- **Uncertainty sampling breaks at small budgets**
  ([arXiv 1909.03585](https://arxiv.org/pdf/1909.03585)).
- **Battleship** combines uncertainty with diversity in latent space
  ([arXiv 2311.15685](https://arxiv.org/abs/2311.15685)).
- **dedupe** (MIT, [repo](https://github.com/dedupeio/dedupe)) learns weights and blocking
  predicates from labelled pairs, and is a useful reference design.

**For this engine:** the queue is the review band. Rank it by uncertainty × cluster impact: an
edge that bridges large clusters, or one that would trigger a cannot-link, comes first. Add
diversity across types and modalities. Each decision becomes both a persisted constraint and a
labelled pair for recalibration.

## 7. Splitting homonyms

Author-name disambiguation splits graph hubs and uses node split/merge
([Shin et al.](https://www.researchgate.net/publication/262937993_Author_name_disambiguation_using_a_graph_model_with_node_splitting_and_merging_based_on_bibliographic_information)),
or uses community detection ([DISC](https://journals.sagepub.com/doi/10.1177/0165551518761011)).
Splink's graph metrics (bridges, density, centralisation) flag over-linked clusters
([docs](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/graph_metrics.html)).

v1 needs two things:

- **Resolution at the mention or record level**, so that a split can be expressed at all.
- **A pass after clustering** that flags clusters with bridge edges, low density, conflicting
  types or incompatible attribute values. It re-clusters a flagged cluster with correlation
  clustering and emits a *proposed* split for review.

A same-surface link becomes a default must-link that a human cannot-link can override.

## 8. Incremental and streaming ER under deltas

- **Incremental record linkage** merges, splits and moves clusters on inserts, deletes and
  updates, aiming at batch-equivalent quality
  ([Gruenheid et al., PVLDB 2014](http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf)).
- **FAMER's incremental repair** reduces sensitivity to insertion order, but leaves some entities
  in clusters that are not optimal
  ([ESWC 2020](https://dbs.uni-leipzig.de/files/research/publications/2020-6/pdf/FAMER_Incremental_eswc2020.pdf)).

ADR-0061's exact recompute over the union is stronger than either: incremental equals rebuild by
construction. Once a learned scorer runs, the risk is cost. Four changes keep the recompute exact
while cutting that cost:

- **Cache pair scores** by (content of both sides, scorer version, calibrator version), so a delta
  scores only new pairs, as Splink's `predict_between` does.
- **Re-cluster only the touched components** of the candidate graph.
- **Replay human constraints first** on every run.
- **Keep entity ids stable.** Map each new cluster to its old id by maximum member overlap with
  deterministic ties, and record id lineage on merges and splits.

## 9. Blocking at warehouse scale

Sparkly does top-k TF-IDF blocking with Lucene on Spark, configures itself, and beats 8 blockers
([PVLDB 16](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf); licence UNVERIFIED). pyJedAI
(Apache-2.0, [repo](https://github.com/AI-team-UoA/pyJedAI)) bundles blocking and meta-blocking.

**Recommendation:** run several blocking passes and union their candidates:

- the current token keys;
- phonetic and initials keys;
- top-k TF-IDF;
- ANN over embeddings, which is where the retrieval embedder stays useful (ADR-0068);
- for warehouse objects, keys from catalog and schema, column type and lineage neighbours.

Measure pair completeness and reduction ratio for each pass. On Databricks the candidate set is a
Delta table built by Spark SQL joins with salting.

## 10. ER evaluation

- **Pairwise P/R/F1** is dominated by large clusters and is over-optimistic on large databases.
- **B-cubed** weights each record equally.
- **Generalized merge distance** counts splits and merges. Variation of information is a special
  case. Its rankings can disagree with the other metrics
  ([Menestrina et al., PVLDB 2010](https://api.crossref.org/works/10.14778%2F1920841.1920871)).
- **Common estimators are biased**, so benchmark F1 can misrepresent population performance
  ([arXiv 2404.05622](https://arxiv.org/abs/2404.05622)). The `er-evaluation` package is reported
  as AGPL-3.0, so the metrics would be reimplemented.

**What the Quality Report should state:** pairwise, B-cubed and GMD/VI against a gold set of
clusters, broken down by modality and type. It should also give the review-band rate and the
audited precision of auto-accepts.

Matching warehouse columns and tables to concepts is schema matching. Magneto (PVLDB 2025)
combines small and large language models for it ([arXiv 2412.08194](https://arxiv.org/abs/2412.08194)).
bdi-kit is Apache-2.0 ([repo](https://github.com/VIDA-NYU/bdi-kit)); whether it includes Magneto
is UNVERIFIED.

## Licences

| Tool | Licence | Use |
|---|---|---|
| Splink | MIT | dependency candidate (Spark, DuckDB) |
| dedupe | MIT | reference |
| pyJedAI | Apache-2.0 | reference or dependency for blocking |
| bdi-kit | Apache-2.0 | reference for the warehouse modality |
| MAPIE | BSD-3-Clause, UNVERIFIED | Learn-then-Test |
| Zingg | AGPL-3.0 | reference only |
| er-evaluation | AGPL-3.0 (reported) | reimplement |
| LinkTransformer | GPL | reference only |
| igraph (Splink `is_bridge`) | GPL-2.0, UNVERIFIED | avoid; use networkx (BSD) |

## Recommended architecture: resolution engine v1

1. **Records.** A text mention carries its surface, type, context and document key. A warehouse
   object carries its UC id, name, comment, type, tags, lineage and usage. Every record has a
   stable member key. This generalises `MentionInput`.
2. **Normalise.** Keep NFKC + casefold. Exact and alias matches emit must-link edges.
3. **Block.** Run several passes and union them (§9), with metrics for each pass.
4. **Features.** The old rungs become comparators:
   - acronym;
   - token-substring ratio;
   - `difflib` ratio;
   - Jaro-Winkler;
   - embedding cosine, as a feature only;
   - context overlap;
   - for warehouse objects, name, comment, type and lineage comparators.

   The type check and the ADR-0046 numeric and leading-code guard become explicit **vetoes**.
5. **Scorer Provider.** It outputs log-odds. Three Providers:
   - Fellegi-Sunter with term-frequency adjustments, trained by EM. This is the default for
     warehouse objects; it runs in core Python/DuckDB, or as Splink on Spark.
   - A fine-tuned multilingual cross-encoder on GPU, for text.
   - A deterministic rule scorer that reproduces today's behaviour, as the CPU baseline.
6. **Calibrate.** Fit Platt or isotonic calibration per (modality, type) on reviewed pairs. Set
   the accept and reject thresholds with Learn-then-Test at a declared precision. A
   (modality, type) without enough labels has no accept band: the link stays unknown.
7. **Review band.** It goes to the optional LLM select step (behind the egress decision, with
   cached answers) and to the human queue. Both are audited with the actor's identity.
8. **Constrained clustering** (§5).
9. **Homonym-split proposals** (§7).
10. **Stable ids** by overlap with the previous run, with id lineage.
11. **Audit.** Each edge records its features, raw and calibrated score, band, actor, the scorer
    and calibrator versions, and Evidence. Each union records its cluster check.

**Decisions store.** A versioned set of must-link and cannot-link facts over member keys, each
with actor, timestamp and reason. It is replayed first on every run and survives rebuilds and
deltas. It also feeds calibration and the active-learning queue. A conflict with new evidence is
surfaced, never resolved silently.

**Keep:** NFKC; the exact and alias rules; the guards, as vetoes; the audit, including
near-misses; determinism; incremental equals rebuild; affected-set scoping; and GLinker as one
candidate source.

**Retire:** first-rung-wins, the constant confidences, the single `min_confidence`, and unchecked
transitive closure.

## Open questions

1. **Granularity of the text path.** Should every mention resolve on its own, or should "same
   surface in one document" be a default must-link? The second is cheaper but cannot split
   homonyms within a document.
2. **Cold start.** What is the minimum labelled set before a (modality, type) gets an accept band?
   Should v1 accept only exact and alias matches until reviewers have labelled enough?
3. **Splink.** Take it as a dependency (heavy), or implement Fellegi-Sunter in core and use
   Splink only in the Databricks runner?
4. **Cross-encoder.** Which Apache-2.0 base works for German and English, and where do its labels
   come from: reviewers, or synthetic pairs from the alias lists?
5. **LLM egress scope.** May surface strings only leave the boundary, or also context windows and
   column comments?
6. **Text to UC links.** Does linking text to UC run through the same clusterer, or is it anchored
   linking with the UC object fixed? The second fits "UC is the source of truth".
7. **Byte-identical replay with a GPU scorer.** Is a score cache enough, or must a replay read
   only cached scores?
