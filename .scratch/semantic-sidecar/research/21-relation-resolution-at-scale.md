# Research 21: relation resolution at corpus and Unity Catalog scale

Ticket 21. Date: 2026-11-08. Builds on [04](04-sota-entity-resolution.md), [05](05-ontology-learning-and-standards.md), [02](02-sota-text-extraction.md), [03](03-sota-schema-matching.md) and [01](01-databricks-semantic-surface.md).

**Method note.** arxiv.org, aclanthology.org, ceur-ws.org, linqs.org, lunadong.com and docs.databricks.com did not resolve from this session. GitHub pages were read directly. Paper figures come from abstracts and search summaries; any figure not confirmed on a primary page is marked UNVERIFIED.

## Summary

- **Today there is label folding, not relation resolution.** `GraphBuilder` emits **one edge per relation mention** (`edge_id` hashes the mention's `record_id`), with raw extractor confidence and endpoint-only Evidence. Ten mentions of a fact are ten parallel edges.
- **Canonicalise the signature, not the mention.** The unit is the distinct `(raw label, head class, tail class)`: thousands, not millions, so a model on the review band is cheap.
- **Classic open-KB canonicalisers (CESI, SIST, JOCL, CMVC) are references only.** They are noun-phrase-centric, and relation-phrase quality is barely measured (ReVerb45K has no relation gold).
- **Define, retrieve, verify** (the EDC pattern, MIT). Each property carries a DE/EN definition, domain, range and characteristics. Signatures are matched against definitions, with domain and range as vetoes. An unmapped signature becomes an `OntologyProposal(kind=property)`, never fuzzy-folded or silently dropped.
- **One edge per fact key** `(head entity, property, tail entity, qualifier key)`, with confidence from a calibrated fusion of per-document evidence (Knowledge Vault, knowledge fusion). Counts are not confidence.
- **Knowledge fusion's error analysis gives three rules** (UNVERIFIED): count distinct documents, because repeated extractor errors inflate support; declare functionality per property, because single-truth assumptions break non-functional predicates; hierarchical values are not contradictions.
- **Only functional properties (SHACL `sh:maxCount 1`; OWL 2 QL lacks functionality) produce contradictions.** These become conflict sets for review, never auto-resolved. Qualifiers separate observation time from validity time.
- **Collective entity–relation resolution, bounded:** at most one extra round, relational features from accepted edges only, UC anchors as fixed seeds (SiGMa-style). Relations alone never accept a merge.
- **Warehouse relations are schema-level properties with a cardinality** (FKs, metric-view joins, verified and query-history joins), emitted as overlays and R2RML. They are reconciled with text predicates by the same matcher, as alignment proposals.
- **Fabrication gates are the input contract.** Only grounded, entailment-checked mentions count as support; aggregation cannot fix systematic extraction errors.
- **Evaluate three levels:** canonicalisation (macro, micro, pairwise F1; redundancy), typing (ontology conformance, relation hallucination), edges (sampled precision with an error bound; ECE). No benchmark is German.

## 0. Current state (read from the tree)

- **`_resolve_label`.** An alias hit wins. Otherwise the closest canonical by `token_similarity` (word-set Jaccard, underscore-split) is taken at ≥0.6.
  - `not_located_in` folds into `located_in` (2/3), so negation collapses.
  - `employed_by` against `works_for` scores 0, so the paraphrase is missed.
  - The rung also ignores direction and endpoint types.
- **The inverse map** rewrites the label and swaps head and tail, recording `inverted`. This is sound and worth keeping.
- **The `allowed_labels` filter** returns `None`, so an out-of-schema relation is dropped and counted. This loses the signal for a new property.
- **`NormalizedRelation`** passes the extractor's `confidence` through uncalibrated. A relation with an unresolved endpoint is dropped (`stages/disambiguation.py`).
- **`GraphBuilder._build_edges`** produces one edge per `NormalizedRelation`, with no aggregation and no support statistic. `Evidence.mention_ids` holds the two endpoint mentions, so the export's `lg:evidenceCount` always equals 2.
- **`RelationMention.raw_label` already exists** (T3), which is the hook for audit.

## 1. Canonicalising open relation phrases

| Method | Idea | Reported quality | Licence | Source |
|---|---|---|---|---|
| CESI (WWW 2018) | side information (PPDB, morphology, KB links, AMIE rules) → embeddings → HAC, for NPs and RPs | best in 7/9 NP settings (secondary). ReVerb45K has no RP gold, so RP results were judged manually | code Apache-2.0 (GitHub) | [repo](https://github.com/malllabiisc/cesi), [arXiv 1902.00172](https://arxiv.org/abs/1902.00172) |
| SIST (ICDE 2019) | side information from the source text | ReVerb45K and NYTimes2018; numbers UNVERIFIED | UNVERIFIED | [data page](https://heathersherry.github.io/ICDE2019_data.html), [dblp](https://dblp.uni-trier.de/rec/conf/icde/LinC19.html) |
| JOCL (SIGMOD 2021) | factor graph that jointly canonicalises and links to a curated KB | beats baselines on average F1 / accuracy (abstract) | UNVERIFIED | [arXiv 2212.01207](https://arxiv.org/abs/2212.01207) |
| CMVC / CMVC+ (KDD 2022 / TKDE 2025) | fact view + context view, multi-view k-means weighted by the CH index, unsupervised | OPIEC59K average F1 0.769 against 0.677 for JOCL (UNVERIFIED, secondary) | UNVERIFIED | [arXiv 2206.11130](https://arxiv.org/abs/2206.11130) |
| EDC (EMNLP 2024) | open extraction → LLM writes a definition per relation → embed the definition → retrieve schema candidates → LLM verifies. Modes: target alignment and self-canonicalisation | target alignment, partial F1: WebNLG 0.783, REBEL 0.546; with the schema retriever 0.820 / 0.601. Self-canonicalisation: 200 relations on WebNLG at precision 0.956 (all UNVERIFIED, secondary) | **MIT** (GitHub). Default embedder e5-mistral-7b; canonicaliser GPT-3.5 | [repo](https://github.com/clear-nus/edc), [paper](https://aclanthology.org/2024.emnlp-main.548/) |
| KGGen (NeurIPS 2025) | LLM triples, then iterative LLM clustering of nodes and edges (`kg.cluster()`) | on par with GraphRAG on MINE, with better sparsity | **MIT** | [repo](https://github.com/stair-lab/kg-gen) |
| AutoSchemaKG (2025) | extraction, then conceptualisation-based schema induction | 92–95% alignment with human schemas (vendor, per report 02) | **MIT** | [repo](https://github.com/HKUST-KnowComp/AutoSchemaKG) |
| TRACE-KG (ICML 2026) | keeps the raw and canonical predicate, induces RelCan → RelCls → RelClsGroup, with qualifiers and provenance | 97–99% schema coverage on DBpedia-WebNLG (UNVERIFIED, overview page) | UNVERIFIED | [arXiv 2604.03496](https://arxiv.org/abs/2604.03496) |

**Takeaway.** The LLM pipelines converge on one shape: **cluster or retrieve cheaply, then let a model verify the small set of distinct strings**, keeping the raw predicate beside the canonical one (our `raw_label`). With a label-driven extractor (GLiNER-Relex) raw labels come from a configured set; truly open strings appear only in the LLM-extractor arm and in induction (report 02).

## 2. Schema-guided typing when an ontology exists

**The pattern.** Every canonical property (report 05: an opaque `prop/P<ulid>`) carries:

- `skos:prefLabel` and `altLabel` in DE and EN, and a `skos:definition`;
- domain and range classes;
- an inverse;
- characteristics: functional (SHACL `maxCount 1`), symmetric, irreflexive.

A signature is typed in five steps:

1. **Rules.** Exact alias and the inverse map are must-map.
2. **Candidates.** Embedding retrieval over property labels and definitions. EDC embeds an LLM-written definition of the *extracted* relation; a Wikidata-grounded pipeline ([arXiv 2412.20942](https://arxiv.org/abs/2412.20942)) matches LLM-written relation descriptions to property descriptions, has an LLM vet the top match, and infers domain and range for novel relations (results UNVERIFIED).
3. **Decision.** A calibrated scorer (cross-encoder) decides; an LLM *select* over a shuffled shortlist runs only in the review band (report 04 §2).
4. **Vetoes.** Domain and range are checked against the endpoints' canonical types with subclass closure; a violation is a veto, not a penalty. Direction is chosen with the property (`has_X` / `X_of` is one property plus its inverse).
5. **Unmapped signatures** become `OntologyProposal(kind=property)` with support (distinct documents), the head/tail class distribution, the nearest property and its margin, and example locators.

ODKE+ (2025) reports production use of ontology-guided prompting over 195+ predicates, with a lightweight-LLM grounder, at 98.8% precision. That is a deployment claim with an unknown sampling method (UNVERIFIED; [paper](https://arxiv.org/abs/2509.04696)).

Wikidata's property-constraint system (single-value, type, value-type, inverse and others) is a working model for per-property vetoes ([portal](https://www.wikidata.org/wiki/Help:Property_constraints_portal); individual pages not checked).

## 3. Edge aggregation, truth discovery, contradictions and time

- **Knowledge Vault** (KDD 2014) fuses extractor outputs with a boosted-stump classifier. Its features are per-extractor confidence and the count of supporting documents, and Platt scaling gives calibrated fact probabilities (UNVERIFIED, slide and survey excerpts) ([pub](https://research.google/pubs/knowledge-vault-a-web-scale-approach-to-probabilistic-knowledge-fusion/)).
- **Knowledge fusion** (VLDB 2014; 1.6B triples, 12 extractors): PopAccu+ is "reasonably well calibrated" and triples scored above 0.9 were 94% accurate. Of 20 sampled false positives, 8 were extraction errors repeated across pages, 13 multiple truths under a single-truth assumption, 7 hierarchical values (overlapping). All UNVERIFIED, excerpt ([arXiv 1503.00302](https://arxiv.org/abs/1503.00302)).
- **Truth discovery** estimates source reliability jointly with truth ([survey](https://arxiv.org/abs/1505.02463)). The multi-truth variants (Zhao et al. PVLDB 2012; Wang et al. CIKM 2015/2016) drop the one-value assumption ([CIKM 2016](https://researchers.mq.edu.au/en/publications/empowering-truth-discovery-with-multi-truth-prediction/)).
- **LLMs handle conflicting evidence poorly** without explicit credibility ([CONFACT, IJCAI 2025](https://www.ijcai.org/proceedings/2025/1073.pdf)), so no LLM arbitrates conflicts.
- **Time.** ATOM separates observation from validity time and end-dates superseded facts ([arXiv 2510.22590](https://arxiv.org/abs/2510.22590)); PaTeCon mines temporal constraints for conflict detection ([arXiv 2312.11053](https://arxiv.org/abs/2312.11053)).

**For v1:**

1. **Fact key** `(head entity id, property IRI, tail entity id, qualifier key)`; qualifiers `valid_from`, `valid_to`, `observed_at` (document date).
2. **Aggregation.** Per-document max over *grounded* mentions, then a deterministic noisy-OR across distinct documents with a near-duplicate discount, then a calibrator fitted on audited edges per property family. Noisy-OR is overconfident under correlated sources, so the calibrator is mandatory; a learned fusion model is a later Provider.
3. **Support on every edge:** mentions, distinct documents and sources, extractor ids, grounding pass rate, warehouse corroboration.
4. **Contradictions** exist only for functional properties whose validity intervals overlap or are unknown. Values related by subsumption or `part_of` are *compatible-hierarchical*. A conflict set is ranked by fused probability and reviewed; nothing is deleted, and only a decision marks a value `superseded`.

## 4. Collective entity and relation resolution

- **Bhattacharya & Getoor** (TKDD 2007) cluster with combined attribute and relational similarity. Similarities are updated iteratively as clusters merge, and this beats both attribute-only and non-collective relational baselines ([record](https://drum.lib.umd.edu/items/699e2e14-c5e3-48cf-aa9f-037dbef7a0e2)). Their query-time variant resolves in seconds with little accuracy loss (KDD 2006).
- **PARIS** (VLDB 2011) aligns instances, relations and classes jointly and probabilistically, with no tuning, at about 90% precision on large ontologies ([arXiv 1111.7164](https://arxiv.org/abs/1111.7164)). Its use of relation functionality as evidence is UNVERIFIED (not read).
- **SiGMa** (KDD 2013) starts from a high-precision seed and propagates greedily through the relation graph to millions of entities ([arXiv 1207.4525](https://arxiv.org/abs/1207.4525)).
- **JOCL** couples canonicalisation with linking (§1).

**Bounding it at scale:**

1. Relational features come only from edges accepted in the previous round: Jaccard of (property, neighbour-cluster) sets; functional-property agreement as positive evidence, disagreement as a **veto**.
2. Relations move a pair from review to accept only above an attribute-score floor; they never merge alone.
3. UC-anchored entities are fixed seeds (report 04, open question 6).
4. At most two deterministic rounds, re-scoring only review-band pairs in touched components; stop when no band changes. Cached pair scores (report 04 §8) keep round 2 at O(touched pairs).

## 5. Relations on the warehouse side

**Sources, in descending trust:**

1. **Declared PK and FK.** They are informational and not enforced. `RELY` means the owner asserts them, so they are still verified by inclusion ([Databricks](https://docs.databricks.com/aws/en/sql/user/queries/query-optimization-constraints); report 01).
2. **Metric-view `joins`**, which carry `cardinality` (spec 1.1).
3. **Joins verified by Metis's exact-key and inclusion checks.**
4. **Query-history joins** with support. `statement_text` is redacted by default.
5. **Column lineage.** It records derivation (`prov:wasDerivedFrom`), not a domain relation, and covers a best-effort subset ([lineage tables](https://docs.databricks.com/aws/en/admin/system-tables/lineage)).

Auto-BI predicts joins locally, then optimises the global join graph (k-MCA). It reports F1 above 0.9 on 100k+ real BI models ([VLDB 2023](https://www.vldb.org/pvldb/vol16/p2578-he.pdf)). Report 03 has the join-inference detail.

**Mapping.** An FK `orders.customer_id → customers.id` gives a property candidate with domain `Order` and range `Customer` (the classes bound to the tables), functionality from many-to-one (`sh:maxCount 1`), a name and definition drafted from the constraint, column and comments, and actor `uc`. It is emitted as an overlay, R2RML mapping and proposal; instances stay in the warehouse (OBDA, report 05).

**Reconciliation with text.** The §2 matcher, plus a domain/range signature check, pairs text predicates with warehouse properties as SSSOM alignment proposals. Grounded text mentions then count as *support* for the warehouse property. A text claim against declared cardinality ("an order has several customers" against a many-to-one FK) opens a conflict for review; UC never loses silently.

## 6. Relation fabrication as input quality

Report 02 recommends offset-anchored triples plus entailment. In the repo's judged sample, 14% of relations were incorrect and 32% partial, and the high-confidence tail was the least reliable band.

Published practice:

- GraphEval and GraphEval+ check each triple with NLI against the *nearest* source sentences, which avoids long premises ([arXiv 2512.00663](https://arxiv.org/abs/2512.00663)).
- PERSEUS adds retrieval plus an NLI gate, for +11% precision on CaRB ([CEUR](https://ceur-ws.org/Vol-4246/llms4kgoe-5.pdf)).
- ODKE+ uses a lightweight-LLM grounder (§2).

**For resolution:** record a grounding verdict on every mention (gate result, entailment score, verifier id and version); only grounded mentions add support, ungrounded ones stay as weak evidence; cross-sentence relations get a lower prior. A permissive multilingual NLI model is still to be found (UNVERIFIED).

## 7. Evaluation

- **Canonicalisation:** macro, micro and pairwise P/R/F1 over predicate clusters (CESI; COMBO), B-cubed, and schema size and redundancy against gold (EDC).
- **Typing:** ontology conformance and relation hallucination ([Text2KGBench](https://github.com/cenguix/Text2KGBench)), the veto rate and the unmapped share.
- **Edges:** sampled precision of accepted edges with an error bound, stratified by band and property. Cluster sampling cut annotation cost by up to 60% (static) and 80% (evolving) ([Gao et al., PVLDB 2019](https://arxiv.org/abs/1907.09657)). Add ECE of the fused probability, the contradiction rate, recall on a gold document set and, for the warehouse, FK recovery on held-out declared keys.

**Benchmarks:**

| Benchmark | Relation gold | Licence |
|---|---|---|
| ReVerb45K, Base, Ambiguous | no relation gold | data ships in the CESI repo (Apache-2.0); data licence UNVERIFIED |
| OPIEC59K | noun phrases only | UNVERIFIED |
| COMBO (EACL 2023) | gold relation-phrase clusters | repo Apache-2.0; data licence UNVERIFIED ([repo](https://github.com/jeffchy/COMBO)) |
| Text2KGBench | — | code Apache-2.0, data CC BY 4.0; but its WebNLG part is CC BY-NC-SA 4.0 and its TekGen part CC BY-SA 2.0 |
| REBEL | — | code CC BY-NC-SA 4.0 ([repo](https://github.com/Babelscape/rebel)); reference only |

None of them is German. The EN+DE gold set (report 02) should be extended with relation clusters and aggregated-edge judgements. Reviewer decisions grow it over time.

## Recommended v1 design

It sits after entity resolution (report 04) and reuses its scorer, calibration, bands, decision store and stable-id machinery.

| Stage | What it does | Contracts |
|---|---|---|
| **R0 Gate** | substring gate, then NLI on the uncertain band (report 02) | `RelationMention` + `grounding` |
| **R1 Endpoints** | map mentions to resolved entities (accepted band only). An unresolved endpoint is *kept and counted* as `unresolved`, not dropped | — |
| **R2 Typing** per distinct signature | rules (alias, inverse), then definition retrieval, then the scorer and calibration, then LLM select in the review band, then domain and range vetoes. Fuzzy Jaccard becomes a feature only | new `PredicateBinding` (signature hash → property IRI, direction, band, actor, Evidence); unmapped → `OntologyProposal(kind=property)` |
| **R3 Mention link** | per-mention typed link | `NormalizedRelation` gains `property_iri`, `binding_id`, `grounding`; Evidence lists the *relation* mention id |
| **R4 Aggregate** | group by fact key; per-document max, then fusion, then the calibrator; Learn-then-Test bands for edges | new `CanonicalEdge`, with stable `edge_id` = hash of (head stable id, property IRI, tail stable id, qualifier key), support statistics and member mention ids. `GraphEdge` becomes one per `CanonicalEdge` |
| **R5 Constraints** | functional, irreflexive and asymmetric checks per property; temporal overlap; hierarchical compatibility | new `ConflictSet` |
| **R6 Warehouse** | FK, metric-view joins, verified joins and query history → property candidates; reconcile with text via R2 | `OntologyProposal(property / alignment / binding)`, SSSOM |
| **R7 Collective round** (optional, ≤2) | accepted edges feed relational features to the entity review band; re-run R1–R5 on touched components | — |
| **R8 Review and export** | decisions keyed on signature hash (type / cannot-type) and fact key (accept / reject / supersede), replayed first. Export: one reifier named graph per `CanonicalEdge`, PROV actors | decision store; `Evidence.actor` |

All of this needs a `SCHEMA_VERSION` bump. Replay reads cached model answers keyed by input hash, model id and prompt hash. A delta re-aggregates only the fact keys whose members changed.

**Keep from `RelationNormalizer`:**

- `_norm_label`;
- `label_aliases`, as must-map rules;
- `inverses`, which become `owl:inverseOf` plus the direction swap;
- the `inverted` audit flag;
- the head-differs-from-tail validator;
- `allowed_labels` as the closed vocabulary, but **mapped to proposal-and-count, not drop**.

**Retire:**

- fuzzy-as-decider;
- confidence pass-through;
- one edge per mention;
- endpoint-only Evidence.

## Open questions

1. Do functionality and other characteristics come from humans only, or may they be induced from data (QSE-style, support ≥ k) and from FK cardinality?
2. Which fusion features and calibration slices apply (per property, or per property family)? What is the minimum audited set before an edge accept band opens?
3. Is a qualifier part of the fact key (one edge per validity interval), or does it annotate one edge?
4. Should text-only properties with high support auto-create a property, or always wait for review? The map's "no silent over-merge" suggests review.
5. Is the second collective round worth its cost? Measure the gain in entity pairwise F1 on the gold set before enabling it by default.
6. Which multilingual NLI and cross-encoder models with permissive licences cover German for R0 and R2?
7. Do Graph Completion (ULTRA) predictions enter R4 as a separate actor with zero document support, or stay outside canonical edges as proposals?
