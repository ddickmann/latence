# Research 03: the state of the art in schema matching, column-to-concept annotation and key/join discovery

Ticket: [03](../issues/03-sota-schema-matching.md). Research only, done 2026-11-08.

**How it was read.** The Metis side was read from the working tree, read-only. On the web side,
GitHub pages could be fetched, but arxiv.org, vldb.org and docs.databricks.com did not resolve
from this session. Paper numbers therefore come from search-engine summaries of the papers, not
from their result tables. A number that only one secondary summary gives, or that the summary
garbled, is marked **UNVERIFIED**. Licences come from the GitHub repository pages that were
fetched, and are UNVERIFIED wherever no repository was read.

## Summary

- **The field has settled on one shape: retrieve cheaply, then rerank or adjudicate expensively,
  under constraints.**
  - Matchers (Magneto, ReMatch, SCHEMORA) retrieve candidates with small embedding models or
    lexical scoring, then rerank them with an LLM.
  - Join and foreign-key finders (LLM-FK, DBAutoDoc, Scalable Join Inference) prune with
    statistics (uniqueness, containment), then have an LLM adjudicate the survivors.
  - The best enterprise result found (constraint-guided mapping, Aug 2026) attributes its quality
    to schema constraints that shrink the candidate set, not to the LLM.
- **Metis's layer job already holds the statistical spine**: in-source salted bottom-k sketches,
  blocking, exact inclusion checks within a source, key and history discovery, lexicon linking
  with abstention, and isotonic calibration on owner decisions. SOTA has nothing better for
  verifying a to-one key, and its calibration and abstention are ahead of every system read here.
- **What SOTA adds is mostly on the semantic side.** It reads names and comments as embeddings,
  not only as lexicon words. It uses an LLM to rerank or adjudicate a bounded candidate set. It
  mines query history for joins and filters. It optimises the join graph as a whole. Column-level
  lineage is used in practice to propagate terms, though research rarely measures that.
- **Treat benchmark numbers with caution.** They are English, mostly biomedical or web tables,
  and partly memorised: one 2026 preprint finds that LLMs recall 80–100 % of a public benchmark's
  target identifiers with no schema given. None tests German and English headers or abbreviations
  the way Latence needs.
- **Recommended v1:** keep Metis's statistical spine, keyed on UC ids. Add four signals, all of
  which read metadata only:
  - multilingual embeddings over names and comments;
  - UC's declared constraints;
  - query-history join and filter mining;
  - column lineage.

  Add an LLM adjudicator over metadata only, behind the egress decision. Defer learned column-type
  models, neural join search and LLM calls that see values.

## What Metis already does

Read from `metis-core/src/metis_core/layer/` and `contracts/catalog.py`.

- **Evidence, read narrowly** (`evidence.py`):
  - **Statistics per column:** type family, rows, distinct, nulls, lengths, shapes and grain.
  - **Sensitivity first:** sensitivity is decided before any value is read, from catalog tags,
    then e-mail and IBAN shapes, the length profile, personal domain entities, and as a reported
    fallback the lexicon. A sensitive column is never read, sketched or shown to a model.
  - **Values:** the top values of a column only when it is code-like (few distinct values).
  - **Sketches:** a bottom-k sketch of `md5(salt‖value)` only for an identifier-like column.
    The sketch is computed **in the source**, so no raw value leaves it, and one salt covers the
    whole estate.
- **Keys** (`keys.py`):
  - **Single-column keys:** a unique, non-null column is a key, and an identifier word ranks it
    as the primary key.
  - **Composite keys:** grown greedily, with at most four columns and a bounded number of checks.
  - **Histories:** an SCD2 history is a business key plus a validity start and end, verified
    unique on the key and its start, and with no overlapping versions of one key.
- **References** (inclusion dependencies, `keys.py`):
  - **Blocking:** a column is compared only with keys of its type family that share its leading
    shape or at least two sketch hashes.
  - **Containment:** estimated from the sketches. Within one source an exact inclusion check
    confirms it. Across sources a reference stays `unknown` when too few hashes are comparable.
  - **Rejections:** an integer sequence whose name shares nothing with the key's is rejected.
  - **Choice:** the same source first, then the key of the table the column's words name, then
    the smallest key. Losing candidates are kept as rejected records with the reason.
  - **Limit:** references are single-column only (`if len(key.columns) != 1: continue`).
- **Semantics** (`semantics.py`):
  - **Linking:** a column is linked to a concept of the domain library by the Jaccard of its
    header words, with a prefix-abbreviation rung (`lief` reads as `lieferant`) and a reading
    qualified by its table's name. Catalog comments count more weakly, and matches of its values
    against the concept's values count too.
  - **Guards:** type guards apply, and a link needs a minimum score and a margin over the next
    concept. Otherwise the column is recorded as `unknown` with its top three candidates.
  - **Propagation:** a reference takes its key's concept, and a key takes its table's entity.
  - **Folding and values:** exact header folding across tables, value-spelling folding
    (`Zürich`/`Zuerich`), and J/N-style code lists.
- **Joins and global constraints** (`resolution.py`): joins are added by priority. A join is held
  back, rejected with its reason, if it would give a table two shortest paths to another; a
  dimension column is held back the same way.
- **Calibration** (`calibration.py`): isotonic regression per proposal class on the owner's own
  decisions, gated by a Wilson lower bound and a minimum count of judgements.
- **Models:** only for drafting descriptions, under a budget. No model takes part in matching.
- **Measured on Metis's synthetic gate estate** (ticket 99): concept links 13/13 and cross-source
  links 7/7 on the legacy Postgres source; join recall 1.00 with precision 0.43, all 8 extras
  verified and accepted by the owner. Blocking compared 42 of 580 possible column pairs. One
  synthetic estate is no benchmark.
- **Not used for matching:**
  - query history;
  - lineage (the port's `CatalogLineage` is table-level, and the matcher doesn't read it);
  - declared PK/FK constraints (they aren't in `CatalogTable`);
  - embeddings.

**The open-source auto-ontology project** (review of 2026-11-03) infers foreign keys in two steps:
- **Proposal:** an LLM suggests keys from names, types and the top five values of a
  `SELECT * … LIMIT 1000` sample, or a vector search plus an LLM pick, falling back to a live
  value-match probe.
- **Pruning:** only zero-overlap foreign keys are pruned.
- **Query history:** it mines filters and aggregates from 24 hours of history with sqlglot.

It computes no cardinality, has no egress gate, and doesn't record its LLM calls. Metis is
already stricter on joins.

## Approaches

"Values?" says whether raw values must be read. *In-source* means they are read inside the
warehouse and only aggregates or hashes leave. *Egress* means values go to a model or service
outside it.

| Approach | Inputs | Reported quality (benchmark) | Cost | Licence | Values? |
|---|---|---|---|---|---|
| **Valentine** suite (COMA 3.0 in Python, Cupid, Similarity Flooding, distribution-based, Jaccard) [V] | names, optionally instances | About 75k experiments. No method wins everywhere; COMA's schema and instance variants trade wins by dataset (via SMUTF's table) [V] | CPU | Apache-2.0 (repo read) | optional, in-process |
| **LLM matching by task scope** (Parciak et al. 2024) [P] | names and descriptions only | MIMIC-IV→OMOP, 49 matches. GPT-4 beats GPT-3.5. Every scope except 1-to-1 beats the string baseline. The GPT-3.5 baseline 1-to-1 mean F1 is 0.241 (cited by a later paper; UNVERIFIED) | an LLM call per scope | paper artefacts | **no** |
| **ReMatch** [R] | names and descriptions, target-schema retrieval | MIMIC-III→OMOP and Synthea→OMOP, accuracy@k against SMAT. Its abstract claims no training, mappings or data are needed | embeddings plus an LLM | UNVERIFIED | **no** |
| **SCHEMORA** (2025) [S] | metadata enriched by an LLM, vector plus lexical retrieval | MIMIC-OMOP HitRate@5 80.4 % (ReMatch 72.9 %, Matchmaker 71.1 %), HitRate@1 54.3 % (Matchmaker 62.2 %). The paper's own HitRate@3 is inconsistent | GPT-4.1 plus text-embedding-3-large | code public, licence UNVERIFIED | **no** |
| **Magneto** (PVLDB 18, 2025) [M] | headers ± sampled values | GDC (over 700 target attributes) and Valentine. Beats COMA, ISResMat and Unicorn on MRR and Recall@GT; ISResMat and Unicorn don't finish GDC beyond 100 columns. No numbers read | small model retrieval (mpnet and others, optionally fine-tuned) plus an LLM rerank (gpt-4o-mini or llama-3.3-70b) | Apache-2.0 (repo read) | header-only mode exists; value modes send samples to the reranker |
| **Contamination check** (MetaHarmonizer, bioRxiv 2026) [MH] | names | 3 of 5 LLMs recall 80–100 % of GDC identifiers with no schema given; under a controlled protocol the LLM-only advantage reverses (authors' claim, UNVERIFIED) | — | UNVERIFIED | — |
| **Constraint-guided mapping** (Aug 2026) [CG] | schema constraints, candidate generation, neural ranker, bounded LLM | Enterprise data, 7 domains with about 12k targets each. One identifier constraint shrinks the candidates about 18× with near-full recall. Per-domain F1 depends on cardinality, not on the LLM. It also reports MRR and Recall@GT on Valentine (numbers not read) | bounded LLM calls | UNVERIFIED | partly |
| **Column-type annotation, learned**: Sherlock, Sato, DoDuo [CTA] | cell values (Sherlock: 1,588 features) | VizNet, 78 types: Sherlock F1 0.89 (support-weighted), Sato 0.925; DoDuo macro-F1 84.6, and on SOTAB micro-F1 84.8 | GPU training, cheap inference | Sherlock MIT, DoDuo Apache-2.0 (repos read) | **yes**; fixed web-type vocabularies |
| **Column-type annotation, LLM** (Korini & Bizer 2023, 2025) [KB] | values plus label definitions | SOTAB (91 types) 89.5 micro-F1 (leaderboard; the subset is unclear, UNVERIFIED). Refined definitions add 3.9 % F1 on average; self-refinement costs over 40× the generated tokens | per-column LLM calls | — | **yes, egress** |
| **LSH Ensemble** (VLDB 2016) [LE] | MinHash of values | Threshold-based containment search at Open Data scale. Precision@k 0.63–0.69 on DeepJoin's Webtable equi-join test (UNVERIFIED) | sublinear index | MIT (in `datasketch`, repo read) | in-source possible: hashes only |
| **JOSIE** (SIGMOD 2019) [J] | full value sets, inverted index | exact top-k overlap | Postgres plus Go; runs take hours to days | **no licence file** (repo read) | **yes** (an index of values) |
| **DeepJoin** (PVLDB 16) [DJ] | column text (names, stats, values) through a fine-tuned PLM, then ANN search | Average precision 72 % equi-join and 91 % semantic join. Beats WarpGate in a 2025 re-evaluation | GPU embedding plus ANN | UNVERIFIED | **yes** |
| **WarpGate** (CIDR 2023) [WG] | value embeddings plus random-projection LSH | Beats Aurum and D3L in precision and recall (authors' claim) | embedding service | UNVERIFIED | **yes** |
| **Starmie** (PVLDB 2023) [ST] | contrastive column encoder over values | Table union search: 6.8 MAP and recall points over the prior best on SANTOS and TUS | GPU pre-training | **no licence file** (repo read) | **yes** |
| **TabSketchFM / LakeBench** (2024) [TS] | sketches (MinHash, numeric) plus metadata into a transformer | Up to 55 % F1 over neural baselines on 8 LakeBench tasks | GPU | UNVERIFIED | sketches only (can be computed in-source) |
| **Freyja** (2024/26) [F] | column profiles, with a learned join-quality model | Open Data US and YADL. Profiles instead of pairwise joins | CPU | UNVERIFIED | profiles (aggregates) |
| **Joinability re-evaluation** (2025) [EJ] | — | 7 benchmarks. Only 42 of 471 high-containment OpenData pairs are joinable in context. Ensembles beat single criteria. Size criteria matter more in relational databases | — | — | — |
| **IND discovery**: SPIDER, BINDER, FAIDA, SINDY [B] | all values | BINDER is up to 26× faster than SPIDER (authors). On 79 GB, BINDER takes 9.5 h and approximate FAIDA 1.8 h | heavy, exact or approximate | Metanome Apache-2.0 (repo read); algorithm licences UNVERIFIED | **yes**, in-engine |
| **HoPF** (JIIS 2020), Rostin 2009 [H] | INDs plus name, position and distribution features | About 88 % of PKs and 91 % of FKs on 3 benchmarks and 2 real datasets (search summary; UNVERIFIED) | CPU | UNVERIFIED | aggregates |
| **Auto-BI** (VLDB 2023) [AB] | metadata plus value overlap per column pair; a global k-MCA graph | Over 0.9 F1 on 100k real BI models and TPC; sub-second for about 100 tables | CPU after training | no public code found | overlap stats |
| **LLM-FK** (ACL Findings 2026) [LF] | profiler, interpreter, refiner and verifier agents; unique-key-driven pruning | F1 over 93 % on 5 datasets; +15 % on MusicBrainz; 2–3 orders of magnitude fewer candidates | many LLM calls | UNVERIFIED | profiles; whether values reach the LLM is UNVERIFIED |
| **DBAutoDoc** (2026) [DA] | cardinality, inclusion, naming, deterministic gates, then LLM review | AdventureWorks with constraints stripped: LLM-proposed FKs 89 % precision (75/84). The statistical pipeline adds 23 F1 points over LLM-only FK detection | LLM rounds | open source, licence UNVERIFIED | statistics in-engine |
| **Scalable Join Inference** (2026) [SJ] | statistics pruning, LLM adjudication, **query-history join paths** | Precision 78–100 % on well-structured schemas (TPC-DS, TPC-H, **BIRD-Dev**, production); poorly normalised schemas are hard. Recall not found | roughly linear in the candidates | UNVERIFIED | statistics plus query text |
| **Query-log join weighting** (2019) [QL] | SQL logs | Logged joins weight a Steiner-tree join path (NL interfaces) | parse only | — | **no** (query text, though literals may hold values) |
| **Lineage propagation** in catalogs [LP] | column lineage | No precision reported. DataHub, Atlas and Atlan propagate terms and tags opt-in, with depth caps and no overwrite by default | cheap | various | **no** |
| **Schema-naming effect** (SNAILS, SIGMOD 2025) [SN] | identifiers by naturalness level | Schema linking degrades on abbreviated ("least natural") names; larger models are less sensitive | — | — | **no** |

**What Databricks offers natively, from search summaries of the docs (UNVERIFIED in detail):**
- **Declared keys:** UC PK/FK constraints are informational, not enforced. `RELY` lets the
  optimiser use them, and its use is the user's risk.
- **Column lineage:** `system.access.column_lineage` keeps one year. It is a subset, emitted only
  where lineage can be inferred, and joins to `system.query.history` on `statement_id`.
- **Query history:** `statement_text` there is redacted unless the reader is an account admin or
  in a PII-access group, and it is empty under customer-managed keys.
- **Generated metadata:** UC has data classification, which auto-tags PII, and AI-generated
  comments. Whether the comments see sample values is UNVERIFIED.

## What the state of the art adds beyond Metis

1. **Semantic similarity for names and comments.**
   - **Metis today:** exact-word Jaccard against the library's lexicon. It misses paraphrase,
     cross-language synonyms the lexicon lacks, and descriptive comments.
   - **SOTA:** a small embedding retriever over name plus comment, in the style of Magneto and
     SCHEMORA, recovers these, and needs no values. Latence keeps its embedders as a resolution
     signal (ADR-0068), so this is the cheapest gain.
2. **A bounded LLM adjudicator.**
   - **What it does:** reranks the top-k concept candidates for a column, or judges a key-pair
     candidate. Every strong 2025–26 system uses one, and always after pruning.
   - **Metis today:** emits `unknown` with three candidates. An adjudicator would turn part of
     those into proposals, as long as each one stays a recorded, replayable decision whose actor
     is the model.
3. **Query history as evidence.**
   - **Joins:** the joins analysts actually write are direct evidence of a relationship, and of
     its direction and kind.
   - **Conventions:** recurring `WHERE` exclusions are evidence of catch-all members and
     null-key conventions.
   - **Measures and dimensions:** recurring `GROUP BY` and aggregate columns are evidence of them.
   - **Metis and the auto-ontology project:** Metis has none of this. The auto-ontology project
     mines it, but with a broken usage score.
4. **Column lineage as evidence.** A downstream column that is an identity or rename of an
   upstream one inherits its concept and keyness. That makes it a strong, value-free signal on
   Databricks, where lineage is native.
5. **Declared constraints and existing metadata.** Read UC PK/FK, tags, AI-generated comments and
   metric-view definitions as prior evidence, verified against the data, never trusted blindly.
   Metis's catalog port carries none of the constraints.
6. **Global join-graph optimisation.**
   - **Auto-BI's approach:** a k-MCA over pairwise probabilities.
   - **Metis today:** greedy priority plus a shortest-path-uniqueness veto. That suffices at
     Metis's scale, but it can lock in an early wrong edge.
7. **Multi-column foreign keys.** Metis finds composite keys but links only single-column
   references. HoPF and BINDER handle n-ary INDs.
8. **Fuzzy and semantic joins** (DeepJoin, WarpGate). These are joins on non-identical
   representations, such as names against codes. They need value embeddings, so they mean egress
   or an in-warehouse model, and they suit data-lake discovery more than a governed warehouse.
9. **Annotation against an ontology's types, not only library concepts.** Learned annotators
   predict fixed web vocabularies (DBpedia, Schema.org) from values. That is useful only if an
   ontology like Schema.org is in scope, and it needs values.

Where Metis is already **ahead** of everything read: sensitivity before reading, in-source
salted sketches, exact verification within a source, histories, abstention with recorded
candidates, and calibration on owner decisions. The auto-ontology project and most papers have
neither abstention nor calibration. The constraint-guided result confirms Metis's premise that
constraints, not model size, carry precision.

## Recommended signal stack for v1

All signals are evidence on a proposal, with an actor and a locator keyed on UC ids. Calibration
is per proposal class, as in Metis.

**Tier 0: metadata only, no egress, always on**
1. **UC structure:** names, types, comments, tags, declared PK/FK and `RELY`, metric-view
   definitions, and the partition columns.
2. **Lexicon linking:** keep Metis's design (Jaccard, the prefix rung, table qualification,
   margin, abstention), in German and English.
3. **Embedding similarity:** a multilingual sentence model over `table.column`, name plus
   comment, against concept names, synonyms and definitions. It is a retrieval signal and a
   feature, never the decider alone. It needs a pinned, permissively licensed checkpoint (still
   to be chosen; see the open questions).
4. **Column lineage** from `system.access.column_lineage`:
   - identity edges propagate a concept as a proposal, with depth capped and no overwrite;
   - lineage between two tables raises the prior of a join between them.
5. **Query-history mining** with a deterministic SQL parser over `system.query.history`. It
   yields join pairs with frequencies, filter literals and aggregate columns. The literals in
   statement text are values, so they need the same sensitivity gate as values, and the text
   needs an admin or PII-group grant.

**Tier 1: aggregates and hashes computed in the warehouse**
6. **Profiles:** Metis's statistics, shapes, code-like top values (gated by sensitivity first)
   and salted bottom-k sketches, run as SQL on the warehouse.
7. **Key and inclusion verification:** exact within a catalog. Across catalogs, by sketch
   containment, abstaining when hashes are too few. Extend it to two-column FKs, bounded.
8. **The join-graph constraint:** keep the shortest-path-uniqueness veto. Add a cardinality
   check before a join is proposed, as Metis already does.

**Tier 2: an LLM over metadata, behind the egress decision, off by default**
9. **An adjudicator** over the top-k candidates from tiers 0 and 1, for columns left `unknown`
   and for key pairs that survive pruning. Its prompt carries names, comments, types,
   statistics and candidate concepts, and **no values**. Its output is a recorded proposal with
   the model as actor, replayed from the record.

**Defer**
- Learned column-type annotators (Sherlock, Sato, DoDuo). They need values and web
  vocabularies, and their fit to business concepts is unproven.
- An LLM that sees sample values, which needs an explicit egress decision per estate.
- Neural join search (DeepJoin, WarpGate, Starmie, TabSketchFM): a lake-discovery problem, it
  needs values and a GPU, and some repositories have no licence.
- Full n-ary IND discovery (BINDER, FAIDA) and a global k-MCA solver. Revisit them when an estate
  outgrows the greedy approach.
- Text-path ontology alignment against these signals. That belongs to the resolution-engine slice.

**Evaluation:** build a held-out estate with German and English abbreviated headers in the style
of SNAILS, and gold concept and join labels. Report precision and recall per class, abstention
rate and owner effort. Don't rely on MIMIC/OMOP or GDC alone, given the contamination finding.
Valentine (Apache-2.0) and the BIRD-Dev schemas, with their FKs stripped, give cheap
reproducible English baselines.

## Open questions

1. **Egress.** May tier 2 run at all, and against which endpoint? Is query-statement text, which
   holds literals, treated as values for egress purposes?
2. **Grants.** Will deployments grant the PII-access group or admin rights needed to read
   `statement_text`? If not, query mining is limited to lineage and table-level usage.
3. **Embedding model.** Which multilingual checkpoint: its weights licence verified, pinned, CPU
   viable? Embedding outputs must replay byte-identically, which needs a determinism story
   (thread pinning, as Latence's harness already requires).
4. **Concept ids.** Does the adjudicator's candidate set come only from the domain library, or
   also from concepts the text path induces? This interacts with "no silent over-merge".
5. **Declared constraints.** Is a UC PK/FK that fails verification a drift proposal upstream, or
   evidence that is ignored?
6. **Cross-workspace sketches.** One salt per estate or per tenant: what are the
   cross-metastore link semantics when salts differ?
7. **UNVERIFIED numbers.** Should the key numbers above be re-read from the papers once arxiv.org
   and vldb.org are reachable? This matters especially for Magneto, ReMatch, HoPF and Scalable
   Join Inference's recall.

## Sources

- [V] Valentine: https://github.com/delftdata/valentine; paper arXiv 2010.07386; COMA comparison
  via SMUTF, arXiv 2402.01685.
- [P] Parciak et al., Schema Matching with LLMs: an Experimental Study, TaDA@VLDB 2024, arXiv
  2407.11852.
- [R] ReMatch, arXiv 2403.01567.
- [S] SCHEMORA, arXiv 2507.14376.
- [M] Magneto, PVLDB 18(8) 2025, arXiv 2412.08194; https://github.com/VIDA-NYU/magneto-matcher;
  https://github.com/VIDA-NYU/data-harmonization-benchmark.
- [MH] MetaHarmonizer, bioRxiv 10.64898/2026.06.13.732088.
- [CG] Constraint-Guided Enterprise Data Mapping with LLMs, arXiv 2608.24218.
- [CTA] https://github.com/mitmedialab/sherlock-project; Sato, arXiv 1911.06311;
  https://github.com/megagonlabs/doduo (SIGMOD 2022).
- [KB] Korini & Bizer, arXiv 2306.00745 and 2503.02718 (ADBIS 2025); SOTAB leaderboard,
  hyper.ai.
- [LE] LSH Ensemble, PVLDB 9(12) 2016, arXiv 1603.07410; https://github.com/ekzhu/datasketch.
- [J] JOSIE, SIGMOD 2019; https://github.com/ekzhu/josie.
- [DJ] DeepJoin, PVLDB 16, arXiv 2212.07588.
- [WG] WarpGate, CIDR 2023, arXiv 2212.14155.
- [ST] Starmie, PVLDB 2023, arXiv 2210.01922; https://github.com/megagonlabs/starmie.
- [TS] TabSketchFM, NeurIPS 2024 TRL workshop, arXiv 2407.01619.
- [F] Freyja, arXiv 2412.06637.
- [EJ] Evaluating Joinable Column Discovery Approaches for Context-Aware Search, arXiv 2510.24599.
- [B] BINDER, VLDB 2015 (HPI); FAIDA and SINDY (Kruse et al.);
  https://github.com/HPI-Information-Systems/Metanome.
- [H] HoPF, J. Intell. Inf. Syst. 2020 (HPI); Rostin et al., WebDB 2009.
- [AB] Auto-BI, VLDB 2023, arXiv 2306.12515.
- [LF] LLM-FK, arXiv 2603.07278; ACL Findings 2026.
- [DA] DBAutoDoc, arXiv 2603.23050.
- [SJ] Scalable Join Inference for Large Context Graphs, arXiv 2603.04176.
- [QL] Bridging the Semantic Gap with SQL Query Logs in NLIDBs, arXiv 1902.00031.
- [LP] DataHub glossary-term propagation docs; Apache Atlas classification propagation (Cloudera
  docs); Atlan tag-propagation docs.
- [SN] SNAILS, SIGMOD 2025; https://github.com/kyleluoma/SNAILS.
- Databricks: the docs on constraints, `system.access.column_lineage`, `system.query.history` and
  data governance (search summaries only).
- The auto-ontology review: the 2026-11-03 research holding (a reference read, not cited in
  shipped docs).
