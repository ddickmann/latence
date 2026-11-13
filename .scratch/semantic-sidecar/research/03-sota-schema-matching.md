# Research 03: the state of the art in schema matching, column-to-concept annotation and key/join discovery

Ticket: [03](../issues/03-sota-schema-matching.md). Research only, 2026-11-08.

**Method.** Metis was read from its working tree, read-only. From this session, GitHub pages
could be fetched, but arxiv.org, vldb.org and docs.databricks.com did not resolve. Paper numbers
therefore come from search-engine summaries, not from the papers' own tables. A number resting
on one secondary summary, or one the summary garbled, is marked **UNVERIFIED**. A licence is
stated only where its repository page was read; every other licence is UNVERIFIED.

## Summary

- **The field has one shape: cheap retrieval or statistical pruning, then an expensive rerank or
  adjudication, all under constraints.**
  - Matchers (Magneto, ReMatch, SCHEMORA) retrieve with small embedders or lexical scoring, then
    rerank with an LLM.
  - Foreign-key and join finders (LLM-FK, DBAutoDoc, Scalable Join Inference) prune by
    uniqueness and containment, then an LLM adjudicates what survives.
  - The strongest enterprise result found credits **constraints**, not the LLM, for its
    precision.
- **Metis's layer job already has the statistical spine**, and it is ahead on governance. It
  holds in-source salted sketches, blocking, exact inclusion checks, key and history discovery,
  and lexicon linking that abstains. It also calibrates on owner decisions, which no system read
  here does.
- **The state of the art adds mostly semantic and usage signals:**
  - embeddings over names and comments;
  - a bounded LLM adjudicator;
  - joins and filters mined from query history;
  - column lineage;
  - declared constraints;
  - optimising the join graph as a whole.
- **Benchmarks are weak guides.** They are English, mostly biomedical or web tables, and partly
  memorised: LLMs recall 80–100 % of one benchmark's target identifiers with no schema shown.
  None tests German and English abbreviated headers.
- **v1 recommendation:** keep Metis's spine, keyed on UC ids. Add four signals that need no value
  egress: multilingual embeddings, UC constraints, column lineage and query-history mining. Add
  an LLM adjudicator over metadata only, behind the egress decision. Defer anything that sends
  values to a model, and defer neural join search.

## What Metis already does

From `metis-core/src/metis_core/layer/` and `contracts/catalog.py`:

- **Evidence** (`evidence.py`):
  - **Sensitivity first.** Before anything is read, a column's sensitivity is settled: catalog
    tags first, then value shapes (e-mail, IBAN), the length profile and personal domain
    entities, and a reported lexicon fallback last. A sensitive column is never read, sketched
    or shown to a model.
  - **What is read.** Statistics and shapes for every column, the top values of a code-like
    column, and a bottom-k sketch of `md5(salt‖value)` for an identifier-like column.
  - **Where.** The sketch is computed **in the source** with one salt for the whole estate.
- **Keys** (`keys.py`):
  - single-column keys;
  - composite keys, grown greedily (at most 4 columns, with bounded checks);
  - SCD2 histories, verified unique on business key plus start, with no overlapping versions.
- **References** (`keys.py`):
  - **Blocking:** candidates share a type family plus a leading shape or two or more sketch hashes.
  - **Containment:** estimated from the sketches. An exact check confirms it within a source;
    across sources, too few comparable hashes leave it `unknown`.
  - **Rejected:** an integer "sequence" whose name shares nothing with the key.
  - **Choice:** same source first, then the key of the table the column's words name, then the
    smallest key. The losers are kept as rejected records.
  - **Single-column only** (`if len(key.columns) != 1: continue`).
- **Semantics** (`semantics.py`):
  - **Linking:** header-word Jaccard to library concepts, with a prefix-abbreviation rung and
    table qualification. Comment words count weakly; value matches count.
  - **Guards and abstention:** type guards, a minimum score and a margin. Otherwise the column is
    `unknown` with its top-3 candidates.
  - **Propagation:** a reference takes its key's concept, and a key takes its table's entity.
  - **Folding:** exact header folding across tables, value-spelling folding, and J/N code lists.
- **Joins** (`resolution.py`): joins are added greedily by priority. One is vetoed when it would
  give a table two shortest join paths to another.
- **Calibration** (`calibration.py`): isotonic regression on owner decisions per class, with a
  Wilson lower bound and a minimum count of judgements.
- **Model use:** models draft descriptions only. None takes part in matching.
- **Not used for matching:** query history, lineage (the port's `CatalogLineage` is table-level
  and unused by matching), declared PK/FK (absent from `CatalogTable`), and embeddings.
- **Self-measured on one synthetic estate** (ticket 99):
  - concept links 13/13 and cross-source links 7/7;
  - joins: recall 1.00 and precision 0.43, with all 8 extras verified and accepted;
  - blocking compared 42 of 580 column pairs.

**The open-source auto-ontology project** (review of 2026-11-03):
- **How it infers foreign keys:** an LLM proposes them from names, types and the top-5 values of
  a 1,000-row sample; alternatively a vector search plus an LLM pick, then a live value probe.
- **What it prunes:** only foreign keys with zero value overlap.
- **Query history:** it mines filters and aggregates from 24 h of history with sqlglot.
- **What it lacks:** cardinality checks, an egress gate and recorded LLM calls.
- **Verdict:** Metis is already stricter on joins.

## Approaches

"Values?": *in-source* means only aggregates or hashes leave the warehouse; *egress* means
values reach a model or service outside it.

| Approach | Inputs | Reported quality (benchmark) | Cost | Licence | Values? |
|---|---|---|---|---|---|
| **Valentine** suite (COMA, Cupid, Similarity Flooding, distribution, Jaccard) [V] | names, ± instances | About 75k experiments; no method wins everywhere, and COMA's schema and instance variants trade wins by dataset | CPU | Apache-2.0 | optional, local |
| **LLM matching by task scope** (2024) [P] | names and descriptions | MIMIC-IV→OMOP (49 matches): GPT-4 beats GPT-3.5, and every scope but 1-to-1 beats the string baseline | one LLM call per scope | artefacts public | **no** |
| **ReMatch** [R] / **SCHEMORA** [S] | metadata (LLM-enriched in SCHEMORA), retrieval plus LLM | MIMIC→OMOP HitRate@5: SCHEMORA 80.4 %, ReMatch 72.9 %. HitRate@1: SCHEMORA 54.3 %, Matchmaker 62.2 % | embeddings plus GPT-4.1 | UNVERIFIED | **no** |
| **Magneto** (PVLDB 18) [M] | headers ± sampled values | GDC (over 700 targets) and Valentine: best on MRR and Recall@GT; ISResMat and Unicorn fail beyond 100 columns. Numbers not read. **Contamination:** under a controlled protocol the LLM advantage on GDC reverses (bioRxiv 2026, UNVERIFIED) [MH] | small-model retrieval plus an LLM rerank | Apache-2.0 | header-only mode; value modes send samples |
| **Constraint-guided mapping** (Aug 2026) [CG] | admissibility constraints, a neural ranker, a bounded LLM | 7 enterprise domains, about 12k targets each: one identifier constraint cuts candidates about 18× at near-full recall. F1 tracks cardinality, not the LLM | bounded LLM | UNVERIFIED | partly |
| **Learned column types**: Sherlock, Sato, DoDuo [CTA] | cell values | VizNet (78 types): Sherlock 0.89 and Sato 0.925 weighted F1. DoDuo: macro-F1 84.6, and SOTAB micro-F1 84.8 | GPU training | Sherlock MIT, DoDuo Apache-2.0 | **yes**; web vocabularies |
| **LLM column types** (Korini & Bizer) [KB] | values plus label definitions | SOTAB (91 types): 89.5 micro-F1 (UNVERIFIED). Refined definitions add +3.9 % F1 at over 40× the generated tokens | per-column LLM | — | **egress** |
| **LSH Ensemble** (VLDB 2016) [LE] | MinHash | Containment search at Open Data scale | sublinear | MIT (`datasketch`) | hashes only |
| **JOSIE** (SIGMOD 2019) [J] | value sets, an inverted index | exact top-k overlap | Postgres plus Go; runs take hours to days | **no licence file** | **yes** |
| **DeepJoin** [DJ] / **WarpGate** [WG] | column text or values through a PLM; ANN or LSH | DeepJoin AP 72 % on equi-joins and 91 % on semantic joins; it beats WarpGate in a 2025 re-evaluation [EJ] | GPU | UNVERIFIED | **yes** |
| **Starmie** [ST] / **TabSketchFM** [TS] | a contrastive encoder over values / sketches plus metadata | Starmie +6.8 MAP on SANTOS and TUS (union search). TabSketchFM up to +55 % F1 on LakeBench | GPU | Starmie: **no licence file** | yes / sketches only |
| **Joinability re-evaluation** (2025) [EJ] | — | Only 42 of 471 OpenData pairs with high containment are joinable in context. Ensembles beat single criteria | — | — | — |
| **IND discovery**: SPIDER, BINDER, FAIDA [B] | all values, in-engine | BINDER is up to 26× faster than SPIDER. On 79 GB: BINDER 9.5 h, approximate FAIDA 1.8 h | heavy | Metanome Apache-2.0 | **yes**, in-engine |
| **HoPF** (2020) [H] | INDs plus name, position and distribution features | About 88 % of PKs and 91 % of FKs (UNVERIFIED) | CPU | UNVERIFIED | aggregates |
| **Auto-BI** (VLDB 2023) [AB] | pairwise join models plus a global k-MCA graph | F1 over 0.9 on more than 100k real BI models and TPC; sub-second for about 100 tables | CPU | no public code found | overlap statistics |
| **LLM-FK** (2026) [LF] | profiler, interpreter, refiner and verifier agents | F1 over 93 % on 5 databases; +15 % on MusicBrainz; candidates cut by 2–3 orders of magnitude | many LLM calls | UNVERIFIED | profiles; whether values reach the LLM is UNVERIFIED |
| **DBAutoDoc** (2026) [DA] | statistics and deterministic gates, then LLM review | AdventureWorks without constraints: 89 % precision on LLM-proposed FKs. The statistical pipeline adds 23 F1 points over LLM-only | LLM rounds | UNVERIFIED | statistics in-engine |
| **Scalable Join Inference** (2026) [SJ] | statistics pruning, LLM adjudication, **query-history join paths** | precision 78–100 % (TPC-DS, TPC-H, BIRD-Dev, production); recall not found | linear in candidates | UNVERIFIED | statistics plus query text |
| **Query-log join weighting** [QL] | SQL logs | Logged joins weight the Steiner-tree join paths | parse only | — | literals only |
| **Lineage propagation** [LP] | column lineage | No precision reported. DataHub, Atlas and Atlan propagate terms opt-in, with depth caps and no overwrite | cheap | various | **no** |
| **SNAILS** (SIGMOD 2025) [SN] | identifiers by naturalness level | Schema linking degrades on abbreviated names; larger models are less sensitive | — | — | **no** |

**Databricks natively** (from search summaries of its docs, UNVERIFIED in detail):
- **Declared keys:** UC PK/FK are informational and not enforced. `RELY` lets the optimiser use
  them, at the user's risk.
- **Column lineage:** `system.access.column_lineage` is a one-year, best-effort subset. It joins
  to `system.query.history` on `statement_id`.
- **Statement text:** `statement_text` is redacted unless the reader is an account admin or in a
  PII-access group.
- **Generated metadata:** UC also offers PII auto-classification and AI-generated comments.
  Whether those comments see values is unknown.

## What the state of the art adds beyond Metis

1. **Semantic similarity of names and comments.**
   - **Metis today:** Jaccard over exact words misses paraphrase, synonyms the lexicon lacks, and
     descriptive comments.
   - **Embedding retrieval** (Magneto, SCHEMORA) recovers them with no values. Latence already
     keeps embedders as a resolution signal (ADR-0068).
2. **A bounded LLM adjudicator.**
   - **What it does:** reranks the top-k candidates for a column, or judges a key pair that
     survived pruning. Every strong 2025–26 system has one.
   - **What it gains:** it converts some of Metis's `unknown` records into proposals. Each must
     be a recorded, replayable decision whose actor is the model.
3. **Query history.** Written joins give relationships with their direction and kind.
   Recurring `WHERE` exclusions give conventions. Recurring `GROUP BY` and aggregate columns give
   dimensions and measures.
4. **Column lineage.** An identity or rename edge passes a concept and its keyness downstream,
   with no values read. It is native on Databricks.
5. **Declared constraints and existing UC metadata.** Read them as prior evidence, verified,
   never trusted outright.
6. **A global join graph.** Auto-BI's k-MCA chooses edges jointly; Metis's greedy pass with a
   veto can lock in an early wrong edge.
7. **Multi-column foreign keys.** Metis finds composite keys but links single columns only.
8. **Fuzzy and semantic joins** (DeepJoin, WarpGate). These need value embeddings and suit lake
   discovery more than a governed warehouse.
9. **Fixed-vocabulary type annotators.** Useful only if Schema.org or DBpedia types are in scope,
   and they need values.

**Where Metis is already ahead of everything read:** it settles sensitivity before reading,
computes salted sketches in the source, verifies exactly, finds histories, abstains with its
candidates recorded, and calibrates on owner decisions. The constraint-guided result supports
Metis's premise: constraints carry precision.

## Recommended signal stack for v1

Each signal is evidence on a proposal, with an actor and a locator keyed on UC ids. Calibration
is per class, as in Metis.

**Tier 0: metadata only, always on**
1. **UC structure:** names, types, comments, tags, declared PK/FK with `RELY`, metric-view
   definitions and partition columns.
2. **Lexicon linking**, Metis's design in German and English: Jaccard, the prefix rung,
   qualification, margin and abstention.
3. **Multilingual embedding similarity** over name plus comment, against concept names, synonyms
   and definitions. It retrieves and is a feature; it never decides alone. It needs a pinned,
   permissively licensed checkpoint.
4. **Column lineage:**
   - identity edges propagate concepts as proposals, depth-capped and never overwriting;
   - lineage between two tables raises the prior of a join between them.
5. **Query-history mining** with a deterministic SQL parser. It yields join pairs with their
   frequency, filter literals and aggregate columns. **Literals are values:** gate them like
   values.

**Tier 1: aggregates and hashes computed in the warehouse**
6. **Profiles:** Metis's statistics, shapes, gated code-list values and salted bottom-k sketches,
   computed as warehouse SQL.
7. **Key and inclusion verification:** exact within a catalog. Across catalogs, by sketch
   containment, abstaining when hashes are too few. Add bounded two-column FKs.
8. **Join constraints:** keep the shortest-path veto and the cardinality checks.

**Tier 2: an LLM over metadata, off by default, behind the egress decision**
9. **Adjudicator:** it handles columns left `unknown` and key pairs that survived pruning, with
   their top-k candidates. Its prompt holds names, comments, types, statistics and candidates,
   **and no values**. Each decision is recorded and replayed.

**Defer**
- learned column-type annotators;
- any LLM that sees samples (an egress decision per estate);
- neural join search (DeepJoin, WarpGate, Starmie, TabSketchFM): it needs values and a GPU, and
  some of its repositories have no licence;
- n-ary IND discovery and a global k-MCA solver, until the greedy pass fails at scale.

**Evaluation:**
- **Held-out estate:** headers abbreviated in German and English in SNAILS style, with gold
  concepts and joins. Report precision and recall per class, the abstention rate and owner
  effort.
- **Public baselines:** Valentine and BIRD-Dev schemas with their FKs stripped, as cheap English
  baselines.
- **Caution:** don't rely on MIMIC/OMOP or GDC alone, given the contamination finding.

## Open questions

1. **Egress.** May tier 2 run at all, and against which endpoint? Do the literals in statement
   text count as values?
2. **Grants.** Will deployments grant the rights needed to read `statement_text`? Without them,
   mining is limited to lineage and table-level usage.
3. **Embedding checkpoint.** Which one: weights licence verified, CPU-viable, and with
   byte-identical replay (thread pinning)?
4. **Candidate concepts.** Does the adjudicator choose only among library concepts, or also
   among concepts the text path induced? This bears on the no-over-merge rule.
5. **Failed declared constraints.** Is a UC PK/FK that fails verification raised as a drift
   proposal upstream, or only recorded as evidence?
6. **Salts.** One salt per estate or per tenant, and what do cross-metastore links mean when the
   salts differ?
7. **Re-reading.** Should the UNVERIFIED numbers (Magneto, ReMatch, HoPF, Scalable Join
   Inference's recall) be re-read once arxiv.org and vldb.org are reachable?

## Sources

- [V] github.com/delftdata/valentine; arXiv 2010.07386; SMUTF, arXiv 2402.01685.
- [P] Parciak et al., TaDA@VLDB 2024, arXiv 2407.11852.
- [R] ReMatch, arXiv 2403.01567.
- [S] SCHEMORA, arXiv 2507.14376.
- [M] Magneto, PVLDB 18(8), arXiv 2412.08194; github.com/VIDA-NYU/magneto-matcher.
- [MH] MetaHarmonizer, bioRxiv 10.64898/2026.06.13.732088.
- [CG] Constraint-Guided Enterprise Data Mapping, arXiv 2608.24218.
- [CTA] github.com/mitmedialab/sherlock-project; Sato, arXiv 1911.06311;
  github.com/megagonlabs/doduo.
- [KB] Korini & Bizer, arXiv 2306.00745 and 2503.02718; hyper.ai SOTAB leaderboard.
- [LE] arXiv 1603.07410; github.com/ekzhu/datasketch.
- [J] github.com/ekzhu/josie.
- [DJ] arXiv 2212.07588.
- [WG] arXiv 2212.14155.
- [ST] arXiv 2210.01922; github.com/megagonlabs/starmie.
- [TS] arXiv 2407.01619.
- [EJ] arXiv 2510.24599.
- [B] BINDER, VLDB 2015; FAIDA and SINDY (Kruse et al.); github.com/HPI-Information-Systems/Metanome.
- [H] HoPF, JIIS 2020; Rostin et al., WebDB 2009.
- [AB] arXiv 2306.12515.
- [LF] arXiv 2603.07278.
- [DA] arXiv 2603.23050.
- [SJ] arXiv 2603.04176.
- [QL] arXiv 1902.00031.
- [LP] DataHub, Apache Atlas and Atlan propagation docs.
- [SN] SNAILS, SIGMOD 2025; github.com/kyleluoma/SNAILS.
- Databricks docs (search summaries only): constraints, the lineage and query-history system
  tables, data governance.
- The auto-ontology project review: the 2026-11-03 research holding (a reference read).
