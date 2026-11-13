# Research: the state of the art in schema matching, column-to-concept annotation and key/join discovery

Type: research
Mode: AFK
Status: resolved
Blocked by: 

## Question

How do we map warehouse columns and tables to business concepts, and discover joins, as well
as the state of the art allows?

Cover:

- LLM-assisted schema matching and its benchmarks;
- semantic column-type annotation;
- table and column embeddings over names, comments and values;
- value sketches for join discovery (MinHash, LSH Ensemble, JOSIE-style, contrastive);
- inclusion-dependency and key discovery at scale;
- using query history and lineage as matching signals.

For each approach give: inputs, precision/recall on the benchmark it reports, cost, licence,
and whether it needs to read raw values (an egress concern).

Metis's layer job already does header folding, code lists, inclusion dependencies with
sketches, and to-one key verification (`metis-core/layer/{keys,semantics,evidence}.py`).
Say what the state of the art adds beyond that.

Resolves with a report at `research/03-sota-schema-matching.md`.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/03-sota-schema-matching.md](../research/03-sota-schema-matching.md). arXiv, VLDB and the Databricks docs were blocked, so most numbers come from search summaries, and single-source figures are marked UNVERIFIED.

- **The field's shape:** the strongest 2025–26 systems find candidates cheaply or prune them with statistics, then have a model rerank or judge what's left. The best enterprise result credits rules ("a matching identifier is required") for its precision, more than the model.
- **Metis already has the statistical core.**
  - Sensitivity is decided before any value is read.
  - Sketches are computed in the warehouse.
  - Keys, inclusion dependencies and single-column references are checked exactly.
  - Weak links stay unknown.
  - Confidence is calibrated on owner decisions; no system read matches this.
- **What Metis doesn't use for matching:**
  - embeddings or any model;
  - query history;
  - lineage (its interface carries table-level lineage that the matcher ignores);
  - declared UC key constraints;
  - multi-column foreign keys.
- **The open-source auto-ontology project** has a model guess foreign keys from names plus top values of a 1,000-row sample. It prunes only joins with zero value overlap, checks no cardinality, has no egress gate and doesn't record its model calls.
- **Benchmark caveat:** the benchmarks are English and mostly biomedical or web tables. One 2026 preprint reports models recall 80–100% of a benchmark's target names with no schema shown. No benchmark has German or English abbreviated headers, so our own gold set is needed.
- **Recommended v1:** keep the Metis-style statistical core keyed on UC ids, and add five signals that send no values out:
  1. UC structure and declared keys;
  2. the lexicon;
  3. multilingual embeddings over names and comments;
  4. column lineage;
  5. query-history mining. Literals in query text count as values and pass the same gate.

  Add an optional model judge that sees metadata only, behind the egress decision.
- **Defer:**
  - learned column-type models;
  - any model that sees sample values;
  - neural join search;
  - multi-column dependency discovery;
  - a whole-graph join solver.
- **For other tickets:**
  - whether the metadata-only judge may run, and rights to read query text (*the warehouse pipeline*, *the Databricks semantic surface*);
  - the embedding model, and whether it replays byte-identically (*the upgrade of the resolution engine*);
  - where candidate concepts come from (*the destination and the domain model*);
  - what to do with declared keys that fail checks;
  - sketch salts across metastores (*the boundary with Metis*).

## From the map (moved 12 Nov)

[The state of the art in schema matching](issues/03-sota-schema-matching.md) — cheap candidates first, then a model judges a short list, and rules drive precision. Metis already has the statistical core (exact key and inclusion checks, calibrated owner decisions). v1 adds five signals that send no values out: UC structure and declared keys, the lexicon, multilingual embeddings, column lineage and query-history mining. A model judge that sees metadata only is optional and gated by egress. Learned column-type models and any model that reads values are deferred.
