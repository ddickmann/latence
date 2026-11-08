# Research: the state of the art in schema matching, column-to-concept annotation and key/join discovery

Type: research
Mode: AFK
Status: open
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
