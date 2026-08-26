MuSiQue-Ans validation split via HF mirror dgslibisey/MuSiQue (original: StonyBrookNLP/musique, arXiv 2108.00573). Licence: CC BY 4.0 (wayfinder ticket 02). First 1000 questions, even=dev-tune / odd=subset-test.

## What is committed here, and what is not

`run_eval_final.py` reads two files out of `export/`: `corpus.jsonl` (the chunk text it scores
against) and `graph-hyperedges.kv` (the point-lookup evidence sidecar). Both are tracked, so the
MuSiQue arm of the harness runs from a clean clone.

The rest of that export — `corpus.parquet`, `graph-nodes.parquet`, `graph-edges.parquet`,
`graph-hyperedges.parquet`, `graph.ttl`, `graph.graphml` — was 110 MB of derived output that no
script, test or document in this repository opened. It is not tracked. Those files are the
serialisation twins of data already present in the two files above, produced by one historical
pipeline run; no committed configuration reproduces that run byte-for-byte, so publishing them as
a "fixture" would have overstated what they are.

The harness's *enterprise* arm reads `benchmark/enterprise-gold/store/`, which is likewise not
tracked and IS regenerable — see `benchmark/enterprise-gold/.gitignore` for the one-line command.
