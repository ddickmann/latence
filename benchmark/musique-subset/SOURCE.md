MuSiQue-Ans validation split via HF mirror dgslibisey/MuSiQue (original: StonyBrookNLP/musique, arXiv 2108.00573). Licence: CC BY 4.0 (wayfinder ticket 02). First 1000 questions, even=dev-tune / odd=subset-test.

## These files are dated records

The folder's one script, `run_eval_final.py`, was removed on 2026-11-10 together with the
retrieval layer
(`docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md`).
It imported `latence_retrieval`, a package that is no longer in this repository, so it could
not run. Git history has it. Nothing in the current tree reads the files that remain:

- `musique-dev.jsonl` and `musique-test.jsonl`: the two question sets described above.
- `export/corpus.jsonl` and `export/graph-hyperedges.kv`: the two files of one earlier
  pipeline run that the script read. The current pipeline does not write a hyperedge file.
- `final-results.json` and `final-results-piecewise.json`: the scores the script recorded,
  for the removed retrieval tooling. They are not measurements of the current code.

## What is committed here, and what is not

`run_eval_final.py` read two files out of `export/`: `corpus.jsonl` (the chunk text it scored
against) and `graph-hyperedges.kv` (the point-lookup evidence sidecar). Both are tracked.

The rest of that export — `corpus.parquet`, `graph-nodes.parquet`, `graph-edges.parquet`,
`graph-hyperedges.parquet`, `graph.ttl`, `graph.graphml` — was 110 MB of derived output that no
script, test or document in this repository opened. It is not tracked. Those files are the
serialisation twins of data already present in the two files above, produced by one historical
pipeline run; no committed configuration reproduces that run byte-for-byte, so publishing them as
a "fixture" would have overstated what they are.

The script's *enterprise* arm read `benchmark/enterprise-gold/store/`, which is likewise not
tracked.
