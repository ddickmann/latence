# benchmark/official: dated records

This folder holds two result files. They were committed in August 2026.

The four scripts that produced them were removed on 2026-11-10, together with the retrieval
layer (`docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md`).
Each of them imported `latence_retrieval`, a package that is no longer in this repository, so
none of them could run. Their names were `official_eval.py`, `run_official_eval.py`,
`run_recall5.py` and `calib_ab.py`; git history has them.

| File | What it records |
|---|---|
| `official-results.json` | For `musique-full` and `hotpot-full`: the dev curves, the configuration tuned on dev, the test score and the gate verdict |
| `recall5-results.json` | Paragraph-level recall@5 on the same two test sets: a BM25 baseline, and the chain-of-evidence arm at two entry tiers |

Both files describe the removed retrieval tooling on exports of the pipeline as it was then.
They are not measurements of the current code, and the current tree cannot reproduce them. The
corpora and exports the scripts read are not tracked.
