# benchmark/enterprise-gold: dated records

This folder holds a synthetic corpus, a gold question set built from it, and three files that
record one evaluation run. They date from August 2026. `PROVENANCE.md` says how the corpus and
the labels were made; read it before citing anything from here.

Three scripts were removed on 2026-11-10, together with the retrieval layer
(`docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md`).
None of them could run:

- `run_eval.py` imported `latence_retrieval`, a package that is no longer in this repository.
- `run_pipeline.py` ran `pipeline.yaml` and then read `graph-hyperedges.parquet` and the
  hyperedge section of the Quality Report. The current pipeline writes neither.
- `make_judgments.py` read `graph-hyperedges.parquet` from the same export.

Git history has all three.

| File | What it is |
|---|---|
| `corpus/` | 40 synthetic business documents, English and German |
| `chains.py`, `CHAINS.md` | The table of planted facts and question chains, and its rendered form |
| `build_gold.py` | Resolves the planted facts against an exported corpus and writes `gold.jsonl` and `CHAINS.md` |
| `gold.jsonl` | 100 questions with supporting-fact spans |
| `judgments.jsonl` | 226 calibration judgments, one per hyperedge of the August export |
| `eval-results.json` | Supporting-fact recall at five token budgets for three retrieval configurations, with the gate verdict |
| `pipeline.yaml` | The pipeline configuration of the run the three files above were measured on |

`judgments.jsonl` and `eval-results.json` describe the removed retrieval tooling and the
hyperedges of the pipeline as it was then. They are not measurements of the current code, and
the current tree cannot reproduce them.

`pipeline.yaml` is kept as the record of that run and no longer runs: its `graph` Stage carries
the `hyperedges` key, which ADR-0068 removed, and a run refuses a Pipeline that carries it.

`build_gold.py` and `chains.py` import nothing that was removed. `build_gold.py` reads an
export under `store/`, which is not tracked and which `run_pipeline.py` used to produce. It has
not been run against an export of the current pipeline.
