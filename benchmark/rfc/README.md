# benchmark/rfc: what is left of the RFC retrieval harness

This folder held the scripts of a retrieval benchmark over a corpus of IETF RFCs. Two of them
remain, with their tests.

Four scripts and four test files were removed on 2026-11-10, together with the retrieval layer
(`docs/adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md`).
None of the four scripts could do its work on the current tree:

- `run_benchmark.py` and `build_questions.py` imported `latence_retrieval`, a package that is
  no longer in this repository, to query a live Qdrant collection.
- `sparse_terms.py` was a helper that only those two used.
- `make_report.py` rendered the result file that `run_benchmark.py` wrote. No such file is
  tracked.

`test_run_benchmark.py`, `test_build_questions.py` and `test_make_report.py` tested those
scripts. `test_pod_path_contracts.py` checked them against contracts of the removed package and
could no longer be imported. Git history has all eight files.

| File | What it does |
|---|---|
| `build_corpus.py` | Downloads RFC texts from rfc-editor.org and writes them as a corpus, plus `_refs.jsonl`: the cross-references it reads from each RFC's `Obsoletes:` and `Updates:` header lines and from RFC numbers cited in the body. Standard library only; needs network |
| `stage_charts.py` | Reads a finished pipeline run directory and writes four SVG charts: chunk sizes, mentions against graph nodes and edges, node degree and centrality, and the overlap of extracted `obsoletes` / `updates` edges with `_refs.jsonl`. Standard library only |
| `test_build_corpus.py`, `test_stage_charts.py` | Offline tests of the two scripts. No gate runs them. Run them with `PYTHONHASHSEED=0 uv run pytest benchmark/rfc -o addopts=""` |

The docstrings of both scripts cite `docs/TUTORIAL-rfc-retrieval-proof.md`. That page is not
in this repository.

`results/dogfood-rfc/` holds the dated record of a retrieval run over an RFC corpus. Its
numbers describe the removed retrieval tooling, not the current code, and its own script
(`results/dogfood-rfc/report/rfc_bench.py`) imports `latence_retrieval` as well.
