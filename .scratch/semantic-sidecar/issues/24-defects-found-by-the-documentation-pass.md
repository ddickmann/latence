# Task: defects the documentation pass found and did not fix

Type: task
Mode: HITL
Status: open
Blocked by:

## Question

Checking the pages against the code for *What the retrieval removal left behind* turned up
places where the code, or a page outside that pass, is wrong. None was fixed there, because
each is a behaviour change or belongs to a page that pass did not cover. Each needs a fix with
a test that is red first, or a decision that the behaviour is intended.

**Configuration that is accepted in silence**

1. Removed config keys are ignored without a word. `graph.canonical`, `export.jsonl_parquet`
   and `export.knowledge_graph` accept `hyperedges`, `context_columns` and `hyperedge_kv`.
   `latence stack check` reports only `graph_features` and `bm25`. The repository's rule is a
   loud typed error for config that cannot be served.

**The tuner**

2. `latence tune` maps `ner.endpoint` to the endpoint knob family. The registered Provider name
   is `entity.endpoint`, so that Provider is never swept.
3. `latence tune --sample N` does not size the sample. The probe times a run of the whole stack
   over the bundled validation corpus and reports `N` divided by that time as records per
   second. The tutorial now says how to read the figure; the flag still promises something else.

**The benchmark campaign**

4. `latence-campaign run <stage> --fresh` does not redo a stage whose artifacts are intact.
5. The wizard's budget prompt says the run blocks at the cap. The stage turns the cap into a
   call budget, and calls past it fail open.

**Benchmark folders outside the campaign**

6. `benchmark/enterprise-gold/run_eval.py` and `benchmark/rfc/run_benchmark.py` import
   `latence_retrieval` and cannot run. Decide as for the other folders: delete the scripts and
   keep the records.
7. `benchmark/corpus/test_manifest.py` has two failing tests. The file is outside
   `pytest packages`, so no gate runs it.

**A performance figure without a record**

8. "About 2.7× faster" (served OCR against in-process) has no measurement record;
   `PERF-RESULTS.md` lists the comparison as pending. It was reworded in `docs/deployment.md`
   and tutorial 5. It is still stated in `docs/serving-lighton-vllm.md`,
   `docs/TUTORIAL-fresh-pod-walkthrough.md` and the `latence-parser-lighton-vllm` package
   (README, `pyproject.toml`, two docstrings). Either the measurement is recorded, or the
   figure goes from those places too.

**Not executed**

9. The compose and Helm commands in `docs/deployment.md` were checked by reading the files.
   They have not been run: that needs a machine with `helm` and a docker daemon.

## Done when

- Items 1–5 are each fixed with a red-first test, or recorded as intended with the help text
  or prompt corrected to match.
- Items 6 and 7 are decided and done.
- Item 8 has a recorded measurement or no remaining unsupported statement.
- Item 9 is run once on a machine that can, and the result recorded.
