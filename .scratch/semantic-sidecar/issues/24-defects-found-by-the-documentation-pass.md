# Task: defects the documentation pass found and did not fix

Type: task
Mode: HITL
Status: open
Decided by the owner (12 Nov): Should the benchmark campaign's spend cap be a hard stop; Decide
when the campaign is next used
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

## Progress (2026-11-10)

Items 1 to 8 are done, one commit each. Item 9 is open. Each fix has a test that was red on the
tree before the fix and is green after it.

1. **Removed config keys.** Reproduced: a stack with `hyperedges`, `context_columns` or
   `hyperedge_kv` ran to the end without a message. `stack check` had no list of removed keys. It
   reported `graph_features` and `bm25` only by accident, when they were written as a
   `{provider, config}` block, through the check for sub-Provider blocks nobody reads. Now there
   is one closed list, `REMOVED_CONFIG_KEYS` in `latence_core/pipeline.py`: `graph_features` and
   `hyperedges` under Graph Assembly; `bm25`, `context_columns`, `fde_converter`, `hyperedge_kv`,
   `keep_multivectors`, `multivector_embedder` and `sparse_embedder` under Export. The Stage kinds
   were read out of the tree before the removal commit (`597e37e^`). `stack check` reports every
   key once. `Pipeline.require_runnable()` does the placeholder check and the removed-key check;
   the local, Delta and Airflow Runners and the CLI loader call it where they called
   `require_resolved()`. It raises `ConfigError` naming the key, the Stage and ADR-0068 before
   anything is written. Unknown keys in general are still accepted, and a removed key under a
   Stage kind that never read it is not refused. Red-first:
   `packages/latence-core/tests/test_removed_config_keys.py` and
   `test_a_pipeline_with_a_removed_config_key_is_refused_before_anything_is_written` in the
   Airflow package. No shipped stack or example carries such a key. The five committed pipeline
   records under `benchmark/sota-campaign/stages/s3-pipeline/pipelines/` carry five of them; they
   are untouched, they do not load (each declares an `enrich` Stage), and no test reads them.
   `benchmark/enterprise-gold/pipeline.yaml` carries `hyperedges`; see item 6.
2. **`ner.endpoint`.** Reproduced. The mapping in `tune/sweep.py` now says `entity.endpoint`.
   Red-first: `test_every_provider_in_the_knob_mapping_is_a_provider_this_workspace_declares` in
   `test_tune.py`. `PERF-RESULTS.md` had the same wrong name, and `embedder.endpoint` for
   `embedding.endpoint`; both corrected.
3. **`--sample`.** Reproduced: `RegistrySampleProbe` accepts `sample_size` and never uses it.
   Sampling was not implemented. The `--sample` and `--repeats` help, the command help, the
   report's field descriptions, table header and footer, `docs/tuning.md` and
   `scripts/tune-on-gpu.md` now say that a figure is `--sample` divided by the median wall time
   of a run of the whole stack over the bundled validation corpus, and is not records per second.
   The YAML keys of `tuned/<device>.yaml` are unchanged. Red-first:
   `packages/latence-core/tests/test_tune_sample_wording.py`.
4. **`--fresh`.** Reproduced. With `--fresh`, `run_stage` no longer consults the done-probe on
   entry, so the stage's `run` is called again; the probe still decides `done` afterwards, and an
   approval is withdrawn. `--fresh` deletes nothing: a fetched file, an OCR'd document and a
   finalized pipeline checkpoint are still skipped or replayed by the stage itself. Red-first:
   four tests in `test_campaign_stages.py` (`test_fresh_redoes_a_done_stage_whose_artifact_is_intact`
   and its neighbours) and `test_run_fresh_redoes_a_done_stage_whose_artifacts_are_intact` in
   `test_campaign_cli.py`.
5. **Budget prompt.** Reproduced. No behaviour changed. The prompt, the `--s3-budget-usd-cap`
   help, two docstrings and the campaign guide now say that the cap becomes a per-dataset limit
   on induction calls from an estimated price per call, that a chunk past the limit gets no call
   and keeps the configured labels, and that the run continues. Red-first:
   `test_the_budget_prompt_says_what_the_cap_does` in `test_campaign_cli.py`.
6. **Benchmark folders.** Verified: `run_eval.py` fails on import; `run_benchmark.py` prints its
   help and fails when it builds its engine. Removed from `benchmark/rfc/`: `run_benchmark.py`,
   `build_questions.py` (the same import, not named in this ticket), `sparse_terms.py` (used only
   by those two), `make_report.py` (renders the result file `run_benchmark.py` wrote; none is
   tracked) and the four test files of these. `build_corpus.py`, `stage_charts.py` and their
   tests stay; they pass. Removed from `benchmark/enterprise-gold/`: `run_eval.py`, and
   `run_pipeline.py` and `make_judgments.py`, which read a hyperedge export the pipeline no
   longer writes. Kept there: the corpus, `gold.jsonl`, `judgments.jsonl`, `eval-results.json`,
   `CHAINS.md`, `chains.py`, `build_gold.py` and `pipeline.yaml` as the record of the run. Each
   folder has a `README.md`; `PROVENANCE.md` has a dated note. Red-first:
   `packages/latence-core/tests/test_no_script_imports_a_removed_package.py`.
7. **`test_manifest.py`.** The tests were wrong, not the module. `manifest.py` states the
   modern-OOXML limit per format, each thin below 1,000 files, and `fetch_corpus.sh` sizes its
   axes against that threshold. One test required the text "SAM.gov", which occurs nowhere else
   in the tree. The other expected no limit for a corpus with `.docx` and `.xlsx` and no `.pptx`.
   Both now assert the per-format wording, and two new tests pin the rule and the threshold.
   14 tests pass. The two tests were the red ones.
8. **The 2.7× figure.** Reworded in the two pages and the package named above, and in two more
   places the search found: `docs/anatomy/01-ingest-and-document-understanding.md` and
   `scripts/validate-lighton-vllm-on-gpu.md`. Removed with it: LightOn's "5.71 pg/s" figure, which
   has no source in the tree, and the sentences about a head-to-head run that
   `PERF-RESULTS.md` says never took place. Red-first:
   `packages/latence-core/tests/test_served_ocr_speedup_claim.py`.

## Left for the owner

- **Should a spend cap be a hard stop?** Today the S3 induction cap is an estimate turned into a
  call budget before the run. Nothing meters the actual spend during the run, and a chunk past
  the budget falls back to the configured labels while the run goes on. A hard stop would be a
  behaviour change: block the stage at the cap, and decide what a half-induced dataset is worth.

## Still open

- **Item 9.** The compose and Helm commands in `docs/deployment.md` have not been run. That
  needs a machine with `helm` and a docker daemon.
- **Found while fixing, not fixed:**
  - The tuner's `endpoint` knob family sweeps `max_concurrency`. None of the three Providers
    mapped to it (`parser.endpoint`, `entity.endpoint`, `embedding.endpoint`) reads that key, so
    both candidates run the same configuration.
  - No gate runs `benchmark/corpus/test_manifest.py` or the tests under `benchmark/rfc/`.
  - `results/dogfood-rfc/report/rfc_bench.py` imports `latence_retrieval`. It is kept as the
    record of how the numbers beside it were produced, and the new import gate exempts it by name.
  - `PERF-RESULTS.md`, in its list of what was not measured, still names retrieval and
    query-time throughput as a gap. That tooling is removed.
  - `benchmark/enterprise-gold/build_gold.py` has not been run against an export of the current
    pipeline. Its input was produced by the removed `run_pipeline.py`.

## Decision brief for the owner (12 Nov)

- **Question:** Should the benchmark campaign's S3 cap block the stage, or stay a call budget
  that fails open?
- **Why his:** a behaviour change on money.
- **Options:**
  - A. Hard stop.
  - B. Keep as is (the wording is now truthful).
  - C. Decide when the campaign is next used.
- **Facts** (24, "Left for the owner", item 5):
  - The cap becomes a per-dataset call budget; a chunk past it keeps the configured labels and
    the run continues.
  - Nothing meters actual spend during a run.
  - The prompt and help were corrected.
  - The campaign belongs to the pre-rescope product; the current route's limit is enforced per
    call (`--max-usd`).
- **Recommendation:** C. No current work runs the campaign, and a hard stop needs a rule for a
  half-induced dataset.
- **Urgency:** after v1. Blocks nothing.
- **Cost:** none.
- **Decided by the owner (12 Nov):** Decide when the campaign is next used.
