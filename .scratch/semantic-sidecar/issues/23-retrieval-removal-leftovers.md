# Task: what the retrieval removal left behind

Type: task
Mode: HITL
Status: resolved
Blocked by:

## Question

*Remove the retrieval tooling* took out the packages, the core modules and the retrieval docs.
The rewrite of the ten stale pages on 2026-11-10 turned up things that removal missed, plus
places where the code and its own help text disagree. Each needs a fix or a decision.

**Left over from the removal**

1. *(Done 2026-11-10; owner chose "replace with the new bars".)* **The release gates still
   demanded removed capabilities.** `RELEASE-GATES.md` Gates 2 and 3,
   and its closing section, require SPLADE proven as a live signal against a real Qdrant and a
   graph leg proven to retrieve. `test_public_release_gate.py` asserts that "splade" and
   "graph leg" appear there. Neither can ever be met now. **Owner's decision:** what the release
   gates are for the re-scoped product. The test changes with the gates, as one decision.
2. **The benchmark campaign still reuses pre-removal exports.**
   `latence_benchmark/campaign/pipeline_stage.py::verify_reused_export` accepts an export by
   finding `graph-hyperedges.kv`, a file only the earlier pipeline wrote. The reuse path exists
   to feed the removed retrieval stage. Decide whether the reuse path goes, or is re-keyed on a
   file the current pipeline writes.
3. **The labelling tool still has a hyperedge mode.** `latence-label calibrate` serves a
   judgment UI over `graph-hyperedges.parquet`, which no pipeline writes any more.
4. **Stale text in kept code:**
   - `latence contracts --capability` help says "e.g. 'tokenizer'", a removed seam;
   - the diagram in `stacks/phase_boundary.py`'s docstring still shows an `enrich` stage.
5. **`ROADMAP.md` is entirely pre-removal** (16 lines mention retrieval, hyperedges, SPLADE or
   Qdrant). It belongs to the docs-and-site rewrite on the map.
6. **`docs/anatomy/02` and `03`** still carry the dated "written before the removal" warning.

**Found on the way, not caused by the removal**

7. *(Done 2026-11-10; owner chose "scrub them on purge".)* A purge now drops every live
   listing of the document from every Corpus Version, and with it the file name and content
   hash. **One deviation from the owner's choice:** the content-addressed id stays, as a
   DELETED tombstone. `DeltaRunner._prior_deletions` reads it to keep a Source that still serves
   the file from re-ingesting the document, so removing it would let an erased document come
   back. The help text says so. Original finding: **a purge kept the document's file name.** `latence purge` says "NOTHING is retained".
   `CorpusStore._reconcile_manifest` rewrites only the fingerprint and the record count of each
   Corpus Version manifest and keeps the delta lineage verbatim, and `DeltaEntry` carries
   `file_name`. So after a purge the document's id, content hash and file name are still in
   `_latence/corpus/wal/v*/version.json`. A file name can be personal data. **Owner's
   decision:** scrub those fields on purge, or correct the claim. Either way the help text and
   the behaviour have to agree, with a test.
8. `latence run` on a shipped `stacks/*.yaml` exits 0 after writing into a literal
   `{STORAGE_URI}` directory with no documents (reported by the page rewrite; not reproduced
   again here).
9. `manifest.json` redacts every key containing "token", which also hides the chunk budget
   (`max_tokens`, `overlap_tokens`, `min_tokens`) (reported by the page rewrite; not reproduced
   again here).
10. `RELEASE-GATES.md` links `issues/e2e-extended/RUNBOOK.md`, which does not exist. The same
    dangling link is on `main`.
11. The deploy artifacts' own comments disagree with what they do: the compose file and the
    Helm notes say a run writes a knowledge graph, but the shipped served-OCR stack has no graph
    stage; the Helm CronJob is described as incremental but passes a fixed run id (reported by
    the page rewrite; not reproduced again here).

**Statements the page rewrite left in place without checking them against code**

12. `docs/deployment.md`: the helm and docker commands (checked against templates and values,
    not executed); the "~24 GB GPU" sizing for the default LLM; the `huggingface-cli` steps; the
    2.7× OCR figure (cited to the serving guide, not re-measured); the Airflow "tested, not yet
    operated" paragraph beyond `build_dag` and the CI job existing.
13. `docs/tutorials/05-production.md` §6: `env placement`, `tune` and `--tuned` are described
    from `--help` text only.
14. `docs/tutorials/04-bringing-your-own-provider.md` §8: the `AdapterBase` helper names are
    verified; the descriptions of what they do are not line-checked.
15. `docs/anatomy/index.md`: the "why this one" model rationales, the one-line corpus
    descriptions and the Gemma sentence.
16. `docs/BENCHMARKS-E2E.md`: the literature figures (Hits@1 0.26–0.31).
17. `docs/guides/benchmark-campaign.md`: stage behaviour is described from reading the code; no
    stage was run.

## Done when

- Items 1 and 7 are decided by the owner and implemented, each with a test that was red first.
- Items 2–4 are fixed or removed, with the tests that covered them adjusted to the kept
  behaviour and nothing weakened.
- Items 8–11 are each reproduced, then fixed or recorded as intended.
- Items 5 and 6 are closed by the docs-and-site rewrite or here.
- Items 12–17 are each verified or reworded to what can be verified.

## Progress (2026-11-10): the code leftovers

Status stays `open`: items 5, 6 and 12–17 are documentation and are not done.

- **2, done.** The reuse path is removed, not re-keyed: nothing current read a reused export.
  `musique` and `hotpotqa` stay selectable; the stage has no pipeline for them and now lists
  them as `not_processed`. **Left for the owner:** whether those two leave the dataset registry.
  The committed August record of the stage no longer satisfies the stage's done-check for a
  selection that includes them.
- **3, done.** The `calibrate` command, its module, endpoints and page section are gone. The
  gold-set, `stats` and `validate` parts are unchanged.
- **4, done.** The help example names a capability that exists (`delta`); the diagram shows
  the stages that exist.
- **8, reproduced and fixed.** `latence run stacks/default.yaml` exited 0 after creating a
  directory named `{STORAGE_URI}`. A Pipeline that still carries a `{UPPER_SNAKE}` placeholder
  is now refused with a `ConfigError` before anything is written, by the local, Delta and
  Airflow Runners; the CLI exits 2. Parsing a template is unchanged, so `stack validate`, the
  tuner and the bake-offs still read the shipped files. **Known limits:** a missing source
  folder still gives an empty run with exit 0; a legitimate config string of the form
  `{UPPER_SNAKE}` would be refused (`{lower}`, `{{DOUBLED}}` and `\N{NAME}` are not).
- **9, reproduced and fixed.** The manifest showed the chunk budget as redacted. A closed list
  of five token-count keys is now shown, and only while the value is a number: `max_tokens`,
  `min_tokens`, `overlap_tokens`, `max_new_tokens`, `microbatch_token_budget`. Every other key
  containing `token` stays redacted. The last two keys go beyond what this ticket named.
- **10, did not reproduce.** The dangling link was already gone. `RELEASE-GATES.md` is now
  under the doc-truth link check, which fails on the page as it was before that fix.
- **11, verified; wording corrected, behaviour unchanged.** The compose file, the chart's values
  and install notes no longer say a run writes a knowledge graph, and the CronJob is no longer
  called incremental (a second run with the same run id replays its checkpoints). The parsed
  YAML of the changed files is identical before and after. **Not checked:** the Helm templates
  were not rendered (no `helm` here).
- **Gates on the final commit:** ruff, strict mypy on 30 packages and the full suite
  (4,382 passed, 17 skipped) exit 0. `mkdocs build --strict` stops on the Mermaid URL check,
  which needs a host this machine cannot reach. `scripts/verify-local.sh` was not run.

## Progress (2026-11-10, documentation)

Items 5, 6 and 12–17 are worked through below. The
owner's question left under item 2 is answered first.

- **2, the registry question, decided and done.** `musique` and `hotpotqa` left
  `KNOWN_DATASETS`. A test, red before the change, requires every registry name to have a
  `CORPUS_PLAN` entry and a place in `PROCESS_ORDER`. Removed with them: the `local-reference`
  kind and its fetcher, the `attribution` lane, and the `not_processed` bookkeeping. The
  manifest keeps `publishable`; it is `False` for every remaining lane. Eight scripts under
  `benchmark/official/`, `benchmark/official-data/` and `benchmark/musique-subset/` could not
  run and are deleted; their result files, question sets and export data stay, and each folder
  has a note. The three `official-data` scripts imported nothing removed; they read a hyperedge
  export no pipeline writes, and each fails with `FileNotFoundError` on a clean clone. Their
  rubric text is copied into `PROVENANCE.md`.
- **5, done.** `ROADMAP.md` is rewritten. Each "built" line was checked against the registered
  Providers, the CLI and the modules it names. "Next" restates ADR-0067 and ADR-0069 to
  ADR-0073 as intentions. A new test fails on the earlier page: a paragraph that names a
  removed capability must say it is gone. The page is in the link roster.
- **6, done.** Both anatomy chapters are rewritten where they described removed features, the
  warning blocks are gone, and the new test covers both. Every `file:line` reference was
  re-read: 31 of 72 corrected in chapter 2; in chapter 3, 7 of 17 corrected and one removed
  with its passage.
  `test_setup_wizard_knobs.py` passes on the old and the new chapter 3.
- **12, `docs/deployment.md`.**
  - (a) The compose and Helm commands were checked by reading `deploy/docker-compose.yml`,
    `values.yaml` and the templates: the `llm` profile, `pipeline.kind`, `pipeline.schedule`
    and every key of the values table exist as written. **Not executed:** no `helm` and no
    docker daemon here.
  - (b) The "~24 GB" sizing had no source. The page now gives the defaults the compose file and
    the chart set, and the weight size that follows from the 7,615,616,512 parameters the
    model's public card lists.
  - (c) `huggingface_hub` 2.4.0 installs `hf` and no `huggingface-cli`. The step uses `hf`.
  - (d) The 2.7× figure has no measurement record. The serving guide states it without one, and
    `PERF-RESULTS.md` lists the comparison as pending. Reworded here and in tutorial 5. **Still
    stated without a record** in `docs/serving-lighton-vllm.md`,
    `docs/TUTORIAL-fresh-pod-walkthrough.md` and the `latence-parser-lighton-vllm` package
    (README, `pyproject.toml`, two docstrings).
  - (e) The `airflow-dag` job exists in `ci-heavy.yml` and fails unless both real `build_dag`
    tests pass. The paragraph now names the tests and says that none starts a scheduler.
- **13, `docs/tutorials/05-production.md` §6.** Checked against `placement.py`, `tune/` and
  `cli.py`. Four sentences reworded. Pinned by `test_placement.py`
  (`test_no_cuda_device_is_a_failure_not_a_skip`,
  `test_weights_resident_on_cuda_are_not_mistaken_for_gpu_work`,
  `test_a_model_loading_provider_absent_from_learned_providers_is_still_probed`),
  `test_tune.py` (`test_picks_the_fastest_correct_setting`,
  `test_rejects_inference_time_compile_crash_named_compile_failure`,
  `test_rejects_a_knob_that_changes_output`,
  `test_implausible_speedup_guard_flags_a_leaked_cold_baseline`,
  `test_overlay_NOT_applied_on_fingerprint_mismatch_falls_back_to_defaults`) and `test_cli.py`
  (`test_cli_run_tuned_*`). **Not covered by any test:** the real probe behind `latence tune`
  and the CLI wrappers of `tune` and `env placement`.
- **14, `docs/tutorials/04-bringing-your-own-provider.md` §8.** The four helper descriptions
  match `providers/adapter.py`, `test_adapter_base.py` and `test_e2e_adapter_base.py`. None was
  wrong; each now states the defaults and error cases the source has.
- **15, `docs/anatomy/index.md`.** Licences and dates match the profiles. The rationale column
  now holds what each profile declares and what the campaign record says. The corpus rows
  follow `CORPUS_PLAN` and the stage code. The Gemma sentence follows
  `THIRD-PARTY-LICENSES.md`: no Provider here is backed by Gemma.
- **16, `docs/BENCHMARKS-E2E.md`.** No ADR, evidence page or research note cites a source for
  the Hits@1 range. The number and the error rate derived from it are removed.
- **17, `docs/guides/benchmark-campaign.md`.** Checked against `stages.py`, `cli.py`, the three
  stage modules, the pod worker and the campaign tests. Three statements changed; the rest
  hold. No stage was run against a pod.

**Found on the way, not fixed.**

- Removed config keys are accepted in silence. `graph.canonical`, `export.jsonl_parquet` and
  `export.knowledge_graph` ignore `hyperedges`, `context_columns` and `hyperedge_kv`.
  `latence stack check` reports only `graph_features` and `bm25`.
- `latence tune` maps `ner.endpoint` to the endpoint knob family. The registered name is
  `entity.endpoint`, so that Provider is never swept.
- `latence tune --sample N` does not size the sample. The real probe times a run of the whole
  stack over the bundled validation corpus and reports `N` divided by that time as records per
  second. The tutorial now says how to read the figure.
- `benchmark/enterprise-gold/run_eval.py` and `benchmark/rfc/run_benchmark.py` import
  `latence_retrieval` and cannot run. They are outside the three folders this pass covered.
- `latence-campaign run <stage> --fresh` does not redo a stage whose artifacts are intact.
- The campaign wizard's budget prompt says the run blocks at the cap. The stage converts the
  cap into a call budget; calls past it fail open.
- `benchmark/corpus/test_manifest.py` has two failing tests. The file is outside
  `pytest packages` and was not touched.

## Answer (2026-11-10)

Every item is closed: 1 and 7 by the owner's decisions, 2–4 and 8–11 by the code pass, 5, 6
and 12–17 by the documentation pass, each as recorded in the two Progress sections above.

- **Owner's decisions on the way:** `musique` and `hotpotqa` leave the campaign's dataset
  registry, and the scripts that could no longer run are deleted while their recorded results
  stay; the manifest keeps showing all five numeric token-count keys.
- **What this ticket did not finish** is carried by
  [Defects the documentation pass found and did not fix](24-defects-found-by-the-documentation-pass.md):
  seven defects in kept code and benchmark folders, one performance figure that is still
  stated without a record in three places, and the compose and Helm commands, which were
  checked by reading and never executed.
- **Gates on the final tree of the documentation pass:** ruff, strict mypy on 30 packages, the
  full suite (4,382 passed, 17 skipped) and the secret scan exit 0. `mkdocs build --strict`
  stops on the Mermaid URL check, which needs a host this machine cannot reach.
  `scripts/verify-local.sh` was not run.
