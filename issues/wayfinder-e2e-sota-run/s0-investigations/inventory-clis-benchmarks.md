# S0 investigation — CLIs, benchmark harnesses, wizard precedent (reader B of 4)

Distilled from the full inventory pass; the campaign-shaping discoveries first.

## Headline discoveries for the campaign wizard

1. **A wizard already exists and defines the house style**: `latence setup`
   (packages/latence-core cli.py:575, ~680 lines of I/O) + a PURE generator
   (`setup/wizard.py`, byte-deterministic) + a per-stage recipe seam (`setup/recipe.py`,
   `StageContribution`/`StackRecipe`). 23 flags, all None-defaulted so unset ⇒ prompt
   (interactive) or documented default (`--non-interactive`). **The campaign wizard must
   mirror this**: frozen `CampaignAnswers` struct produced identically by prompts and flags,
   contradictions rejected in `__post_init__`, flag > prompt > default, "why it matters"
   lines to stderr, advanced knobs flags-only, generate → dry-validate → write → summary +
   exactly one next command. Exit 2 usage / 1 invalid artifact / 0 written.
2. **OCR is already a first-class seam**: `latence setup` has `--ocr-base-url`,
   `--ocr-model`, `--ocr-dpi`, `--ocr-max-longest-dim`; `stacks/gpu-ocr.yaml` exists; the
   e2e supervisor runs an `ocr_server` daemon stage. LightOnOCR-2 integration = a vLLM
   OpenAI-compatible endpoint plugged into the existing seam, NOT new pipeline code.
3. **The staged-campaign precedent is `benchmark/e2e/supervise.sh`**: stage table
   (`add_stage <name> <task|daemon> <desc> <done-probe> <progress-probe> <cmd>`), resumable
   by FILESYSTEM PROBE not exit code, stall detection via monotonic progress probes
   (per-stage budgets), MAX_ATTEMPTS retries, atomic STATE.json (temp+rename), `blocked` as
   a terminal state, flock single-instance, pid-file liveness (never pgrep -f), separate
   keeper process. Shell-only + not importable — the campaign runner productizes this shape
   in Python.
4. **No "campaign" concept exists in the repo** — the name is free.
5. **`benchmark/rfc/` is the model for testable stage scripts**: proper argparse per stage,
   injectable engine factory + Signals protocol, six pytest files proving logic offline.
6. **`latence stack check/validate`** = the reusable config gates (offline structural check;
   end-to-end validate on a bundled corpus, run twice for determinism, device-honesty).
   `validate_config_text()` (wizard.py:822) is the dry-validation to reuse verbatim.

## Existing console scripts (7)

`latence` (typer: run, stack check/validate, bake-off, tune, setup, process, delta, retract,
purge, env verify/placement/render/matrix, retrieval splade-index) · `latence-benchmark`
(run/list, suite registry seam) · `latence-label` (gold/calibrate/stats/validate) ·
`latence-console` (read-only loopback web UI over `_latence/runs/`) · `latence-demo` ·
`latence-label-profile` (label-candidate inducer for `setup --label-candidates`) ·
`latence-sink-qdrant` (load/verify-absent).

## Key CLI facts for stage specs

- `latence process --config latence.stack.yaml --run-id X` runs check_stack first, refuses
  on errors. `latence run <pipeline.yaml> --run-id --tuned` is the direct form.
- `latence setup --profile {lite|enterprise-sota}` — SOTA profile forces cuda + prompts OCR
  endpoint/model only when the OCR parser path is chosen. Intake byte-cap derived from ONE
  corpus walk (floor 2 MiB, ceiling 2 GiB, headroom ×2).
- `latence tune` sweeps dtype/quantize/compile/batch knobs behind a hard correctness gate
  (checkpoint-bytes fingerprint); `latence run --tuned` overlays winners only on
  device_fingerprint match. The campaign's pod stages should use this.
- `latence bake-off matrix/<stage>.yaml` — per-stage provider bake-offs (parse, chunk,
  entity, disambiguation, embed, graph_completion, redaction, schema_induction matrices
  committed) — the S3-spec instrument for "which provider per stage".
- `latence-benchmark run --suite {parse|entity_extraction|relation_extraction|
  disambiguation|redaction|all}` — registered offline suites incl. **parse on
  olmOCR-bench** (ParseSuite) — reusable to score the LightOnOCR leg; license gate runs
  BEFORE any byte is fetched (`licensing.require_usable`).
- `latence-sink-qdrant load <export_dir>` — the engine-load step for retrieval stages
  needing a vector engine (dense/sparse/both).
- `benchmark/corpus/manifest.py` — reusable library (scan_tree/build_manifest/
  render_markdown/write_manifest) for the S1 corpus manifest artifact.
- `benchmark/corpus/fetch_corpus.sh` already fetches OHR-Bench (`FETCH_OHR`), GovDocs1,
  SEC VPRR — idempotent/resumable; the S1 acquisition extends this pattern.

## Existing eval harness reuse map (S5)

- `benchmark/musique-subset/run_eval_final.py` — recall@budget engine (BUDGETS 1k–16k,
  gate() AUC+endpoints+2%); `benchmark/official/` — per-tier curves + 4 A/Bs + recall@5;
  both are script-glue with sys.path cross-imports — the campaign should lift the logic
  into `latence_benchmark` proper (suite registry: one module + register_suite()).
- `latence-label validate` = the machine gate for gold sets; gold format defined by
  `labeling.goldset`.

## Gaps the campaign fills

- No pipeline.yaml in any benchmark dir except enterprise-gold; official harnesses consume
  pre-built exports via hardcoded paths.
- No importable Python multi-phase orchestrator (only supervise.sh).
- `docs/BENCHMARKS-E2E.md` full of `PENDING:` placeholders — this campaign's S6 report is
  what fills them.
- Three commands independently reimplement temp `--storage-root` defaulting.
