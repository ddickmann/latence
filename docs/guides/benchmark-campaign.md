# The benchmark campaign wizard

`latence-campaign` drives the framework's end-to-end benchmark campaign — acquire real
public corpora, OCR the scanned ones, then push each corpus through the full pipeline — as
**gated, resumable stages** with filesystem done-probes and an atomic `STATE.json`. It is
deliberately drivable from a clone with the framework installed and nothing else: the CLI is
stdlib `argparse`.

```text
latence-campaign init      # the wizard: answers -> a validated campaign.yaml
latence-campaign status    # the stage table + probe results
latence-campaign run <stage>   # runs ONE stage (refused unless the previous one is approved)
latence-campaign approve <stage>  # operator sign-off on a done stage (opens the next gate)
latence-campaign resume    # continue an interrupted campaign (filesystem probes, not exit codes)
```

The campaign ends at the pipeline stage. Its earlier fourth stage, a retrieval feature matrix,
was removed with the query-time retrieval tooling
([ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)),
together with the `retrieval:` and `s5:` blocks of `campaign.yaml`.

## The wizard contract

`init` follows the `latence setup` contract: **flag > prompt > default**, a "why it matters"
line per question, one frozen answers struct from either path, generate → dry-validate →
write, and exactly one next command printed. Identical answers produce a **byte-identical**
`campaign.yaml` — the config is an artifact, not a session. `--non-interactive` takes every
default/flag for CI use. Exit codes: `2` usage, contradiction or a refused gate, `1` invalid
artifact or a blocked/failed stage, `0` success.

The prompts cover five things: the campaign root, the dataset selection (from the
known-dataset registry), the pod SSH target for the GPU stages, the S3 label-induction mode
(`per_chunk`, `doc` or `none`) and the S3 induction budget cap in USD. Every other knob of the
`s3:` block is flag-only (`--s3-chunk-max-tokens`, `--s3-extraction-model`,
`--s3-induction-model`, `--s3-no-guardrails` and the rest that `latence-campaign init --help`
lists): the defaults are the settings the recorded campaign ran, so a deviation is a
deliberate choice rather than an answer worth interrogating up front.

`campaign.yaml` holds three blocks: `campaign` (root, pod SSH target), `corpus` (the dataset
list) and `s3` (chunking, extraction, induction, labels, redaction). An unknown key inside
the `s3` block is rejected, because a mistyped knob that silently fell back to its default
would run the campaign on the wrong setting. An unknown top-level block is accepted and
ignored, so a file that still carries a `retrieval:` or `s5:` block loads.

## The stage chain

Order **is** the gate chain — each stage refuses to run until the previous one is approved,
and `approve` is an explicit operator sign-off, never automatic:

| Stage | What it does | Gate artifact |
| --- | --- | --- |
| `s1-corpus` | Acquires the selected public datasets under their **machine-enforced license lanes** (a research-only corpus requires `--accept-non-commercial`; an unverifiable one is refused unconditionally) | `stages/s1-corpus/metrics.json` + the corpus manifest, with per-dataset counts |
| `s2-ocr` | OCRs the PDF corpora (`ohr_bench`, `vidoseek`, `uda`) into markdown with served LightOn-OCR-2, page boundaries carried in a `.pagemap.json` sidecar beside each document; the text-native datasets are recorded as skipped | `stages/s2-ocr/metrics.json` + a spot-check directory of sampled pages |
| `s3-pipeline` | Generates one pipeline YAML per dataset from the `s3:` block and runs it on the pod: chunking, optional per-chunk label induction, fused entity and relation extraction, redaction, profiling, type consolidation, disambiguation, graph assembly, the knowledge-graph export and the corpus export with embeddings | `stages/s3-pipeline/metrics.json`, folded from each dataset's Quality Report |

Three details of `s3-pipeline` are worth knowing before you run it:

* **Induction follows the OCR boundary.** The three OCR'd datasets run LLM label induction
  under the budget cap; `multihop_rag` and `wiki2multihop` use the hand-authored label sets
  and make no LLM call. Mode `none` turns induction off everywhere.
* **The budget cap is converted into a call budget.** The inducer enforces a number of calls,
  not a number of dollars, so the stage turns the USD cap into one `max_induction_calls` per
  dataset when it generates the YAML. The allocation is proportional to the measured chunk
  counts and scaled down when the estimate would exceed the cap.
* **`musique` and `hotpotqa` are not processed.** Both stay selectable: `s1-corpus` references
  their source files in place and `s2-ocr` records them as text-native. `s3-pipeline` generates
  no pipeline for them and lists them under `not_processed` in its `metrics.json`. The stage
  used to verify an export of an earlier campaign for the two; that export fed the removed
  retrieval matrix, nothing reads it now, and the check was removed with it.

Each dataset's corpus and toolkit licence, and its lane, are declared in the stage's source
(`packages/latence-benchmark/src/latence_benchmark/campaign/corpus_stage.py`) and written to
`stages/s1-corpus/metrics.json` when the stage runs. Corpus-derived artifacts (raw
acquisitions, OCR output, exports, reports) stay local by `.gitignore` discipline; only
aggregate metrics, the corpus manifest, the chunk-validation report and the generated
pipeline files are committed.

## Interruption is a state, not a failure

`resume` finds the first stage that is not approved and acts on it; it never trusts a recorded
exit code. A pending or interrupted stage is run under the same gate rules as `run`, and its
done-probe is consulted **before** any work starts, so a stage whose artifact already exists
is marked done without re-running. For a done stage, `resume` prints the `approve` command
that opens the next gate.

A blocked stage records its reason in `STATE.json`, and `status` renders it. Blocked is
terminal: fix the cause, then re-run that stage with `latence-campaign run <stage> --fresh`.

Inside a stage, the work itself is resumable. The OCR worker skips every document it has
already written. Each pipeline run uses a fixed run id (`s3-<dataset>-v1`), and re-invoking
the Runner with the same run id replays its finalized checkpoints from disk, so a dead pod job
is relaunched under the same id and a live one is re-entered rather than started twice.

## What the committed campaign artifacts record

The files under `benchmark/sota-campaign/stages/` are the record of the campaign as it ran in
August 2026 (`s3-pipeline/metrics.json` is dated 2026-08-06, with one dataset re-run on
2026-08-09). They were produced by the pipeline of that date, which included parts that
ADR-0068 has since removed: Context Enrichment, hyperedges, graph features and the BM25
export. Three consequences:

* `s3-pipeline/metrics.json` contains `hyperedges` and `chunks_enriched` counts. They describe
  that run, not the current pipeline, which produces neither.
* The same file lists `musique` and `hotpotqa` inside `datasets`, as reused from an earlier
  export. The current stage writes no such entry; it lists the two under `not_processed`.
* The five files under `s3-pipeline/pipelines/` are the configurations that run used. They
  name an `enrich` Stage and config keys the current code rejects, so `latence stack check`
  fails on them. Running `s3-pipeline` again (`--fresh`) regenerates them in the current
  shape.
