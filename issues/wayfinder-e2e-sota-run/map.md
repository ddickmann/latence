# Wayfinder map — End-to-End SOTA Corpus Run (full-feature pipeline + retrieval benchmarks)

Label: `wayfinder:map`

## Destination

The framework runs its FULL end-to-end capability — OCR-to-markdown entry, the complete
data-generation pipeline with the most advanced features enabled, graph + hypergraph build,
and the retrieval benchmark suite — over a **final corpus composed of SOTA benchmark
datasets**, producing inspectable metrics at every stage and a final benchmark report with
published-number comparisons. Delivered as a **stage-gated, resumable campaign wizard CLI**
(config-schema-first, so a UI frontend can wrap it later without redesign).

## Operating contract (Dennis, 2026-08-02 — binding for every stage)

1. **Stage-gated, never one forced end-to-end run.** Each stage ends with durable artifacts
   + a metrics/inspection summary; Dennis reviews before the next stage starts.
2. **Resumable by construction.** All progress lives in files (campaign config + per-stage
   STATE + artifacts); any session continues from the last completed stage.
3. **OCR gate**: input not already in markdown goes through the LightOn OCR step first to
   enforce proper markdown (stage-1 investigation verifies model + licenses before commit).
4. **Wizard-as-CLI**: every option configurable via CLI + declarative config file; the CLI
   is the API a future UI frontend calls. No interactive-only paths.
5. **Investigate → spec → run, per stage.** Before each stage: an investigation of the
   available features/options; Dennis decides the spec; only then does the stage run.
6. Subagents execute the heavy work; long runs get an on-host supervisor with
   progress-freeze detection (carved lesson, never bare polling).

## Standing rules carried over

- **Sacrifices only with data** (Dennis, 2026-07-31); shipped default = strong configuration.
- **No fake green**: real models, real datasets, real deps for every claim.
- **License discipline** (research-diligence directive): every dataset and model verified
  weights-vs-code / dataset-vs-underlying-text separately BEFORE acquisition; share-alike
  material stays local/gitignored; UNVERIFIED is a blocking state, not a footnote.
- Default pipeline behavior stays byte-identical for existing users; campaign features are
  additive/opt-in.

## Stage plan (skeleton — each stage gets its spec ticket AFTER its investigation)

| Stage | Name | Gate artifact Dennis inspects |
|---|---|---|
| S0 | Charting investigations (RUNNING) | 3 reports: OCR model diligence, dataset landscape + licenses, framework feature/config inventory |
| S1 | Corpus selection + acquisition | corpus manifest (datasets, sizes, licenses, handling class) + downloaded raw data |
| S2 | OCR → markdown normalization | conversion metrics (pages, failures, spot-check samples) + markdown corpus |
| S3 | Pipeline data generation (full-feature spec) | per-dataset Quality Reports + extraction metrics |
| S4 | Graph + hypergraph build + exports | graph stats (nodes/edges/hyperedges), KV stores, export integrity checks |
| S5 | Retrieval benchmarks | per-tier curves, A/Bs, gates, recall@5, SOTA comparisons |
| S6 | Final report | consolidated benchmark report + campaign retrospective |

Stage boundaries are the INSPECTION points; internally a stage may reuse the Runner's own
checkpointing. S0's inventory decides whether S3/S4 are one pipeline run inspected at the
Runner's stage checkpoints or two separate runs.

## Wizard architecture (to be confirmed by the S0 inventory)

- `campaign.yaml` — the single declarative config (corpus composition, OCR settings, full
  pipeline spec, benchmark spec). The wizard's `init` subcommand builds it interactively;
  every prompt maps 1:1 to a config key (UI-frontend-ready).
- Per-stage `STATE.json` + `metrics.json` + artifacts under a campaign directory; `status`
  renders the gate summary; `run <stage>` refuses to start if the previous gate is not
  approved; `resume` continues an interrupted stage.
- Placement candidate: `latence-benchmark` package beside the `latence-label` CLI precedent
  — confirmed or revised by the S0 inventory of existing CLIs.

## Decisions so far

**S0 gate (Dennis, 2026-08-02)** — all four recommendations accepted:

1. **Corpus = the full 6+2**: MuSiQue + HotpotQA (incumbents, local) + 2WikiMultiHopQA +
   MultiHop-RAG + OHR-Bench + ViDoSeek + UDA (FinHybrid/PaperTab). License lanes per
   [dataset-landscape.md](s0-investigations/dataset-landscape.md); share-alike/research-only
   material local + gitignored, never redistributed.
2. **OCR = LightOnOCR-2-1B served** via the existing `parser.lighton_vllm` provider
   (already its default model) + vLLM on the 16GB pod.
3. **Wizard = new `latence-campaign` CLI** in the latence-benchmark package, mirroring
   `latence setup` house style; productizes supervise.sh's stage-table/probe/STATE.json
   shape in Python.
4. **Schema induction = LLM via OpenRouter** (`label_inducer.llm`) in the S3 full-feature
   profile — offline indexing cost only, online path stays zero-LLM; exact budget estimate
   presented before the S3 run is approved.

## Fog

- Whether tier-2 query NER and the (unbuilt) eu_vecs/dense leg are in scope for S5 or the
  benchmark runs ship per-tier as in the official M6 run — decide at S5 spec time against
  the CLOSEOUT-AUDIT residuals (F-VEC/F-BASE).
- GPU budget: OCR + GLiNER2 at full-corpus scale on the 16GB pod — sized at S1 once corpus
  scale is known.
- Whether the datagen-quality map's quick wins (e.g. relation-label pruning) land BEFORE S3
  so the corpus benefits, or S3 runs the current extraction for comparability — Dennis's
  call at S3 spec time.

## Out of scope

- UI frontend itself (the CLI contract is the deliverable that enables it).
- Changing retrieval/algorithm internals (that is the parent hypergraph map's territory);
  this campaign RUNS the framework at full strength and productizes the orchestration.
