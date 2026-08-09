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
5a. **OCR host = a dedicated big pod, FULL corpus** (Dennis, 2026-08-03, supersedes the
   fin-cap half of decision 1): OCR is the only heavy-lifting stage, so it runs UNCAPPED
   (~150k pages, all 788 fin docs) on a dedicated RTX PRO 6000 Blackwell 96GB pod
   (`ssh root@198.13.252.15 -p 11791`, driver 580/CUDA-13, stock vLLM 0.26 works natively);
   markdown results transfer back as input for the subsequent stages. Same pod scripts,
   scaled (max-num-seqs 64, worker concurrency 48). Full corpus = more reliable and
   comparable benchmark numbers. The small pod (paused at 4,862 pages/441 docs, work
   preserved) hosts everything downstream.
5. **Processing host = the existing RunPod pod** (Dennis, 2026-08-02):
   `ssh root@213.173.108.103 -p 11108 -i ~/.ssh/id_ed25519` — RTX 2000 Ada 16GB, verified
   alive with `/workspace/latence-framework` + `podcheck.sh` already in place from the M6
   campaign. Heavy stages (S2 vLLM OCR serving, S3 GLiNER2 extraction/graph build) run
   there under a pod-side supervisor with progress-freeze detection; the campaign wizard's
   `pod_ssh` config key carries this target.

6. **S3 spec** (Dennis, 2026-08-04): chunk.markdown 640/80/8; split label strategy
   (per-chunk LLM induction on OCR'd sets ~$90-120, hand-authored schemas for text sets);
   REUSE M6 v20 exports for musique+hotpot (comparability), newly process 2wiki +
   multihop_rag + OCR'd; redaction.gliner2 with guardrails ON. All wizard-applicable via a
   new `s3:` campaign.yaml block. Ticket: tickets/s3-pipeline-datagen.md. S2 gate closed
   (3,470 docs; OHR+ViDoSeek zero failed pages; UDA 74-doc residual accepted); OCR pod
   closed after hash-verified transfer (142,779 files) to benchmark/sota-corpus/s2-output.

7. **S3 first-launch incident + gate rework (2026-08-04)**: the first S3 launch silently
   processed ZERO documents for the OCR'd datasets — `SubprocessTransport.push` copied the
   `markdown` SYMLINKS (not their s2-output targets) to the pod, and a zero-doc run still
   wrote a quality report and passed as `done` (fake green). Both fixed + tested (64eac68):
   push now `--copy-links`, zero-doc/zero-chunk runs BLOCK. multihop_rag's partial real run
   (gliner2 had started) was stopped per Dennis's gate instruction — checkpoints kept for
   same-run_id resume. New gate flow per Dennis: chunk-quality validation (done locally,
   chunk-validation/REPORT.md — 297,548 chunks; UDA alone 220k, so induction-eligible
   chunks = 236,472 vs ~176k estimated) + an induction-model A/B
   (deepseek/deepseek-v4-flash-0731 vs gpt-4.1-mini, 10 real chunks, cost+quality) BEFORE
   any GLiNER2/induction processing runs. Zollklar OpenRouter key authorized by Dennis.

8. **S3 CLOSED + S5 spec (Dennis, 2026-08-09)**: all 6 datasets through the full-feature
   pipeline — 297,548 corpus rows / 3.43M mentions / 910k entities / **900,583 hyperedges**,
   ~$29 induction against the $150 cap, every export verified and held on the Mac. Corpus text
   is UNMASKED (`unsafe_unmasked_corpus`, core b70ab29): PII masking had replaced entity names
   in the embedded text — 82% of 2Wiki/MultiHop-RAG rows carried placeholders INCLUDING the
   article titles questions target, while the reused M6 exports were raw, so retrieval was
   neither valid nor cross-comparable. Detection still runs; only the substitution is skipped.
   S5 spec: **2–4 native hops only** (no dataset exceeds 4; synthesized deep hops rejected as
   non-comparable) and the leg ladder **dense → +bm25 → +KG → +hyperedges → +CoE → +packer**
   (SPLADE, MaxSim/MUVERA, rerank/fusion ablations OUT). Ticket: tickets/s5-retrieval-benchmarks.md.
   Seven scaling bugs fixed en route (dead O(N²) disambiguation loop, three whole-corpus
   materializations, three monitoring blind spots) — all with regression tests, all invisible
   below ~200k chunks.

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
