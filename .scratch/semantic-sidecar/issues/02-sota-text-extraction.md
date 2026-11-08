# Research: the state of the art in text extraction for the upgraded text path

Type: research
Mode: AFK
Status: resolved
Blocked by: 

## Question

Which models and serving set-up should replace the current extraction stack? The current
stack:

- GLiNER2 / GLiNER 2.5 fused NER plus relation extraction;
- `gliner-relex`;
- an LLM relation extractor with `gpt-4o-mini` as default;
- an LLM schema inducer with `gpt-4.1-mini` as default;
- the glinker linker.

Its documented weaknesses:

- cost grows with label count (O((n+m)²), with 20 labels or fewer advised);
- the 1,024-token window that is reliable in practice;
- relation extraction fabricates triples;
- uncalibrated confidences;
- 247,944 LLM calls for per-chunk induction on one campaign;
- the glinker stage is 73% of the 82-RFC dogfood run.

Cover:

- joint NER/RE and open information extraction models as of late 2026;
- small LLMs with structured output (JSON-schema decoding) used as extractors;
- type/schema induction approaches;
- German and English quality;
- long-document handling;
- calibration;
- licences, checking weights and code separately (ADR-0012);
- serving efficiency: vLLM, ONNX/TensorRT, batching, quantisation, and throughput per GPU class.

Compare against the baseline facts in `docs/adr/0064-*`, `packages/latence-label-profiler/README.md`
and `results/dogfood-rfc/`.

Resolves with a report at `research/02-sota-text-extraction.md`: a ranked shortlist of 2–4
candidate stacks for a bake-off (ADR-0036), each with its expected cost and quality and the
sources behind it.

## Answer

Resolved 2026-11-08 by research (AFK). Full report: [research/02-sota-text-extraction.md](../research/02-sota-text-extraction.md). Hugging Face and arXiv were blocked from the session, so external numbers come from search summaries. No licence meets ADR-0012's evidence rule yet.

- **Extraction is not the bottleneck.** In the 82-RFC run the glinker linker took 73% of 1,743 s and GLiNER2 extraction 15%. Across the campaign, per-chunk LLM schema induction (247,944 calls) capped throughput at about 3 chunks/s, against about 13 without it.
- **Never measured:** German quality, GLiNER 2.5 quality, calibration error, throughput on other GPU classes, and how the linker's time splits.
- **Strongest relation model: GLiNER-Relex** (May 2026). Zero-shot relation F1 25.6, against 22.1 for GPT-5-mini and 17.1 for GLiNER2. Document-level 31.3 against 12.4. A Relex provider already exists here, so testing it is cheap.
- **GLiNER 2.5 is not a quality step** (56.17 against 56.09), and the repo measured silent recall loss above about 1.3k tokens.
- **The GLiNER bi-encoder removes the label-count limit.** Labels are precomputed, so cost stays nearly flat up to 1,024 labels. That removes the main reason for per-chunk LLM induction. Its licence and German support are unconfirmed.
- **Permissive small LLMs for extraction:** Qwen3.5, Gemma 4 (now Apache-2.0, which ADR-0012 may need to revisit), Ministral 3, NuExtract3 4B. Excluded as non-permissive: LFM2-Extract, ReLiK, Triplex, MiniCheck, NuExtract 2.0 4B.
- **Fabricated relations need a separate check, not a better model.** First, each triple must point to text in the chunk at known offsets. Then an entailment check runs on the uncertain band (about +11% precision reported). Calibration is fitted per corpus with a conformal threshold.
- **Serving:** GLiNER-type models need ONNX or TensorRT with dynamic batching, not vLLM. LLM extractors use one fixed JSON schema per run so the grammar compiles once.
- **Bake-off shortlist, in order:**
  1. GLiNER-Relex plus the bi-encoder, with no LLM induction.
  2. Stack 1 plus a small LLM that checks only the uncertain triples.
  3. A small LLM doing extraction in one call per chunk.
  4. GLiNER 2.5 with the same fixes, as the control.
- **For other tickets:**
  - licences confirmed on the model cards, and the extraction-model egress decision (*the text-path upgrade*);
  - a human-labelled DE/EN gold set (*evaluation and the acceptance bars*);
  - profile the linker before any end-to-end timing (*record the baseline*);
  - byte-identical replay of LLM outputs (*the destination and the domain model*).
