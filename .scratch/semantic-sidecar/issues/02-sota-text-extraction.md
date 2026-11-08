# Research: the state of the art in text extraction for the upgraded text path

Type: research
Mode: AFK
Status: open
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
