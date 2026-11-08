# Research 02: the state of the art in text extraction for the upgraded text path

Ticket: [02-sota-text-extraction](../issues/02-sota-text-extraction.md). Date: 2026-11-08.

**How this was researched.** Repo baselines were read directly. External facts come from web search.
Direct fetches of `huggingface.co` and `arxiv.org` were blocked by this session's network policy, so
no model card or paper was opened. Every external number below comes from search-engine summaries
of those pages. **No licence here meets the ADR-0012 evidence bar yet.** Each one must be re-read
from the model card and the LICENSE file before a `ProviderProfile` claims it as verified. Claims
from a single secondary source, or from a vendor only, are marked **UNVERIFIED**.

## Summary

- **The extractor is not the bottleneck.** In the 82-RFC dogfood run, `fused_entity_relation.gliner2`
  took 266 s (15%) and `disambiguation.glinker` took 1,274 s (73%). On the campaign, per-chunk LLM
  induction (247,944 calls) bounded throughput. An upgrade must fix induction and the linker as well
  as the model, or it will not beat the baseline on throughput (map criterion 5).
- **For relations, GLiNER-Relex is the strongest permissive encoder found.** Its May 2026 paper
  reports a zero-shot RE average of 25.6 micro-F1, against 22.1 for GPT-5-mini and 17.1 for GLiNER2.
  It leads on DocRED with 31.3 against 12.4 for GLiNER2. A Relex checkpoint is already a Provider
  here (`relation.gliner_relex`), so a bake-off costs little.
- **GLiNER 2.5 is not a quality step over GLiNER2.** The vendor reports 56.17 against 56.09 macro-F1,
  with the gain concentrated in classification. The repo measured silent recall collapse above
  about 1.3k tokens.
- **The label-count wall has a known fix: the GLiNER bi-encoder.** It precomputes label embeddings,
  so cost stays near-constant from 1 to 1,024 labels. That removes the main reason for per-chunk LLM
  induction.
- **Small LLMs with JSON-schema decoding are now permissive and good enough to bake off.** Qwen3.5
  (0.8–9B), Gemma 4 (now Apache-2.0), Ministral 3 and NuExtract3 (4B) are all Apache-2.0 per their
  release sources. Their German extraction quality is unmeasured anywhere we found.
- **Fabrication needs a separate gate, not a better model.** The 2026 pattern is span-anchored
  triples plus an entailment or grounding check, for example PERSEUS, which reports +11% precision
  on CaRB. Calibration is domain-dependent, so it has to be fitted per corpus, with conformal
  risk control at the acceptance threshold.

## Baseline: what the repo measured, and what it did not

**Measured**

| Fact | Value | Source |
|---|---|---|
| Dogfood run, 82 RFCs, 3,565 chunks | 1,742.9 s total. glinker 1,274.4 s (73.1%), extract 266.0 s (15.3%), induce 79.5 s, redact 64.4 s. Extract runs about 13.4 chunks/s and emits 58,299 records, on a Blackwell GPU at 21.6 GB peak. | `results/dogfood-rfc/dataset/quality-report.md`, `PROOF.md` |
| Campaign (s3) | 297,548 chunks, 3.43 M mentions. Induction took 247,944 calls at $0.000118/call (DeepSeek-v4-flash, $29.26) against $0.00042/call for gpt-4.1-mini. With induction, `ohr_bench` ran about 3 chunks/s end-to-end. Without it, `wiki2multihop` ran about 13 chunks/s. | `benchmark/sota-campaign/stages/s3-pipeline/metrics.json` |
| Relation yield | `ohr_bench` produced 4,339 relations from 153,414 mentions at a threshold of 0.5, about 2.8 per 100 mentions. | same file |
| GLiNER 2.5 window | Coverage is 99–100% up to about 1,262 tokens. It falls to 34% at 2,438 tokens, and the curve is non-monotonic. One long chunk ran out of memory on a 16 GB card. A 1,024-token budget was chosen. | ADR-0064 §7 |
| Label count | The author advises 20 labels or fewer. Quality degrades past 30, the training cap is 25, and CPU latency rises 60% from 5 to 50 labels. | `latence-label-profiler/README.md` |
| Relation fabrication | 250 hyperedges were judged by an LLM: 54% correct, 32% partial, 14% incorrect. The high-confidence tail was the least reliable band. | `benchmark/official-data/musique-full/calibration-judgments.jsonl`, ROADMAP |
| Shipped calibration map | It changed the downstream metric by 0.003 or less at every budget in the dev A/B. | `calib-ab-dev.json` |

**Not measured.** Each gap below is something the bake-off has to produce.

- Entity or relation P/R/F1 against a human gold set, in either language. **German was never
  measured.**
- GLiNER 2.5 quality. The catalog itself says "UNMEASURED".
- Expected calibration error (ECE), or any calibration metric on extraction scores.
- Throughput on any GPU class other than the RTX 2000 Ada 16 GB and the Blackwell pod.
- How glinker's 1,274 s splits between the neural linker and the resolver cascade.
- The quality of `relation.llm` against GLiNER on the same corpus.

## Candidate landscape

"W" is the weights licence and "C" the code licence. "Search" means the licence was read from
search summaries of the card or release post and must be re-verified.

| Model | Task | Params | Licence W / C | Languages | Reported quality | Serving notes | Source |
|---|---|---|---|---|---|---|---|
| `fastino/gliner2.5-multi-v1` (**baseline**) | Fused NER+RE, classification, JSON | 287M (mDeBERTa-v3) | Apache-2.0 / Apache-2.0 (repo-verified 2026-08-25) | Multilingual | Zero-shot average over 16 datasets: 56.17 against 56.09 for GLiNER2-multi. The extraction average *dipped* (vendor). | The boundary decoder is linear in sequence length (vendor). The repo measured collapse above about 1.3k tokens. ONNX ports exist from the community. | [MarkTechPost][g25], ADR-0064 |
| GLiNER-Relex (`knowledgator/gliner-relex-large-v1.0`, `-multi-v1.0`) | Joint NER+RE | DeBERTa-based. Size **UNVERIFIED**. | Apache-2.0 (multi-v1.0, repo-verified) / Apache-2.0 (`gliner`) | multi-v1.0 is multilingual. German **UNVERIFIED**. | RE micro-F1: CoNLL04 40.4, DocRED 31.3, FewRel 12.5, CrossRE 18.1. Average 25.6, against 22.1 for GPT-5-mini and 17.1 for GLiNER2. | About 0.9 s/doc on an L4, against about 64 s/doc for GPT-5-mini (secondary, **UNVERIFIED**). ONNX export. Separate entity and relation thresholds. | [arXiv 2605.10108][relex], catalog |
| GLiNER bi-encoder v2 (`gliner-bi-base/large-v2.0`) | NER (many labels) | 194M / 530M | Apache-2.0 **UNVERIFIED** / Apache-2.0 | German **UNVERIFIED** | CrossNER zero-shot 60.3 (base) and 61.5 (large) | Label embeddings are precomputed, so cost is near-constant from 1 to 1,024 labels. Up to 130× faster at 1,024 labels and 2–3× faster with dozens of labels (vendor). | [arXiv 2602.18487][bi] |
| GLiFormer (`gliformer-large-v1`, `gliformer-multi-base-v1`) | Multitask: NER, RE, classification, hierarchical JSON | 575.6M (large) | Apache-2.0 per a community discussion, **UNVERIFIED** | The multi-base variant is multilingual, **UNVERIFIED** | "On par with 20× larger" (author post). A 16,384-token context is claimed. Both **UNVERIFIED**. | `gliformer` package. ONNX and LiteRT ports exist. | [HF card][glif], [HackerNoon][glif2] |
| SauerkrautLM-GLiNER | NER | ~307M (mmBERT-base) | MIT per a mirror, **UNVERIFIED** / `gliner` Apache-2.0 | EN, DE, FR, IT, ES | +23.02 F1 over `gliner_multi-v2.1` (vendor, **UNVERIFIED**) | The ModernBERT backbone has 8,192-token native context (mmBERT, MIT). It avoids DeBERTa's quadratic disentangled-attention memory. | [demo][skg], [mmBERT][mmbert] |
| NuExtract3 | Schema→JSON extraction plus OCR, with optional reasoning | 4B (Qwen3.5-4B) | Apache-2.0 (search) / code at `numindai/nuextract`, licence **UNVERIFIED** | Tagged "multilingual". German **UNVERIFIED**. | Only an internal benchmark of about 600 documents. No independent results. | vLLM and SGLang. W8A8 and GGUF variants. Temperature near 0. | [NuMind blog][nu3], [W8A8 card][nu3q] |
| NuExtract 2.0 2B / 8B | Schema→JSON | 2B / 8B | MIT (2B, 8B). **The 4B is Qwen Research licence (non-commercial): exclude it.** | Multilingual | Vendor claims it beats frontier LLMs | 32k context | [NuExtract-2.0 card][nu2] |
| Qwen3.5 0.8B / 2B / 4B / 9B | General LLM used as an extractor with constrained decoding | 0.8–9B | Apache-2.0 (card) / vLLM Apache-2.0 | Multilingual. 262K context. | Community document KIE: 86.5 (9B) and 86.0 (4B), **UNVERIFIED** | Hybrid Gated DeltaNet, with a reported weakness on exact recall. Thinking must be disabled with `enable_thinking: false`. | [Qwen3.5 card][q35], [Artificial Analysis][q35aa] |
| Gemma 4 E2B / E4B / 26B-MoE / 31B | General LLM | 2.3B / 4.5B effective | **Apache-2.0** (it changed from the Gemma terms on 2026-04-02) | 140+ languages | E4B reaches 100% valid JSON, but one IDP discussion reports weak field accuracy against Qwen3.5-4B, **UNVERIFIED** | 128K context (E-series) | [The Register][g4], [field report][g4f] |
| Ministral 3 (3B / 8B / 14B) | General LLM | 3–14B | Apache-2.0 | Multilingual. German **UNVERIFIED**. | No extraction benchmark found | 256K context. Base, instruct and reasoning variants. | [arXiv 2601.08584][min3] |
| LFM2-Extract (350M / 1.2B) | Schema→JSON | 0.35B / 1.2B | LFM Open v1.0: **free only under $10M revenue, not permissive** | — | — | Opt-in at most | [LFM licence][lfm] |
| ReLiK | Entity linking + RE | — | CC BY-NC-SA 4.0 (secondary): **exclude** | EN | — | — | [SourcePulse][relik] |
| Triplex | KG triple extraction | 3.8B | CC BY-NC-SA with a revenue waiver: **exclude** | EN | — | — | [HF][triplex] |
| Bespoke-MiniCheck-7B / LettuceDetect | Grounding verifier | 7B / ModernBERT | CC BY-NC 4.0: **exclude** / MIT (paper) | EN. German **UNVERIFIED**. | — | LettuceDetect has 4–8k context | [MiniCheck][mc], [LettuceDetect][ld] |
| GLiGuard (reference for serving) | PII and safety | 145–209M | — | — | — | On an A100: 193 RPS, 1.6× GLiNER2. GLiNER2-multi under TensorRT FP16 ran 122.6 RPS at 740 ms P50. | [arXiv 2605.05277][gg] |

## Findings by area

**Joint NER/RE and open IE.** Only GLiNER-Relex publishes a document-level RE head-to-head (DocRED
31.3 vs 12.4), which is where our multi-sentence fabrication lives. All RE F1s are below 41, so no
model promotes triples safely without a gate. GLiREL is relation-classification only. 2026 open IE
is LLM-based and paired with verification.

**Small LLMs as extractors.** Constrained decoding (XGrammar/LLGuidance in vLLM) makes JSON validity
near-free in principle (XGrammar: 9.0→9.2 ms TPOT at batch 16, on MLC-LLM). In vLLM, guided decoding
reportedly loses throughput at batch ≥8 and grammars are cached per schema (secondary,
**UNVERIFIED**). So: **one fixed JSON schema per run**, types as free strings validated afterwards; a
per-chunk `enum` of induced labels would recompile every request. Valid JSON is not correct JSON.

**Type/schema induction.** EDC (EMNLP 2024) and AutoSchemaKG (ACL 2026; 92–95% alignment with human
schemas, vendor) induce *after* open extraction, corpus-wide, matching ADR-0054. The 247,944 calls
come from per-chunk induction *before* extraction. Either an extractor that emits open types itself
(induce and extract in one call) or a bi-encoder scored against the label profiler's whole pool
(label count becomes free) moves the calls from O(chunks) to O(sample) plus one canonicalization
pass over distinct type strings.

**German and English.** No 2025–26 German result was found for any candidate. Indirect evidence:
GLiNER-multi beat ChatGPT on Latin-script languages incl. German (MultiCoNER); SauerkrautLM-GLiNER
trained on DE/EN/FR/IT/ES; mmBERT covers 1,800+ languages. Zero-shot claims carry a contamination
caveat (ZeroNER, ACL Findings 2025). **The bake-off must bring its own German gold set.**

**Long documents.** DeBERTa backbones are memory-quadratic (the repo's 16 GB OOM). ModernBERT/mmBERT
(8k, local/global attention) is the route to longer encoder windows, but no GLiNER-RE head on one
was found; GLiFormer's 16k is **UNVERIFIED**. LLMs advertise 32k–262k, but extraction recall at
length is unmeasured; the ADR-0064 coverage harness applies unchanged to any offset-returning
extractor.

**Calibration.** Miscalibration flips direction across domains (Shrestha & Kim 2026), so a shipped
global map is the wrong unit, consistent with our flat A/B. Pipeline-aware conformal prediction
(PASC) held 96.4% end-to-end coverage vs 86.5% for per-stage CP. Recommendation: a **per-corpus
conformal FDR threshold** at triple acceptance; for LLMs, score from constrained-decoding log-probs
or k-sample agreement.

**Fabrication.** Span-anchoring plus entailment: AEVS anchors elements to offsets and sends only
partial matches to an LLM; PERSEUS adds NLI (+11% precision on CaRB). Both fit our offset contract.
A free deterministic first gate (subject, object and evidence quote must be substrings of the chunk)
then an entailment check on the uncertain band only.

**Serving.** GLiNER is not a vLLM architecture (vLLM token classification lists ModernBERT/BERT/
XLM-R, not DeBERTa), so encoders run on PyTorch/ONNX Runtime/TensorRT with dynamic batching:
GLiNER2-multi 122.6 RPS at TensorRT FP16 on an A100; ONNX INT8 about 2× on CPU but lossy (community).
For 8B LLMs on vLLM, public output throughput is roughly 1.8k tok/s (L4), 4k (A100), 8–12k (H100
FP8), inconsistent across sources; no 4B vLLM numbers were found.

## Ranked shortlist for the bake-off (ADR-0036)

All arms share chunks, a new human EN+DE gold set and the downstream type-consolidation stage, and
report entity/relation P/R/F1, unsupported-triple rate, ECE, chunks/s per GPU class and cost.
**Arm 4 is the control** and always runs.

### 1. Encoder-only, induction-free (recommended first)

- **Stack.** GLiNER-Relex (large-v1.0 EN, multi-v1.0 DE) for joint NER+RE, plus bi-encoder v2 NER
  against the label profiler's *whole* pool; per-chunk LLM induction dropped; deterministic
  span-anchoring gate and per-corpus conformal threshold.
- **Cost.** Zero LLM calls (minus 247,944 per campaign); extraction at about baseline's ~13
  chunks/s or better (estimate); permissive pending re-verification.
- **Quality.** +8.5 average RE F1 over GLiNER2 on published benchmarks; NER over 40+ types should
  hold (CrossNER 61.5).
- **Risks.** Bi-encoder German unverified; two models to reconcile; 1,024-token window remains; RE
  F1 still under 41, so the gate is load-bearing.

### 2. Encoder + small-LLM verifier cascade

- **Stack.** Arm 1 at high recall; Qwen3.5-4B or Ministral 3 8B (Apache-2.0) on vLLM FP8 *verifies
  and types* only triples in the uncertain conformal band, fixed JSON schema, offset-checked
  evidence quote.
- **Cost.** LLM calls on a fraction of candidate triples (fraction **UNVERIFIED**; measure it).
- **Quality.** Best expected precision per dollar: short anchored questions, not open extraction.
- **Risks.** Most moving parts. vLLM batching is not bit-deterministic, so LLM outputs must be
  stored as Evidence and replayed from cache to meet map criterion 1.

### 3. Small-LLM extractor

- **Stack.** NuExtract3 (4B) or Qwen3.5-4B/9B does open-type NER+RE in **one call per chunk**,
  replacing induce and extract; corpus-level EDC-style canonicalization afterwards; same gates.
- **Cost.** About 300 output tokens per chunk gives about 89 M tokens per campaign: roughly 3
  H100-hours at 8k tok/s or about 14 L4-hours (estimate).
- **Quality.** Highest ceiling on relation semantics and German (unmeasured); long context
  available.
- **Risks.** Far below encoder throughput on small GPUs; replay; grammar overhead at batch ≥8;
  model-written offsets need substring re-anchoring; Gemma 4/NuExtract3 accuracy is anecdotal.

### 4. Control: GLiNER 2.5 + fixes

- **Stack.** Baseline `fused_entity_relation.gliner25` at 1,024 tokens; sampled corpus-level
  induction (about 1–2k calls instead of 248k); same gates and threshold.
- **Cost.** Lowest change cost; isolates the gates' contribution from the model's.
- **Risks.** The ~20-label ceiling and window collapse stay.

**Decisive for map criterion 5, outside extraction.** glinker is 73% of runtime: profile its
1,274 s (linker vs resolver) before timing any arm end-to-end. GLinker's vendor attributes 10–100×
linking speedups to precomputed label embeddings.

## Open questions

1. Which of Knowledgator's checkpoints (Relex large and multi, bi-encoder v2, GLiFormer) carry
   Apache-2.0 *weights*? This must be confirmed on the cards, with the LICENSE file. Was the
   bi-encoder trained on any German data?
2. Does GLiNER-Relex share GLiNER 2.5's silent long-input collapse? Re-run the ADR-0064 coverage
   harness on each candidate.
3. Who labels the EN+DE gold set, and is it human-labelled? The current judged sample is LLM-labelled
   (ROADMAP).
4. Can LLM-extractor outputs be stored as Evidence so that replay is byte-identical? Seeded vLLM is
   not enough. This needs a decision record.
5. Sending samples to a hosted model needs an egress decision (map). Local vLLM arms avoid it.
6. Gemma 4 is now Apache-2.0. Does ADR-0012's "Gemma opt-in" rule need an amendment scoped to
   Gemma ≤3?
7. What fraction of triples falls in the uncertain band under conformal FDR control at, say, 10%?
   This sets the cost of arm 2.

## Sources

[relex]: https://arxiv.org/abs/2605.10108
[g25]: https://www.marktechpost.com/2026/08/24/fastino-releases-gliner2-5-a-boundary-prediction-architecture-that-removes-span-enumeration-from-information-extraction/
[bi]: https://arxiv.org/html/2602.18487
[glif]: https://huggingface.co/knowledgator/gliformer-large-v1
[glif2]: https://hackernoon.com/understanding-gliformers-benchmarks-and-schema-driven-extraction
[skg]: https://vagosolutions-sauerkrautlm-gliner-demo.hf.space
[mmbert]: https://huggingface.co/jhu-clsp/mmBERT-base/blob/main/README.md
[nu3]: https://numind.ai/blog/nuextract-3-release
[nu3q]: https://huggingface.co/numind/NuExtract3-W8A8
[nu2]: https://huggingface.co/numind/NuExtract-2.0-2B
[q35]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md
[q35aa]: https://artificialanalysis.ai/articles/qwen3-5-small-models
[g4]: https://www.theregister.com/2026/04/02/googles_gemma_4_open_weights/
[g4f]: https://mpklu.github.io/posts/gemma-4-benchmark
[min3]: https://web3.arxiv.org/pdf/2601.08584
[lfm]: https://liquid.ai/lfm-license
[relik]: https://www.sourcepulse.org/projects/2396418
[triplex]: https://huggingface.co/SciPhi/Triplex
[mc]: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B/blob/main/README.md
[ld]: https://arxiv.org/html/2502.17125v1
[gg]: https://arxiv.org/html/2605.05277v1

- GLiNER-Relex: [arXiv 2605.10108][relex]. GLiNER 2.5: [MarkTechPost][g25], [Fastino blog](https://fastino.ai/blog/gliner2-5-span-free-information-extraction).
- Bi-encoder: [arXiv 2602.18487][bi], [gliner-bi-base-v2.0](https://huggingface.co/knowledgator/gliner-bi-base-v2.0). GLinker: [PyPI](https://pypi.org/project/glinker/).
- GLiFormer: [card][glif], [licence discussion](https://huggingface.co/knowledgator/gliformer-large-v1/discussions/1), [HackerNoon][glif2].
- German: [SauerkrautLM-GLiNER][skg], [mmBERT][mmbert], [GLiNER NAACL 2024](https://aclanthology.org/2024.naacl-long.300.pdf), [ZeroNER](https://aclanthology.org/2025.findings-acl.805.pdf).
- LLMs: [NuExtract3][nu3], [NuExtract 2.0][nu2], [Qwen3.5][q35], [Qwen3.5 (Artificial Analysis)][q35aa], [Qwen 3.6–3.8 overview](https://codersera.com/blog/qwen-3-5-complete-guide-2026/), [Gemma 4][g4], [Gemma 4 field report][g4f], [Ministral 3][min3], [Qwen3.5 thinking switch (Unsloth)](https://unsloth.ai/docs/models/qwen3.5.md).
- Induction: [EDC](https://aclanthology.org/2024.emnlp-main.548/), [AutoSchemaKG](https://arxiv.org/abs/2505.23628v3).
- Calibration: [PASC](https://arxiv.org/html/2605.18812), [Shrestha & Kim](https://arxiv.org/html/2603.00924v2).
- Fabrication: [AEVS](https://www.mdpi.com/2073-431X/15/3/178), [PERSEUS](https://ceur-ws.org/Vol-4246/llms4kgoe-5.pdf), [Enoki](https://arxiv.org/pdf/2609.00581), [MiniCheck][mc], [LettuceDetect][ld].
- Serving: [GLiGuard][gg], [XGrammar](https://arxiv.org/pdf/2411.15100), [vLLM structured-output thread](https://discuss.vllm.ai/t/general-questions-on-structured-output-backend/1444), [structured-output guide](https://www.spheron.network/blog/structured-output-function-calling-inference-guide/), [vLLM token classification](https://docs.vllm.ai/en/latest/models/pooling_models/token_classify/), [GLiNER2.5 ONNX INT8](https://huggingface.co/nishparadox/gliner2.5-decide-onnx), [L4 vs A100](https://jarvislabs.ai/blog/l4-vs-a100), [vLLM benchmarks (E2E Networks)](https://docs.e2enetworks.com/tir/benchmarks/inference_benchmarking.html).
- Exclusions: [LFM licence][lfm], [ReLiK][relik], [Triplex][triplex].
