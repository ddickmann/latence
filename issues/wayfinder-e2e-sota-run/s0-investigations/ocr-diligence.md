# S0 investigation — LightOn OCR diligence (2026-08-02)

Verdict: **LightOnOCR-2-1B is the right commit** — current olmOCR-bench leader (83.2 ± 0.9),
smallest and fastest of the top tier, and fully Apache-2.0 on both weights and serving stack.

## Model

- **`lightonai/LightOnOCR-2-1B`** (released 2026-01-19; paper arXiv 2601.14251). ~1B-param
  end-to-end VLM (native-res ViT from Mistral-Small-3.1 vision tower + Qwen3-based decoder),
  distilled from a Qwen3-VL-235B teacher over 43M pages.
- Variants: `-2-1B-bbox` (figure bounding boxes), soup merges, `-base` for fine-tuning;
  0.9B European-vocab pruned v1 variants exist (~96% retention).

## Licenses (weights vs code, verified separately)

- **Weights: Apache-2.0** (HF card + paper).
- **Serving code: Apache-2.0** — no dedicated LightOn repo; inference is upstream vLLM
  (≥0.11.1, native support) or Transformers v5+ (`LightOnOcrForConditionalGeneration`).
- Training code: UNVERIFIED (not publicly located) — irrelevant to our use.
- Output restrictions: none (Apache-2.0, no extra usage policy).
- Net: **clean for pipeline use, commercial-safe.**

## Fidelity

- Native markdown + LaTeX math. **v2 emits TABLES AS HTML inside the markdown stream**
  (v1 emitted markdown tables) — pipeline-design caveat: either the markdown parser accepts
  HTML tables or S2 adds an HTML→MD-table normalization pass. Figures = image placeholders
  (bbox only in `-bbox` variants); charts are NOT transcribed to data.
- olmOCR-Bench 83.2 overall (SOTA; Tables 84.8, Tiny Text 91.4, arXiv 89.6); OmniDocBench
  competitive; German: in-distribution (European-language training emphasis) but no published
  German number — quantitatively UNVERIFIED; CJK/Arabic degraded.

## Practicalities (16GB pod)

- ~2.5 GB BF16 → fits 16 GB comfortably with vLLM KV cache.
- Input = **image per page** (render PDFs at 200 DPI, longest edge 1540 px), not PDF-direct.
- Serve: vLLM with `--limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0
  --no-enable-prefix-caching`, temp 0.2 / top-p 0.9; on 16 GB add
  `--gpu-memory-utilization 0.85`, `--max-model-len 8192`.
- Throughput: 5.71 pages/s on H100 (published); 16 GB-class estimate ~0.5–1.5 pages/s
  (UNVERIFIED — no published mid-GPU numbers) → **~15–35 min per 1k pages** on the pod.
- Failure modes: ~1% repetition loops (flag via n-gram repetition ratio, retry once),
  handwriting inconsistent, blank-page hallucination suppressed by training.

## Alternatives sanity check (one line each; weights/code licenses verified)

| Model | Licenses | Note |
|---|---|---|
| olmOCR-2-7B | Apache/Apache | 82.3 bench at 8× the size; tight on 16 GB |
| Nanonets-OCR2-3B | Qwen RESEARCH (non-commercial) | disqualified by weights license |
| dots.ocr | MIT/MIT | strong, 5× slower |
| PaddleOCR-VL | Apache/Apache | strong, pipeline-style, heavier integration, 109 langs |
| DeepSeek-OCR | MIT/MIT | 75.7 bench — below |
| Granite-Docling-258M | Apache/MIT | DocTags only, Docling-internal |
| Marker | modified OpenRAIL-M (revenue clause) | restricted |

## S2 design consequences (carry into the stage spec)

1. Render step (pypdfium2, 200 DPI/1540 px) is separate + resumable from inference.
2. File-based manifest: one output per page, skip-if-exists, failures log → idempotent rerun.
3. Independent per-page requests (concurrency 8–32) so a crash loses nothing.
4. Post-check: n-gram repetition flag + single retry.
5. Decide the HTML-table normalization question against the parser inventory.
