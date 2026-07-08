# W1-glm — `latence-parser-glm`: real GLM-OCR Parser (Wave-1, GPU, MIT)

**Phase-3 Wave-1 GPU adapter. GitHub issue #91 (P3-A-parse, GPU).** A real OCR Parser over
**GLM-OCR** (`zai-org/GLM-OCR`, Zhipu/Z.ai — **MIT** weights + code; layout component Apache-2.0 —
license-verified 2026-07, cite the HF model card), built through `AdapterBase`. The second permissive
OCR provider alongside `parser.lighton`: a compact 0.9B image→markdown VLM, transformers-native
(`GlmOcrForConditionalGeneration`, contributed to HF transformers), so it validates cleanly on the
torch GPU pod (confirmed: the class imports on the pod's transformers).

**Build it as a near-twin of the pod-validated `latence-parser-lighton` (#125) — reuse its shape.**
LightOn-OCR is the reference: same AdapterBase structure, same image→markdown model, same
generate→**SLICE-prompt-tokens**→decode contract, same PageMap-from-page-texts offset path, same
type-spoof + guard + skip-with-flag + evidence-license discipline. Copy that proven design; change the
model + loader + license, not the architecture.

## Non-negotiables carried from the LightOn + PaddleOCR lessons
1. **Decode contract (the false-green trap):** `generate()` returns prompt + continuation; the adapter
   MUST slice `output_ids[:, inputs["input_ids"].shape[1]:]` before `processor.decode`, or the OCR
   markdown is prefixed with chat/prompt scaffolding. The offline **fake must be FAITHFUL** to the
   real `generate` contract (return prompt-tokens + continuation; an un-sliced provider must FAIL an
   anchor test). Do NOT let the stub mirror a buggy provider — that is exactly what made the first
   PaddleOCR attempt a false green.
2. **Real-OCR test must be able to pass on a rig:** gate on `find_spec('transformers')` + a
   CUDA-available probe, and **override the autouse `LATENCE_CUDA=0` conftest pin** (`monkeypatch.setenv
   "1"` / `delenv`) so it genuinely routes to cuda; assert extracted TEXT fidelity (legible content,
   no `<|...|>` scaffolding) + offset round-trip. Skipped offline.
3. **Pixel-bomb DoS cap (from LightOn's security finding):** this parser rasterizes untrusted input.
   Clamp the render scale by a max-megapixel budget (probe page size via pypdfium2, reduce scale to
   stay under ~25 MP) and set a conservative `Image.MAX_IMAGE_PIXELS` before `.load()`; a fixture
   asserts an oversized page/image → graceful PARSE_ERROR (not a multi-GB allocation). Build this in
   from the start (LightOn deferred it to a follow-up; GLM should not repeat that).

## What to build
Package **`latence-parser-glm`** registering `parser.glm`, an `AdapterBase` subclass satisfying the
`Parser` Protocol. Loader: `GlmOcrForConditionalGeneration.from_pretrained("zai-org/GLM-OCR",
dtype=torch.bfloat16, device_map=…)` + `AutoProcessor` (lazy import; heavy deps transformers/torch/
pillow/pypdfium2 ONLY in this package, ADR-0016). Image (or rasterized PDF page) → chat message →
generate → slice → decode → **markdown string per page** → `PageMap.from_page_texts` (exact offsets,
ADR-0031). `compute="gpu"`, device seam skip-with-flag on CPU; native fault → typed `ProviderError`
with no image/OCR-text leak (C2/C6); offset-0 type-spoof guards (PDF+image, #62).

## Profile + license + conformance
- `ProviderProfile`: `compute="gpu"`, honest `memory_mb` (~3–4 GB for 0.9B bf16), `model_id=
  "zai-org/GLM-OCR"`, `deterministic` truthfully (VLM decoding — declare False unless greedy+seeded
  proven; verify on the pod), `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** `license_code` = **MIT** (+ note the Apache-2.0 layout
  component), `license_weights` = **MIT** (cite the HF model-card URL + date). Verified; passes C3.
- **Passes P3-F3 conformance C1–C6** offline against the faithful stub; roster count updates;
  conformance + stack-validate stay green.

## Wire-in
- CI/verify-local: add `latence-parser-glm --no-deps` to the heavy-provider install (ci.yml **and**
  `scripts/verify-local.sh`), the wheel-build list, and `parser.glm` to `matrix/parse.yaml` +
  `stacks/gpu-ocr.yaml` (skip-with-flag on CPU, #64, no fabricated numbers).
- `scripts/validate-glm-on-gpu.md`: turnkey pod steps (pip install + real-OCR run + `run_conformance`).

## Definition of Done
- `latence-parser-glm` builds, registers `parser.glm`, AdapterBase subclass, `Parser`, produces
  DocumentRecord+PageMap from the (faithful) fake whose shape matches the real generate→slice→decode
  markdown; the pixel cap is present with a fixture; native failure → typed error, no leak.
- Evidence-bearing MIT license (weights+code, cited); passes conformance C1–C6; conformance +
  stack-validate green; the real-OCR test overrides the CUDA pin, asserts text fidelity + offsets,
  gated on dep+CUDA.
- `bash scripts/verify-local.sh` green (Actions is out of minutes — local gate is the merge gate);
  ADR-0036 note; seam-report; turnkey pod-validation script. (Real GPU numbers come from the pod run
  I'll do over SSH — never fabricated.)

## Reviewer should check
- The stub is faithful to the real `generate`→slice→decode contract (un-sliced provider FAILS) — the
  false-green trap; the real-OCR test can actually pass on a rig (overrides the pin, asserts text).
- The pixel-bomb cap is present + tested (this parser rasterizes untrusted input).
- MIT license verified for weights AND code (cited); no heavy dep in core; GPU rows skip-with-flag.
- Mirrors the pod-validated `parser.lighton` design (markdown-per-page → PageMap.from_page_texts).
