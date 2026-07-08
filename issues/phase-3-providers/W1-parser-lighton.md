# W1-lighton — `latence-parser-lighton`: real LightOn-OCR Parser (Wave-1, GPU, Apache-2.0)

**Phase-3 Wave-1 GPU adapter. GitHub issue #91 (P3-A-parse, GPU).** A real OCR Parser over
**LightOn-OCR** (`lightonai/LightOnOCR-1B-1025`, **Apache-2.0** weights + code — license-verified
2026-07, cite the HF model card), built through `AdapterBase`. Chosen as the lighter, torch-based
alternative to PaddleOCR-VL: a compact end-to-end vision-language OCR model that emits **markdown**,
running on transformers/torch (already on the validation pod) rather than the paddle stack.

**Hard lesson baked in from the PaddleOCR-VL attempt (which was a false offline green):** the offline
stub MUST mirror the **real** `LightOnOCR` output shape, and a real-OCR test must be validated on the
GPU pod. If the stub and the real model output ever diverge, that is the bug. Do NOT invent an output
shape — mirror exactly what the real model returns (confirm the shape from the model card / a pod run;
LightOn-OCR is an image→markdown VLM, so the output is generated markdown text per page).

## What to build
Package **`latence-parser-lighton`** registering `parser.lighton`, an `AdapterBase` subclass
satisfying the `Parser` Protocol (bytes/`ParserInput` → `DocumentRecord` + `PageMap`), like
`parser.pdfplumber`/`parser.paddle`.
- **Real LightOn-OCR integration:** lazy-import; load the model (transformers `AutoModelForImageTextToText`
  / the model-card's documented loader, or vLLM if that's the sanctioned path — pick the simplest that
  runs on a single 16 GB GPU; document the choice). Feed the page image(s); the model returns **markdown
  text per page**. Take that text directly → per-page texts → `PageMap.from_page_texts` (the pdfplumber
  path, exact offsets, ADR-0031). No bbox/line model.
- Heavy deps (`transformers`, `torch`, `pillow`, maybe `vllm`, `accelerate`) live ONLY in this package
  (ADR-0016); nothing touches `latence-core`. Lazy import so provider discovery stays torch-free.
- **AdapterBase for real:** `compute="gpu"` (or `"either"` if CPU is tolerable-but-slow — decide +
  document); device routing through the shared seam (skip-with-flag on CPU here, `cuda` on the pod);
  native model/CUDA failure → typed `ProviderError`, **no image bytes / OCR text in the message**
  (C2/C6 — OCR output is PII-bearing); batching over pages/documents. Offset-0 type-spoof guard at
  intake (mirror pdfplumber/#62); malformed input → graceful PARSE_ERROR.

## The stub MUST match the real output (the anti-false-green rule)
The offline `Fake…`/monkeypatched model must return the **exact real LightOnOCR output shape** (the
generated markdown string per page, in whatever container the real API returns), documented against
the real contract. A conformance/e2e reviewer must be able to point at the stub and see it matches the
real model's output — this is the single thing whose divergence made PaddleOCR-VL a false green.

## Profile + license + conformance
- `ProviderProfile`: `compute="gpu"` (document `either`), honest `memory_mb` (a 1B VLM ~3–4 GB VRAM),
  `model_id="lightonai/LightOnOCR-1B-1025"`, `deterministic` truthfully (VLM decoding is often
  non-deterministic — declare False unless greedy+seeded proves otherwise; verify on the pod),
  `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** `license_code` = **Apache-2.0** (the loader lib, cited),
  `license_weights` = **Apache-2.0** (LightOnOCR-1B, cite the HF model-card URL + date). Both verified;
  passes conformance **C3**.
- **Passes P3-F3 conformance C1–C6** offline against the shape-matching stub. Roster count updates;
  conformance + stack-validate suites stay green.

## Wire-in
- CI: install `latence-parser-lighton --no-deps` in the heavy-provider step; tests monkeypatch the
  model with the shape-matching fake. A real-OCR test gated on `find_spec('transformers')` + a
  CUDA-available probe, **that overrides any autouse `LATENCE_CUDA=0` conftest pin** (setenv "1" /
  delenv) so it can actually route to cuda on a rig — skipped offline. Add `parser.lighton` to
  `matrix/parse.yaml` + `stacks/gpu-sota.yaml` (skip-with-flag on CPU, #64, no fabricated numbers).
- `scripts/validate-lighton-on-gpu.md`: turnkey pod steps (the exact pip install + a real-OCR run +
  `run_conformance`) so the maintainer's pod run (or my SSH run) validates the real model.

## Definition of Done (offline slice; real numbers come from the pod run)
- `latence-parser-lighton` builds, registers `parser.lighton`, is an `AdapterBase` subclass, satisfies
  `Parser`, and produces `DocumentRecord` + `PageMap` from the (faked, offline) model output whose
  shape MATCHES the real model. Heavy dep isolated. Native failure → typed error, no leak.
- Evidence-bearing Apache-2.0 license (weights+code, cited); passes conformance C1–C6; conformance +
  stack-validate green. The real-OCR test overrides the CUDA pin + is dep/CUDA-gated.
- `mypy --strict`/`ruff`/`pytest` green (via `scripts/verify-local.sh`); ADR-0036 note; seam-report;
  turnkey pod-validation script.

## Reviewer should check
- The stub's output shape is the documented real LightOnOCR output (the false-green trap).
- The real-OCR test genuinely routes to cuda (overrides the autouse pin) and asserts extracted text +
  offset round-trip — able to pass on a rig, not decorative.
- License really Apache-2.0 for weights AND code (cited); no heavy dep in core; GPU rows skip-with-flag.
- Markdown-per-page → PageMap.from_page_texts (exact offsets), not a reinvented mapper.
