# W1-paddle — `latence-parser-paddle`: real PaddleOCR-VL OCR Parser (Wave-1, GPU)

**Phase-3 Wave-1, first GPU adapter. GitHub issue #91 (P3-A-parse, GPU).** A real OCR Parser over
**PaddleOCR-VL** (PaddlePaddle, **Apache-2.0**), built through `AdapterBase`. Unlocks the capability
the CPU parsers can't: **scanned / image-only PDFs** (no text layer) — `parser.pdfplumber` /
`parser.document` extract an existing text layer; PaddleOCR-VL *reads pixels*. It's the first
GPU-class provider, so it also proves the device-gated path end to end on a **real CUDA rig**.

**Real working plugin, validated on a real GPU pod (not a stub).** Two-part validation, both honest:
1. **Offline (here):** real adapter code; conformance C1–C6 against a monkeypatched OCR engine
   (the established gliner/embedder pattern — install `--no-deps`, fake the model); evidence-bearing
   license; `mypy --strict`/`ruff`/pytest green. The device seam skips it with-flag on this CPU host.
2. **On the RunPod CUDA pod (the maintainer provides SSH):** install PaddleOCR-VL + CUDA deps, run
   the REAL OCR over a scanned-PDF/image fixture, and validate the **real** offset/PageMap round-trip
   + a bake-off row with real quality/latency. **No fabricated numbers** — the real numbers come from
   the pod run and are folded into the PR; until then the GPU quality is explicitly "pending pod run."

## What to build (offline, this slice)
Package **`latence-parser-paddle`** registering `parser.paddle`, an `AdapterBase` subclass satisfying
the `Parser` Protocol (bytes/`ParserInput` → `DocumentRecord` + `PageMap`, like `parser.pdfplumber`).
- **Real PaddleOCR-VL integration:** lazy-import the engine; run OCR per page → text + per-line/word
  bounding boxes; assemble the page text in reading order and build the `PageMap`. Heavy deps
  (`paddlepaddle-gpu`/`paddleocr`) live ONLY in this package (ADR-0016); nothing touches `latence-core`.
- **Offsets are the hard part (validate on the pod):** OCR yields text + coordinates, not a text
  layer, so char offsets are assigned as the page text is assembled. Map them onto the document
  coordinate system via AdapterBase's provenance/offset helpers + the ADR-0031 seam; per-page pages
  correct; offsets must round-trip. Because the real offset behavior depends on the real OCR output,
  the **pod run** is what confirms the round-trip on a real scanned page — the offline test asserts
  the assembly logic against a fixed fake OCR result.
- **AdapterBase for real:** `compute="gpu"` (or `"either"` if CPU-Paddle is acceptable-but-slow —
  decide + document); device routing through the shared seam (skip-with-flag on CPU here, `cuda` on
  the pod); native OCR/CUDA failure → typed `ProviderError`, **no image bytes / OCR text in the
  message**; batching over pages/documents. Report any AdapterBase misfit (H-A1 discipline).
- **Type-spoof guard** at intake mirrors `parser.pdfplumber`/#62 (an image/PDF whose bytes don't
  match) — defense-in-depth; malformed input → graceful PARSE_ERROR.

## Profile + license + conformance
- `ProviderProfile`: `compute="gpu"` (document if `either`), honest `memory_mb`, `model_id` = the
  PaddleOCR-VL model, `deterministic` set truthfully (OCR on fixed input+config is typically
  deterministic — **verify on the pod**, declare False if not), `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1) — verify weights AND code separately (research-diligence
  directive):** `license_code` = PaddleOCR/PaddlePaddle SPDX (**Apache-2.0**, cited GitHub LICENSE),
  `license_weights` = the PaddleOCR-VL model SPDX (verify the model card — Apache-2.0 expected but
  **confirm**; if the model card doesn't substantiate it, `UNVERIFIED`/`False`, no guessing), with
  cited `license_source` + `license_verified_on`. Passes conformance **C3**.
- **Passes P3-F3 conformance C1–C6** offline against the monkeypatched engine (C2 no-PII-in-error, C6
  no-leak especially — OCR text can be PII). Roster count updates; conformance + stack-validate green.

## Wire-in
- CI: install `latence-parser-paddle --no-deps` in the heavy-provider step; tests monkeypatch the OCR
  engine. A real-OCR test gated on the dep + CUDA (`find_spec` + a cuda-available probe), skipped
  offline. Add `parser.paddle` to `matrix/parse.yaml` (GPU candidate → skip-with-flag offline, #64).
- Add `parser.paddle` to `stacks/gpu-sota.yaml` (skip-with-flag on the CPU host).

## Definition of Done (offline slice — the pod run is a follow-up step, not this PR's CI)
- `latence-parser-paddle` builds, registers `parser.paddle`, is an `AdapterBase` subclass, satisfies
  `Parser`, and produces a `DocumentRecord` + `PageMap` from a (faked, in offline CI) OCR result with
  a correct page map + assembled-offset logic. Heavy dep isolated (ADR-0016).
- Native failure → typed `ProviderError` with no image/OCR-text leak (C2); evidence-bearing license
  (C3, or honest UNVERIFIED pending the model-card check); passes conformance C1–C6 offline; the
  conformance + stack-validate suites stay green.
- `matrix/parse.yaml` + `gpu-sota.yaml` include `parser.paddle`, skip-with-flag on CPU (no fabricated
  numbers); a real-OCR test exists, gated on dep+CUDA.
- `mypy --strict`/`ruff`/pytest green; deterministic seeded; ADR-0036 + a docs note; the seam-report
  says whether AdapterBase fit an OCR parser (offset assembly especially).
- **A `scripts/validate-paddle-on-gpu.md` (or a small runnable script)** documenting the exact pod
  steps to install + run the REAL OCR validation, so the maintainer's pod run (or my SSH run) is
  turnkey. The real quality/latency/offset-round-trip numbers are produced on the pod and folded in
  after — flagged "pending pod validation" until then, never fabricated.

## Reviewer should check
- The adapter genuinely calls PaddleOCR-VL (real OCR path), not a permanent mock; the offline fake is
  only the CI double, and a real-OCR test is present + gated on dep+CUDA.
- No image bytes / OCR text leak into any error/log/report (C2/C6) — OCR output is untrusted PII-bearing content.
- License verified for BOTH code (Apache-2.0) and the model weights with cited sources, or honest UNVERIFIED.
- No heavy dep leaked into `latence-core`; the GPU candidate is skip-with-flag offline (no fabricated numbers).
- The offset-assembly logic is sound against the fake OCR result; the pod-validation script is turnkey.
