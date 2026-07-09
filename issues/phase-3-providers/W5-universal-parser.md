# W5-universal-parser — `parser.render`: any-file → page-images/text universal document parser

**Ports the colsearch batch converter (any file → batches of images), license-clean.** A universal
document Parser that turns ANY common file type into a `DocumentRecord` + `PageMap`, dispatching by
type — the "files to images" front-end an OCR pipeline needs so a real corpus (PDF, office docs,
images) all flow through one Parse Stage.

## License discipline (research diligence — the whole reason for a clean port)
colsearch's `preprocessing.py` renders PDF with **PyMuPDF (`fitz`) — AGPL-3.0** (copyleft). Do NOT
vendor or depend on `fitz`. Use only permissive deps, the same the framework already ships:
- **PDF → images:** `pypdfium2` (Apache-2.0/BSD — already the `parser.lighton` renderer). NOT fitz.
- **Office (docx/pptx/xlsx):** prefer **LibreOffice headless** (`soffice --convert-to pdf`, MPL-2.0, a
  subprocess — faithful page render) → pypdfium2 images, WHEN `soffice` is on PATH; else fall back to
  **python-docx / openpyxl** (MIT) text extraction → markdown (no OCR). `soffice` is OPTIONAL — a host
  without it degrades to text extraction, never a crash (skip-with-flag / graceful).
- **Images:** `Pillow` (permissive) — a bare image is a one-page document.
All heavy deps isolated in this package (ADR-0016); lazy imports so discovery stays light.

## What to build
Package **`latence-parser-render`** registering `parser.render`, an `AdapterBase` `Parser`:
- Input `ParserInput` (bytes + filename/type) → dispatch by extension/signature:
  - **pdf** → pypdfium2 rasterise page-by-page → images → OCR each via a **configured OCR provider**
    (`ocr: {provider: parser.lighton | parser.glm, config: {...}}` — reuse the pod-validated OCR
    models; the renderer feeds them page images). markdown-per-page → `PageMap.from_page_texts`.
  - **png/jpg/jpeg/webp/tif/tiff/bmp** → one page image → OCR → markdown.
  - **docx/pptx/xlsx** → if `soffice` present: soffice→pdf→pypdfium2→OCR; else python-docx/openpyxl
    text → markdown (per-paragraph / per-sheet page split), `renderer` recorded honestly in metadata.
  - unknown/unsupported → graceful `PARSE_ERROR` (never crash), like the other parsers.
- **Batch**: the runner already batches documents; this handles multi-page + multi-format per document.
- **Security (carry the parser lessons):** pixel-bomb cap on rasterised pages (the `parser.lighton`
  `_MAX_RENDER_MEGAPIXELS` + `Image.MAX_IMAGE_PIXELS` discipline); offset-0 type-spoof guard (#62); a
  `soffice` subprocess with a timeout + no shell; native fault → typed `ProviderError`, no bytes/text
  leaked (C2/C6). `compute="either"` (rendering CPU; OCR routes to the OCR provider's device).
- Offsets/pages exact (`PageMap.from_page_texts`, ADR-0031) — the same page-map contract as pdfplumber.

## Profile + license + conformance
- `ProviderProfile`: `compute="either"`, honest `memory_mb`, `model_id` = the configured OCR model's
  (or None for text-only office), `deterministic` truthfully (OCR decoding may be False), `batch=True`.
- **Evidence-bearing license (H-B1):** pypdfium2 Apache-2.0/BSD + Pillow HPND + python-docx MIT +
  openpyxl MIT, all cited/dated; NOTE that LibreOffice (if used) is an external MPL-2.0 tool invoked as
  a subprocess (not linked/vendored) — record it, verified. NO AGPL anywhere (assert `fitz`/PyMuPDF is
  never imported — a test greps the package).
- Passes conformance C1–C6 against a faithful offline fake (a tiny in-memory renderer + a stub OCR);
  roster +1; conformance + stack-validate green.

## Wire-in
- `latence-parser-render --no-deps` in ci.yml + verify-local heavy-install + wheel list; add
  `parser.render` to `matrix/parse.yaml`; a `stacks/` example using it as the Parse Stage feeding OCR.
- `scripts/validate-universal-parser-on-gpu.md`: turnkey pod steps (real PDF + a docx via soffice/text).

## Definition of Done
- `parser.render` builds, registers, AdapterBase `Parser`, produces DocumentRecord+PageMap from
  pdf/image/office; NO fitz/PyMuPDF (test asserts it); soffice OPTIONAL with graceful text fallback;
  pixel-bomb cap + type-spoof guard + timeouts; native fault → typed error no leak.
- Evidence-bearing permissive licenses (cited, AGPL-free); conformance C1–C6; `bash
  scripts/verify-local.sh` green; ADR-0036 note; seam-report; pod-validation script.
- Real numbers from the maintainer's pod run (real PDF + office doc); never fabricated.

## Reviewer should check
- **NO AGPL:** `fitz`/PyMuPDF is never imported or depended on (a grep test enforces it) — PDF is
  pypdfium2. Licenses cited permissive + LibreOffice noted as an external subprocess tool.
- Any-file dispatch is correct + graceful (unknown → PARSE_ERROR; soffice-absent → text fallback).
- Pixel-bomb + type-spoof + subprocess-timeout guards present (this rasterises untrusted input).
- Page map/offsets exact via PageMap.from_page_texts; OCR delegated to the configured pod-validated model.
