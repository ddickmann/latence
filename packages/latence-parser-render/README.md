# latence-parser-render — `parser.render`

The **universal any-file Parser** for the Latence framework (W5-universal-parser): one Parser
Provider that turns **ANY** common file type — PDF, image, or office document (docx/pptx/xlsx) — into
a `DocumentRecord` + `PageMap`, dispatching by type. It is a **license-clean port of the colsearch
batch converter** ("any file → batches of images"): the "files to markdown" front-end an OCR
pipeline needs so a real corpus (scanned PDFs, photos, office files) all flows through one Parse
Stage.

## What it does

| Input type | How it is rendered | OCR? |
|---|---|---|
| `pdf` | `pypdfium2` rasterises page-by-page → page images | yes (delegated) |
| `png/jpg/jpeg/webp/tif/tiff/bmp` | `Pillow` decodes → one page image | yes (delegated) |
| `docx/pptx/xlsx` **with `soffice`** | LibreOffice-headless `soffice --convert-to pdf` → `pypdfium2` | yes (delegated) |
| `docx/pptx/xlsx` **without `soffice`** | `python-docx`/`openpyxl`/`python-pptx` text extraction | no (text layer) |
| unknown / spoofed / corrupt | — | graceful `PARSE_ERROR` |

Page images are OCR'd by a **configured OCR Provider** — `parser.render` is the *renderer*, and the
actual pixels→markdown OCR is delegated (reusing the pod-validated OCR models, never reinventing
them):

```yaml
- name: parse
  capability: parse
  provider: parser.render
  config:
    dpi: 200
    ocr:
      provider: parser.lighton      # or parser.glm — the pod-validated GPU OCR models
      config: { device: cuda }
```

Each rasterised page is handed to the OCR Provider as a one-page `png` `ParserInput` **through its
`Parser.parse` seam** — no private method reached into, no seam forked. The office **text-fallback**
path needs no OCR. Per-page markdown is assembled onto `PageMap.from_page_texts` (ADR-0031) so
offsets/pages are exact by construction.

## License discipline (the whole reason for a clean port — ADR-0012)

colsearch renders PDF with **PyMuPDF (`fitz`) — AGPL-3.0** (copyleft). This package **NEVER** imports
or depends on `fitz`/PyMuPDF — a test greps the installed package to enforce it. Every dependency is
**permissive** and evidence-verified (these are pure-code libraries; weights-vs-code does not apply):

| Dependency | License | Evidence (verified 2026-07-09) |
|---|---|---|
| `pypdfium2` (PDF→image) | **Apache-2.0 / BSD-3-Clause** (wraps Google PDFium, BSD-3) | <https://github.com/pypdfium2-team/pypdfium2> |
| `Pillow` (image decode) | **HPND** (permissive, BSD-like) | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| `python-docx` (docx text) | **MIT** | <https://github.com/python-openxml/python-docx/blob/master/LICENSE> |
| `openpyxl` (xlsx text) | **MIT** | <https://foss.heptapod.net/openpyxl/openpyxl> |
| `python-pptx` (pptx text) | **MIT** | <https://github.com/scanny/python-pptx/blob/master/LICENSE> |

**LibreOffice** (`soffice`) is an **OPTIONAL, EXTERNAL** tool (**MPL-2.0**) invoked as a **subprocess**
— it is *not* linked, imported, or vendored, so its copyleft never reaches this package. A host
without it degrades gracefully to the MIT text-extraction path (no crash). The Provider ships **no
model weights of its own** (`model_id=None`): OCR weights belong to the delegated OCR Provider, whose
own profile records its weights license.

**No AGPL anywhere.** `parser.render` uses `pypdfium2` for PDF — the same permissive renderer the
`parser.lighton` Provider already uses.

## Security (this rasterises untrusted input)

- **Pixel-bomb DoS cap** — PDF render scale clamped by a max-megapixel budget (a giant page can never
  rasterise to a gigapixel bitmap); `Image.MAX_IMAGE_PIXELS` tightened so a decompression-bomb image
  raises rather than allocating multi-GB (restored afterwards, no global side effect).
- **Offset-0 type-spoof guard (#62)** — a `.pdf`/declared-image/declared-office file whose bytes are
  not that type is rejected before the renderer.
- **soffice subprocess** — explicit argument list (no shell), a hard timeout, an isolated temp
  profile dir; a timeout/non-zero exit degrades to the text path, never a hang.
- **native fault → typed error, no leak** — a native fault maps through `AdapterBase.guard` to a
  typed `ProviderError` carrying only provider + op + native type name, never input bytes / extracted
  text / OCR content (C2/C6).

## Packaging

`compute="either"` (rendering is CPU; OCR routes to the OCR Provider's device). Heavy deps ship in
this package, not the thin core (ADR-0016), and are lazy-imported so Provider discovery stays light.
Like the other heavy Provider packages, it installs `--no-deps` into the CPU-first dev/CI env; its
tests use a faithful in-memory fake renderer + a stub OCR, and the real soffice/PDF/office run is
validated on the pod (`scripts/validate-universal-parser-on-gpu.md`).
