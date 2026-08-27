# Validate `parser.render` (universal any-file Parser) on a real GPU — turnkey pod steps (W5)

The offline slice is validated against a **faithful in-memory fake** (`pypdfium2`/`Pillow`/the office
readers are not in the CPU env; a fake renderer produces deterministic page bytes and a stub OCR
round-trips them — so the REAL dispatch, security guards, OCR-delegation seam, and PageMap assembly
run, only the native rasteriser bytes + the OCR model are faked). The **real** numbers — a real PDF
rasterised by `pypdfium2` and OCR'd by a real GPU OCR model, a real `docx` rendered by LibreOffice
(`soffice`) or text-extracted, offset/PageMap round-trip, and a live conformance verdict — come from
a **pod run**. Until this runs, the real quality is explicitly **"pending pod validation"**, never
fabricated.

> **License catch (the whole reason for this slice).** `parser.render` renders PDF with `pypdfium2`
> (Apache/BSD), NEVER PyMuPDF/`fitz` (AGPL). Step 1 installs it with only permissive deps; a test
> (`test_no_fitz_or_pymupdf_imported_anywhere`) already enforces no `fitz` import. Do not `pip install
> pymupdf` on the pod for this Provider.

The **rendering** is CPU (pypdfium2/Pillow/soffice all run on the pod CPU); only the **page OCR** needs
the GPU (it is delegated to `parser.lighton`/`parser.glm`). So a ≥16 GB CUDA GPU is needed only for the
OCR sub-provider. LibreOffice is OPTIONAL — with it, office files get a faithful page render; without
it, office files degrade to MIT text extraction (no OCR, no crash).

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible (for the OCR sub-provider)
python --version               # 3.11+
which soffice || echo "no soffice — office files will use the MIT text-extraction fallback"
# To get the faithful office render path, install LibreOffice (Debian/Ubuntu):
#   sudo apt-get update && sudo apt-get install -y libreoffice
```

## 1. Install the framework + this provider WITH its (permissive) render deps + an OCR provider

```bash
# from a clone of the repo on the pod:
cd latence
pip install -e packages/latence-core
pip install -e packages/latence-parser-render    # pulls pypdfium2 + pillow + python-docx + openpyxl + python-pptx
pip install -e packages/latence-parser-lighton   # the OCR sub-provider (transformers + torch); or latence-parser-glm

# NO PyMuPDF/fitz — PDF is pypdfium2 (Apache/BSD). Confirm the render deps import:
python -c "import pypdfium2, PIL, docx, openpyxl, pptx; print('render deps ok')"
python -c "import fitz" 2>&1 | grep -q "No module" && echo "good: no fitz (AGPL) installed"
```

## 2. Sanity: the provider constructs, resolves its OCR sub-provider, is not device-skipped

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"          # or leave unset — a real probe finds the GPU
from latence_parser_render.provider import UniversalRenderParser
p = UniversalRenderParser({"ocr": {"provider": "parser.lighton", "config": {"device": "cuda"}}})
# parser.render is compute="either": it is NOT device-skipped (the OCR sub-provider carries the GPU).
print("render device:", p.device.device, "skipped:", p.device.skipped)   # expect: cpu False (auto)
PY
```

## 3. Real PDF → pypdfium2 rasterise → real OCR → markdown + offset round-trip

Point `RENDER_REAL_PDF` at a real (ideally scanned/image-only) PDF with legible text, then:

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"
from latence_core.contracts import Classification, ParserInput, Provenance
from latence_core.contracts import Disposition
from latence_core.pagemap import PageOffsetIndex
from latence_parser_render.provider import UniversalRenderParser

path = os.environ["RENDER_REAL_PDF"]
data = open(path, "rb").read()
inp = ParserInput(
    provenance=Provenance(
        source_uri=f"file://{path}", file_name="real.pdf", file_type="pdf",
        file_size=len(data), document_id="sha256:real-pdf",
    ),
    classification=Classification(language="und", sensitivity="internal"),
    content=data,
)
p = UniversalRenderParser({"dpi": 200, "ocr": {"provider": "parser.lighton", "config": {"device": "cuda"}}})
(rec,) = list(p.parse([inp]))
assert rec.disposition is Disposition.PARSED, rec.error
assert rec.page_map is not None
idx = PageOffsetIndex(rec.page_map)
for span in rec.page_map.pages:
    assert idx.page_for_offset(span.char_start) == span.page_number
print("pages:", len(rec.page_map.pages), "chars:", len(rec.content), "drift_recovered:", idx.diagnostics.recovered)
print("--- first 400 chars of OCR'd markdown (eyeball vs the page) ---")
print(rec.content[:400])
PY
```

Record: page count, extracted char count, zero (or recovered) drift, and a **sample of the markdown**
eyeballed against the real page (the false-green catch — the text must be legible page content, not
prompt/chat scaffolding from the OCR model).

## 4. Real `docx` → soffice render (or MIT text fallback) → markdown

Point `RENDER_REAL_DOCX` at a real `.docx`:

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"
from latence_core.contracts import Classification, ParserInput, Provenance, Disposition
from latence_parser_render.provider import UniversalRenderParser

path = os.environ["RENDER_REAL_DOCX"]
data = open(path, "rb").read()
inp = ParserInput(
    provenance=Provenance(
        source_uri=f"file://{path}", file_name="real.docx", file_type="docx",
        file_size=len(data), document_id="sha256:real-docx",
    ),
    classification=Classification(language="und", sensitivity="internal"),
    content=data,
)
# With soffice present: soffice->pdf->pypdfium2->OCR (a faithful render). Without: python-docx text.
p = UniversalRenderParser({"ocr": {"provider": "parser.lighton", "config": {"device": "cuda"}}})
(rec,) = list(p.parse([inp]))
assert rec.disposition is Disposition.PARSED, rec.error
print("docx pages:", len(rec.page_map.pages) if rec.page_map else 0, "chars:", len(rec.content))
print("--- first 400 chars ---")
print(rec.content[:400])
PY
```

Run it **twice** — once with `soffice` on PATH (faithful page render → OCR) and once with it removed
from PATH (MIT text extraction fallback). Both must produce a PARSED record with legible content;
record which path each used and a text sample.

## 5. Full stack over a mixed corpus + conformance

```bash
export LATENCE_CUDA=1
# Put a few PDFs, images, and office files in a folder, point CORPUS_DIR/STORAGE_URI at it:
CORPUS_DIR="file:///path/to/mixed-corpus" STORAGE_URI="file:///tmp/render-store" \
  latence stack validate stacks/universal-parse.yaml       # expect PASS, Parse runs (cuda) on the rig

# The live conformance verdict (runs C1–C6 for real on the pod, not device-skipped):
python -c "from latence_core.conformance import run_conformance; from latence_core.capability import ProviderRegistry; \
  v = run_conformance(ProviderRegistry(), 'parser.render'); \
  print('passed:', v.passed, 'device_skipped:', v.device_skipped, [(c.check, c.status.value) for c in v.checks])"
```

## What to paste back (the real numbers, never fabricated)

- PDF: page count, extracted char count, drift, a markdown sample eyeballed vs the page.
- docx: the render path used (soffice vs text), page/char counts, a text sample — for BOTH runs.
- `stacks/universal-parse.yaml` validation: PASS + the per-check table (contracts/report/rag).
- The live `parser.render` conformance verdict (passed + the C1–C6 statuses).
- GPU + soffice versions (`nvidia-smi`, `soffice --version`) for reproducibility.
