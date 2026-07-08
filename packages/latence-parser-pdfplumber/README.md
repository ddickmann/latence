# latence-parser-pdfplumber

The **first real Wave-1 adapter built through `AdapterBase`** (H-A1) — a CPU, page-aware PDF (and
plain-text) Parser Provider (`parser.pdfplumber`) over the
[pdfplumber](https://github.com/jsvine/pdfplumber) library (MIT), implemented as an
[`AdapterBase`](../latence-core/src/latence_core/providers/adapter.py) subclass.

Before this slice, the Phase-3 provider seam (`AdapterBase` P3-F2 + `ProviderProfile` P3-F1) was
validated only by tests that fabricate fake subclasses — it had **zero real production consumers**.
This package is the first real one, turning the hypothetical seam into a real two-adapter seam and
serving as the **template** every subsequent Wave-1 adapter (docling, GLiNER variants, embedders,
PII) copies.

## Install

```bash
pip install latence-parser-pdfplumber
```

Registers `parser.pdfplumber` under the `latence.providers` entry-point group (ADR-0004) — any
Pipeline can name it with no core change, alongside `parser.plaintext` and the pypdf
`parser.document`. CPU-only, **no OS-level system binaries** (no pandoc/tesseract to apt-install);
installs from wheels. NOTE — *not* pure-Python: the closure pulls native code via wheels (`pillow`
C-ext, `pypdfium2` bundled PDFium, `cryptography` under pdfminer.six), all permissive and latent
(the used path is pdfminer.six-only). See the `profile()` license note.

## Usage

```yaml
stages:
  - name: parse
    capability: parse
    provider: parser.pdfplumber
    depends_on: [source]
    config:
      page_separator: "\n\n"   # markdown joined between PDF pages
      # encoding: utf-8        # for text inputs; a per-input hint overrides
      # batch_size: 32         # documents grouped per batch (memory-shaping only)
```

It reads the PDF **text layer** only. A scanned page with no text layer yields empty page text
(recorded, not crashed); OCR is a separate Provider *technique* (CONTEXT glossary), a drop-in
second Parser behind the same seam.

---

## Seam report — did `AdapterBase` fit a real adapter? (the audit's actual question)

**Verdict: yes, with one small, justified widening.** `AdapterBase` fit a real Parser adapter well.
Every facility the issue named was used *for real*, not bypassed:

| Facility | Used | How |
|---|---|---|
| `ProviderProfile` (P3-F1) | ✅ as-is | Declared `compute="cpu"`, `model_id=None`, evidence-bearing MIT license, `deterministic=True`, `batch=True`. |
| `super().__init__` device resolve | ✅ as-is | Device resolved against the profile; trivially CPU here, but flows through the seam (`self.device`). |
| `guard(op)` — native → typed | ✅ as-is | pdfminer's native `PdfminerException` on a corrupt/encrypted PDF → typed `ProviderError`, carrying only provider + op + native *type name* (no content/PII). |
| `batched(items, run_batch)` | ✅ as-is | `parse()` runs the batching path over multiple input documents, input-order-preserving, one batch at a time (streaming-friendly, ADR-0033). |
| `carry_provenance(...)` | ✅ after a 1-line widening (below) | Source lineage carried, only char/page span narrowed — no hand-rolled mapper (the audit found the in-core GLiNER provider hand-rolled; this does not repeat it). |
| `PageMap.from_page_texts` (core seam) | ✅ as-is | Per-page pdfplumber text assembled into exact-by-construction page boundaries; offsets round-trip, pages correct. |

### The one AdapterBase change (a real finding, fixed not documented-around)

`carry_provenance` was typed `source: Record`. But a Parser's source is a **`ParserInput`** (a
*raw document*, ADR-0019), which carries `Provenance` + `Classification` yet is **not** a `Record`
subclass. So the first real Parser over the base could not reuse the helper for a whole-document
carry without a hand-rolled cast — the exact "weaker mapper" the audit forbids. The minimal,
backward-compatible fix: widen the source parameter to a new structural Protocol
`ProvenanceCarrier` (just `.provenance` + `.classification`, the only two attributes the helper
reads). Every `Record` satisfies it, and so does a `ParserInput` — so sub-document carriers
(mentions/chunks) and the whole-document Parser now carry through the *same* base helper. This is
the seam validation working as intended: a real consumer exposed a misfit, and the seam was
sharpened rather than bypassed.

### Inert `ProviderProfile` fields (the trim-vs-keep signal H-A1 asked for)

For this adapter, two profile fields are **inert** (declared truthfully, but nothing reads them):

- **`cost_per_1k`** — `None`; correct for a local library (no per-call cost). Inert here; it earns
  its keep only for endpoint/API providers (`parse.endpoint`, the embedder endpoint).
- **`memory_mb`** — declared `96` (an honest small footprint for pdfplumber's page buffers), but
  the device seam routes only on `compute`; there is no memory-gated skip, so nothing consumes it.
  Inert here; it would earn its keep once a memory-aware host-fit check exists.

Recommendation: **keep both** — they are honest, cheap, and load-bearing for the endpoint/GPU
providers the bake-off (P3-F5) ranks; trimming them would re-fork the descriptor per provider
class. `compute`, `license*`, `deterministic`, and `batch` were all live and load-bearing here.

## Library choice — why pdfplumber, not the preferred kreuzberg

H-A1 preferred [kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) but made it conditional on an
**early install-and-run verification** that it is pip-installable without system binaries and
permissively licensed. That verification (2026-07-08) found kreuzberg unsuitable on **both** counts:

- **License (blocking, ADR-0012):** kreuzberg's current major (**4.9.9**) is **`Elastic-2.0`** — a
  source-available, **non-permissive** license (PyPI classifier `License :: Other/Proprietary
  License`). ADR-0012 requires permissive-or-opt-in; Elastic-2.0 is neither.
- **Import (blocking):** kreuzberg's last permissive line (**3.x, MIT**) **fails to import** without
  OCR system dependencies (`from kreuzberg._ocr import get_ocr_backend` at import time) — the
  "requires system packages" failure mode the issue names.

So, exactly as the issue's fallback clause directs ("substitute a pure-Python permissive PDF lib —
`pdfplumber`, MIT — and name the package accordingly"), this uses **pdfplumber** (MIT, CPU,
page-aware via `page.extract_text()` per `page.page_number`, no OS-level system binaries). Its
direct extraction core `pdfminer.six` is likewise MIT. Page structure is preserved (the PageMap/
offset path is genuinely exercised), and text-layer extraction is deterministic (verified).
Accuracy note: pdfplumber is *not* strictly pure-Python — its wheel closure includes native
`pillow` (C-ext) + `pypdfium2` (bundled PDFium), and `pdfminer.six` pulls compiled `cryptography`;
all permissive, and latent (the used code path stays in pdfminer.six). Full closure in the license
note below.

**License verification (evidence-bearing, H-B1 / ADR-0012):**
- Library: pdfplumber, installed version 0.11.x (+ `pdfminer.six` core); wheel closure also pulls
  native `pillow` (HPND/MIT-CMU), `pypdfium2` (Apache-2.0/BSD-3), `cryptography` (Apache-2.0/BSD-3) —
  all permissive (no ADR-0012 violation), latent on the used pdfminer.six-only path.
- SPDX: **MIT** (code) — verified from the installed distribution's PyPI metadata (License
  classifier `OSI Approved :: MIT License`) and the upstream GitHub `LICENSE.txt`
  (`https://github.com/jsvine/pdfplumber/blob/stable/LICENSE.txt`), checked 2026-07-08.
- Weights: none (`model_id=None`, `license_weights=None`) — a pure-code provider, no model.

## Conformance

`parser.pdfplumber` passes the P3-F3 conformance suite (C1–C6) as a registered provider — the
first real adapter to pass the gate end to end. Its name maps to the `parse` Capability, so it is
auto-covered by the Parse `ConformanceCase` (no override needed); the full suite stays green with
it in the roster.
