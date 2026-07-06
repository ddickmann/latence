# latence-parser-document

CPU reference **Parser** Provider for the Latence framework: turns a source
document (PDF, plain text, markdown) into a markdown `DocumentRecord` with a real
**page map**, a first-class **Classification** attached at Parse, and complete
**Provenance** — the S2 Parse slice.

## What it does

- **PDF → per-page markdown + page map.** Extracts text page by page (pure-Python
  [`pypdf`](https://pypi.org/project/pypdf/), CPU-only, BSD-3-Clause), assembles
  the pages into one markdown string, and records each page's `[char_start,
  char_end)` span. Any downstream character offset (a chunk boundary, an entity
  mention) resolves back to its original **source page** via
  `latence_core.pagemap.PageOffsetIndex`.
- **Text / markdown → single-page record.** A `.txt`/`.md` document is one page.
- **Classification at Parse.** Language is auto-detected from the *decoded* text
  (the shared `latence_core.language` heuristic — reused, not forked); category
  and sensitivity come from config, else inherit the Source's stamp.
- **Graceful failure.** A corrupt, encrypted, or type-spoofed file does **not**
  crash the run: the Parser emits a `PARSE_ERROR` record carrying its Provenance
  and the failure reason, auditable in the Quality Report.

OCR is a Provider *technique*, never the Capability name. This reference does not
OCR scanned images (a page with no text layer yields empty text, recorded not
crashed); an image/OCR Provider is a drop-in second Parser behind the same seam,
kept out of the lean default install (ADR-0016).

## Use it

Install the package and reference it by its entry-point name in a Pipeline:

```yaml
stages:
  - name: parse
    capability: parse
    provider: parser.document      # registered via the 'latence.providers' group
    depends_on: [source]
    config:
      encoding: utf-8              # text decode default; per-input hint overrides
      category: contract           # optional content category
      page_separator: "\n\n"      # markdown joined between PDF pages
```

## Licensing

- Code: Apache-2.0 (this package).
- `pypdf` dependency: BSD-3-Clause (permissive; commercial use, redistribution,
  and modification allowed). Verified 2026-07-06 per ADR-0012.
