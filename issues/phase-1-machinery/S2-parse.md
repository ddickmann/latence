# S2 — Parse

**Parent**: S1 · **Phase 1 · Baseline bar**

## What to build
A real **Parse** Capability: document (PDF/image/office) → markdown + a **page map** preserving original page boundaries, plus first-class **Classification** attached to the parsed document (language, content category, sensitivity) and complete **Provenance** (file name, type, size-after-parse, source, document id, page/char offsets). Ships one CPU-viable reference Parser Provider. OCR is a Provider technique, never the Capability name.

## Ports (algorithmic assets)
- **PageOffsetIndex** (`dataset-intelligence/page_mapper.py`, ~500 LOC, 5/5): cumulative page boundaries + `bisect` O(log n) offset→page, fuzzy drift-recovery fallback, drift diagnostics, never-crash degradation. PORT-AS-IS.

## Acceptance criteria
- [ ] Parser Capability protocol defined against the contracts; ≥1 CPU reference Provider (permissive license).
- [ ] Output carries a page map; any downstream offset resolves to the correct source page via the ported PageOffsetIndex, with drift diagnostics in the Quality Report.
- [ ] `Classification` (language auto-detected, content category, sensitivity default) attached at Parse and present on the record contract.
- [ ] `Provenance` includes size-after-parse and page/char offsets; validated at the Stage boundary.
- [ ] Handles a corrupt/unparseable file gracefully (records an error disposition, does not crash the run).
- [ ] Baseline bar met.

## Blocked by
S1.
