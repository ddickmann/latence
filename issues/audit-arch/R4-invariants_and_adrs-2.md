# R4 — The conformance suite cannot observe the schema-v18 page-map regression: one page, one chunk, and the resolved page is thrown away

- **Severity:** major
- **Dimension:** invariants and ADRs (ADR-0021/0031 provenance is true), test quality / false green
- **Branch:** feat/arch-deepening
- **Site:**
  - `packages/latence-core/src/latence_core/conformance/fixtures.py` — `_build_document`
    (`PageMap.from_page_texts([text])`, a single page) and `canonical_corpus`
    (`MarkdownChunker({})`, one chunk at the 512-token default)
  - `packages/latence-core/src/latence_core/conformance/cases.py` — `_page_offsets_resolve`
    (:277–299), consumed by the entity / relation+fused / header-refs / inline-refs cases
- **Claim contradicted:** `docs/CONFORMANCE.md` rule 3, **amended by this change set** to read
  *"not a throwaway per-chunk page index: since v18 the map rides a document's FIRST chunk only,
  so use ONE `PageIndexResolver` per call and `observe()` every chunk"* — and the suite's own
  headline, *"Passes conformance is the definition of enterprise-ready"*.
- **Invariants at stake:** ADR-0021/0031 (provenance is true — a chunk offset resolves to a real
  original-document offset/page), ADR-0034 (no silent errors)
- **Status:** FIXED (this commit)
- **Not on main:** on main the page map rode **every** chunk, so the pre-v18 pattern was correct
  and there was nothing for the suite to catch. C1 of this change set made it wrong; the machine
  gate that would catch it was not moved with it.

## What was wrong

C1 moved the page map onto a document's **first chunk only** (schema v18) and made
`PageIndexResolver` the seam that every downstream Stage must use — one resolver per call,
`observe()` every chunk. It then wrote that rule into `docs/CONFORMANCE.md`,
`docs/providers/README.md`, `docs/writing-an-adapter.md` and ADR-0022/0031/0042, removed the old
pattern from four shipped ecosystem Providers and three PII Providers, and added
`test_adr_page_map_carriage_currency.py` — a machine gate on the ADR **prose**.

The suite that gates **Providers** was left unable to see the failure, for two independent
reasons:

1. **The canonical corpus was one page and one chunk.** `_build_document` called
   `PageMap.from_page_texts([text])` — a single span covering everything — and `canonical_corpus`
   chunked the tiny document at the 512-token default, producing exactly ONE chunk: the chunk that
   *carries* the map. So no consumer of the corpus ever saw a chunk with `page_map=None`, and on a
   one-page document every possible page fallback answers `1`, which is coincidentally right.
2. **`_page_offsets_resolve` discarded the resolved page.** It fed `prov.char_start` / `char_end`
   into the document's `PageOffsetIndex` purely to assert `diagnostics.max_drift_chars == 0`, and
   never compared the resolved page numbers to the record's own `page_start` / `page_end`. A record
   that resolves perfectly but *cites the wrong page* is exactly the v18 regression, and it was
   invisible — so even a multi-page fixture would not have caught it.

Net effect: `latence conformance` reported a Provider carrying the exact pre-v18 defect as
conformance-verified, on the one invariant this change set moved.

## Failure scenario (proven by probe)

Registered `entity.prev18`: the in-core reference `GazetteerEntityExtractor` with **one** flaw —
the pattern R1 had to remove from seven shipped Providers:

```python
page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
# ... pages=None  ->  the mention inherits chunk.provenance.page_start/page_end
```

`run_conformance(registry, "entity.prev18")` on the pre-fix suite:

```
CONFORMANCE passed: True
  C3_license: pass — permissive: Apache-2.0 (verified=True)
  C5_device: pass — device honored: cpu (provider is cpu-only)
  C1_valid_records: pass — 4 contract-valid records
  C2_graceful_failure: pass — graceful (empty), no PII in surfaced text
  C4_determinism: pass — byte-identical over two runs
  C6_no_pii_leak: pass — no raw PII in non-content record fields
```

The **same Provider**, over a 24-page document chunked at `max_tokens=256` (8 chunks; the map is
carried on chunk 0 only):

```
reference (v18-correct):    62 mentions, MISPAGED = 0
prev18 (conformance-green): 62 mentions, MISPAGED = 54
```

54 of 62 mentions cite a plausible-looking WRONG source page, and the framework's advertised
contract check for first- and third-party Providers says green.

## What was built

Both halves of the blindness, and nothing else — no new layer, no new seam, no wider interface.

1. **The canonical corpus got the shape the invariant is defined over** (`fixtures.py`). The
   document is now declared as its source pages (`CANONICAL_PAGES`, three of them, assembled with
   a single-space separator) and chunked at a 16-token budget. The assembled text is
   **byte-identical** to the pre-pagination fixture — pagination moved where the page boundaries
   are, not one character of the document — and `_build_document` asserts that equality rather
   than trusting it. The result:

   | chunk | offsets | pages | carries `page_map` |
   |---|---|---|---|
   | 0 | `[0, 37)` | 1–1 | yes |
   | 1 | `[37, 79)` | 2–3 | **no** (v18) |

   Chunk 1 is the discriminating shape: it carries no map *and* spans two pages, so inheriting its
   range is a wrong range for every mention in it, wherever the mention lands. The second sentence
   deliberately straddles the page 2/3 boundary — the realistic PDF case that makes provenance hard.

2. **`_page_offsets_resolve` now asserts the record cites the page it resolved to.** The resolved
   `page_start` / `page_end` are compared against `prov.page_start` / `prov.page_end`; a
   contradiction is a FAIL. The drift assertion is kept (it is the out-of-range half, pinned by
   `test_c1_bites_an_out_of_range_mention_offset`). Records are resolved against **their own**
   document's map (keyed by `document_record_id`), so the check stays correct if the corpus ever
   grows a second document. A record that cites **no** page is not failed — `page_start` /
   `page_end` are optional on `Provenance` and "no page claim" is honest; what is failed is a page
   claim that contradicts the map.

3. `docs/CONFORMANCE.md`'s C1 row now states what C1 actually asserts for a sub-document carrier,
   so the doc and the gate say the same thing.

Deliberately NOT done:

* **A second "paged" fixture corpus alongside the canonical one.** That would have left the
  canonical corpus — the thing every other Capability's case runs on — in the shape that cannot
  observe the invariant, and added a parallel fixture path to compensate. Fixing the corpus itself
  is the smaller change and it is the one that makes the existing check real.
* **Weakening or relaxing anything to absorb the change.** The full suite (3887 tests) is green
  with the new fixture shape; no test was edited to accommodate it.

## Acceptance criteria

1. **AC1 — the corpus can express the failure.** The canonical corpus has >1 page, >1 chunk, at
   least one chunk after the first with `page_map is None`, and at least one such chunk spanning
   two pages.
2. **AC2 — pagination changed no bytes.** `documents[0].content == CANONICAL_TEXT`, and
   `CANONICAL_PAGES` assembles to `CANONICAL_TEXT` byte-for-byte (enforced in `_build_document`).
3. **AC3 — the check bites.** A Provider whose only flaw is the pre-v18 per-chunk page index FAILs
   C1, and therefore fails conformance.
4. **AC4 — it bites on the PAGE claim, not on something else.** The same records pass the bounded
   in-range half of C1 (`_offsets_in_range` is `True`) and are non-empty; only
   `_page_offsets_resolve` is `False`.
5. **AC5 — no false red.** The v18-compliant reference Provider passes on the same corpus, and
   every registered Provider still passes the sweep.

## Tests

- `packages/latence-core/tests/test_conformance.py`
  - `test_the_canonical_corpus_has_the_shape_v18_carriage_is_defined_over` (AC1, AC2) — **fails on
    the pre-fix code**: `assert 1 > 1 / where 1 = len([PageSpan(page_number=1, char_start=0,
    char_end=79)])`
  - `test_c1_bites_a_provider_that_inherits_the_chunk_page_instead_of_resolving` (AC3, AC4, AC5) —
    **fails on the pre-fix code**: `assert verdict.passed is False` → `AssertionError: assert True
    is False`, i.e. the pre-v18 Provider was reported conformance-verified
  - `test_c1_bites_an_out_of_range_mention_offset` (the drift half, unchanged, still green)
  - `test_provider_passes_conformance[*]` / `test_full_report_all_pass_and_flags_stub_scoped`
    (AC5, unchanged)
