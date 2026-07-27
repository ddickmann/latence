# R2 / invariants-and-ADRs #4 — C1 invalidated ADR-0022's recorded decision and amended NO ADR

**Status: FIXED.** ADR-0022 (and the two supporting sentences in ADR-0031/ADR-0042 that inherited
its claim) now record the schema-v18 carriage, and a machine-checked gate keeps the ADR set from
drifting away from what the Chunk Stage actually does.

## Reported defect (severity: major)

- **Dimension:** invariants and ADRs (ADR-0017 files are the source of truth; ADR-0021/0031
  PROVENANCE IS TRUE)
- **Branch:** `feat/arch-deepening`
- **Site:** `docs/adr/0022-entity-mention-record-and-zero-shot-provider-split.md:56` (unchanged in
  the diff), against `packages/latence-core/src/latence_core/stages/chunk.py:220`
  (`page_map=doc.page_map if i == 0 else None`).

ADR-0022 states, as part of its **decision**:

> To make this resolvable at the mention Stage, Chunk now carries the parent document's `page_map`
> forward on each `ChunkRecord` (optional field; absent ⇒ the mention degrades to the inherited
> chunk range).

C1 (schema v18) inverted the mechanism: the map is stamped on a document's **first** chunk only and
offset→page resolution moved behind the document-scoped `latence_core.page_index.PageIndexResolver`
seam. C1 recorded that only in `CHANGELOG.md`. Three other lanes in this same diff DID amend their
ADRs (0003, 0033, 0036 all carry new amendment blocks); `git diff main...feat/arch-deepening
--name-status -- docs/` lists no change to 0022. The authoritative design record therefore still
asserts the superseded mechanism — and, worse, it spells out the exact wrong degradation contract
(`absent ⇒ inherit the chunk range`) as the intended one.

## Failure scenario

A Provider author (first- or third-party) follows the ADR set — the repo's stated source of design
truth — reads ADR-0022's decision, and writes `PageOffsetIndex(chunk.page_map)` per chunk with the
documented `None ⇒ inherit the chunk range` degradation. Under v18 `chunk.page_map is None` on
every chunk after a document's first, so every mention/relation/PII span on those chunks silently
takes the parent chunk's inherited page range instead of its own page — a plausible-looking wrong
page in the exported KG, and a direct violation of ADR-0021/0031.

This is not hypothetical. R1 found precisely that code in four shipped ecosystem Providers
(`extract-gliner2`, `ner-gliner`, `relation-gliner`, `ner-endpoint`) and in the PII redactors; those
call sites were fixed by `0bd1165` (see `issues/audit-arch/R1-depth_actually_gained-0.md`), and
`docs/writing-an-adapter.md` was rewritten in this same audit — but the ADR that told every author to
write it that way was never corrected, so the next author reproduces the bug from the record.

## What to build

1. **Amend ADR-0022.** Add a dated `C1 amendment` block in the convention the other three lanes used
   (a blockquote update at the top of the affected section), and correct the decision sentence
   itself so the ADR states the v18 carriage (map on the first chunk; resolution through
   `PageIndexResolver`; `pages_for_span(...) -> None` is the only degradation contract). Keep the
   superseded sentence visible as history — ADR-0022 already has a `Correction (…superseded by
   ADR-0031)` block, so the file's own convention is correct-in-place, not rewrite-history.
2. **Correct the two sentences that inherited the claim.** ADR-0031 cites ADR-0022's per-chunk
   carriage as precedent and calls the map "already carried per chunk"; ADR-0042 describes PII-span
   page resolution as going "through the chunk's `offset_map` + `page_map`", which is the v17 route.
3. **Add the machine-checked gate** so this class of ADR drift is caught by `make test` rather than
   by an audit round.

Do **not** revert C1. The seam is a genuine deepening (one `PageIndexResolver` replaced five
hand-rolled per-chunk indices, and shed ~92% of a chunk checkpoint line); the defect is that the
design record was not updated with it.

## Acceptance criteria

- [x] No ADR asserts, in normative present-tense voice, that the page map rides each/every
      `ChunkRecord`.
- [x] Every ADR that makes a page-map carriage claim names the v18 mechanism (first chunk /
      `PageIndexResolver`).
- [x] The gate derives the forbidden claim from the CODE (it runs `MarkdownChunker` and observes
      which chunk indices carry the map) — if carriage ever legitimately changes back, the gate
      changes with it instead of pinning prose to a constant.
- [x] A regression test that FAILS on the pre-fix ADR text and passes after.
- [x] `make test` green; no test weakened or deleted.

## Regression test

`packages/latence-core/tests/test_adr_page_map_carriage_currency.py`

- `test_the_chunk_stage_stamps_the_page_map_on_exactly_one_chunk_per_document` — the code fact the
  docs gate is derived from.
- `test_no_adr_asserts_the_superseded_per_chunk_page_map_carriage` — fails on ADR-0022:56 and
  ADR-0031:71 pre-fix.
- `test_every_adr_that_claims_page_map_carriage_names_the_v18_mechanism` — fails on ADR-0022
  pre-fix.
