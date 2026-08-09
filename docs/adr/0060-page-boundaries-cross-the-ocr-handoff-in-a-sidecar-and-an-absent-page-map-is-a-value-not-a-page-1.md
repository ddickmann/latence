# Page boundaries cross the OCR handoff in a sidecar; an absent page map is a value, not a page 1

Pre-paginated markdown carries its page boundaries into the pipeline in a **sidecar file
discovered next to the document** (`<file>.pagemap.json`, holding a serialized `PageMap`), found by
the Source and interpreted by the Parser. Where no such information exists, the resulting map is
labelled **`PageMapOrigin.ASSUMED_SINGLE_PAGE`** — a distinct, inspectable value that travels on the
`PageMap`, on every chunk's `PageSlice`, and into the export — and a Parser configured with
`require_page_map` refuses the document outright with a `PARSE_ERROR`. What must never happen again
is the third option: a confident `page 1`.

## Context

The framework advertises page-accurate citation as a first-class capability. `PageMap` is a core
contract (ADR-0020), `PageOffsetIndex` resolves any downstream offset back to its page, `PageSlice`
makes every chunk self-describing (ADR-0034), and `PageSliceMissingError` exists precisely so a
chunk that cannot answer *raises* instead of guessing. The docstring states the promise outright:
a knowledge-graph edge should cite "page 7 of the contract", not "character 41,213 of an opaque
blob".

The promise dies at one seam, and it is the seam every real enterprise corpus goes through: **OCR
happens out of band and its markdown is re-ingested.** The campaign's own S2 stage OCRs a PDF page
by page, writes `pages/<dataset>/<doc>/0001.md …`, joins them into one `markdown/<doc>.md`, and
hands that to a pipeline wired `source.local_folder → parser.plaintext`. A plaintext parser has no
page structure to report, so it emitted one span covering everything.

Measured on the S3 SOTA corpus — ohr_bench / vidoseek / uda, 3,470 OCR'd documents, 236,000
exported chunks — **every single chunk carries `provenance.page_start == page_end == 1`**, and
`page_slice` reports a 7-page paper as one page:
`{"pages":[{"char_start":0,"char_end":31711,"page_number":1}]}`.

Three properties of that failure are what make it worth an ADR rather than a patch:

1. **It is silent.** `page_start` is populated. Nothing is null, nothing raises, no counter moves.
   The output is shaped exactly like a correct one.
2. **It is plausible.** `1` is a legal page number. A reviewer spot-checking a citation for a
   one-page document sees a correct answer.
3. **It defeated the contract that was built to catch it.** `PageSliceMissingError` fires when a
   chunk carries *no* slice. Here every chunk carries a perfectly valid slice — of a map that was
   fabricated one Stage earlier.

The cost surfaced downstream: the M6 benchmark harness could not use the exported page provenance
at all and had to reconstruct page spans by re-running the S2 assembly byte-for-byte
(`benchmark/s5/pagemap.py`) to obtain usable page-level gold. A benchmark reimplementing a shipped
framework capability is the symptom; the workaround must not become the answer.

## The design question

How does markdown that was paginated *somewhere else* carry its page boundaries into the pipeline?

### (a) A sidecar page-map file discovered next to the markdown — CHOSEN

`report.md` is accompanied by `report.md.pagemap.json`, containing exactly a serialized `PageMap`
(`pages` + `total_chars`) in the character coordinates of the document's decoded text. The Source —
the only side of the Parse seam holding a `Storage` handle — finds it and attaches the raw bytes to
`ParserInput.page_map_sidecar`; the Parser validates them into a real `PageMap`.

- **The markdown is untouched, byte for byte.** This is not a nicety: the S3 corpus, its 236k
  chunk offsets, and `benchmark/s5/pagemap.py`'s byte-equality assertion are all built on those
  exact bytes. Any design that writes into the document invalidates all of it.
- **The document text and the page boundaries stay separable**, which is the property the whole
  offset system depends on: offsets index the text, and nothing that is not the text may occupy an
  offset.
- **The producer already has the boundaries.** The OCR worker is holding the per-page strings at
  the moment it joins them; the boundaries were never unknown, only discarded by the join.
- **Absence is expressible.** No file means no page information — a first-class, detectable state,
  which is the half of this problem that the transport mechanism alone does not solve (below).

Costs, accepted: two files must travel together (mitigated — both the S2 pull and the S3 push move
whole directories, so they already do), and a stale sidecar is possible (mitigated — see the
cross-check below, which turns staleness into a `PARSE_ERROR`).

### (b) An in-band page-break convention the parser understands

Rejected, on three independent grounds, the first of which is fatal on its own.

- **It cannot be byte-identical.** Injecting `<!-- latence:page-break -->` (or any marker) changes
  the assembled markdown. The S3 corpus was built from the current bytes and `benchmark/s5` asserts
  reproduction of them; the existing corpus would have to be rebuilt to gain page provenance, and
  every recorded chunk offset would shift. The one design constraint that is genuinely
  non-negotiable rules this out.
- **It puts non-content at content offsets.** Either the markers stay in the text (polluting
  chunks, embeddings and extraction with synthetic tokens) or the parser strips them and every
  offset shifts — requiring exactly the strip-offset-map machinery ADR-0031 already had to build
  once for markup, now compounded.
- **It is unenforceable.** A markdown document may legitimately contain any comment; there is no
  marker whose absence proves the document is unpaginated, and no way to tell a genuine marker
  from one that survived a copy-paste.

### (c) The Source stage attaching page metadata

Rejected as the *primary* mechanism, and partially adopted as the transport.

Having the Source produce the `PageMap` itself would mean every Source Provider — SharePoint, S3,
future connectors — grows page semantics for a format it does not decode, and would invert
ADR-0019: the Source hands over *undecoded* bytes precisely because character offsets are a
property of the decode, which only the Parser performs. A `total_chars` computed by a Source that
guessed an encoding is a wrong page map with extra steps.

What (c) gets right is that the Source is the side with the `Storage` handle. So the split is:
**the Source discovers and carries; the Parser interprets and validates.** `page_map_sidecar` rides
`ParserInput` as raw bytes, exactly like `content`, and the Parser owns the turn into a contract.
This also keeps the Parser a pure function of its input record — testable with no filesystem.

### Why a sidecar file is not the "out-of-band page-map file" ADR-0020 rejected

ADR-0020 rejected "a separate out-of-band page-map file keyed by document id", because it would
break the invariant that every *Stage boundary* is one validated versioned record, which
checkpoint/resume depends on. That reasoning is untouched and still binding: nothing here crosses a
Stage boundary out of band. The sidecar is a **source artifact at the input edge**, alongside the
document itself, and it becomes part of the single validated `ParserInput` record before the first
Stage boundary — the same status the document's own bytes have.

## Decision

**1. `PAGE_MAP_SIDECAR_SUFFIX = ".pagemap.json"`**, appended to the document's *full* file name
(`report.md` → `report.md.pagemap.json`, so `report.md` and `report.txt` cannot contend for one
sidecar). Contents: a serialized `PageMap`. The convention is named once, in `latence_core.contracts`,
so producer and consumer cannot drift.

**2. `ParserInput.page_map_sidecar: bytes | None`.** `LocalFolderSource` skips any file whose name
ends with the suffix (a sidecar is not a document — checked *before* the extension filter, so
widening `extensions` to include `json` cannot turn page boundaries into a contentless "document"),
and pairs each document with its sidecar's raw bytes.

**3. `PageMap.from_sidecar_json(data, *, text_length)`** validates the sidecar and **cross-checks
`total_chars` against the length of the decoded document text**. That check is what makes the
sidecar trustworthy rather than merely present: a sidecar that has drifted from its markdown still
produces contiguous, ascending, plausibly-sized spans, and would resolve real offsets to
confidently wrong pages. A mismatch — like malformed JSON, non-contiguous spans, or a
schema version from the future — raises `PageMapSidecarError`, which the Parser turns into a
`PARSE_ERROR` record (ADR-0020): counted in the Quality Report, reason attached, run continues. The
sidecar's own `origin`, if it carries one, is ignored: a file cannot vouch for its own provenance.

**4. `PageMapOrigin` on `PageMap` and `PageSlice`** — `PARSER` (the Parser segmented the source),
`SIDECAR` (supplied out of band and cross-checked), `ASSUMED_SINGLE_PAGE` (there was no page
structure; the single span is a convention so offset→page resolution keeps one code path, *not* a
claim that the document has one page). `PageSlice.for_span` carries the parent map's origin, so a
chunk stays self-describing about how much its page number is worth, all the way into the export.

**5. `parser.plaintext` gains `require_page_map` (default `False`).** Off: a document with no
sidecar parses exactly as before, one span, page 1 — but the map now says `ASSUMED_SINGLE_PAGE`.
On: no sidecar is a `PARSE_ERROR` per document.

**6. The S2 OCR path emits the sidecar.** `ocr_lib.assemble_document_with_pages` returns the
markdown *and* its page spans; `assemble_document` is now literally that function's `.text`, so
there is no second implementation of the join to drift from and the artifact is byte-identical by
construction. The worker writes the sidecar beside the markdown, page-map first (the markdown's
existence is the stage's done-probe, so anything that looks assembled has its map), and
**back-fills** the sidecar for documents a pre-ADR-0060 pass already assembled — but only when
re-running the assembly reproduces the markdown byte for byte, so a document whose page artifacts
no longer match stays honestly unpaginated rather than becoming confidently mis-paginated. The S3
pipeline sets `require_page_map: true` for the OCR'd datasets and leaves the text-native ones (which
genuinely have no pages) on the honest default.

### What happens when page information is ABSENT

This is the part the transport mechanism does not solve, and the part the original defect actually
turned on. **Nothing in an assembled markdown blob distinguishes a one-page note from a 40-page
report whose page breaks were dropped upstream.** No parser can infer it. So the framework stops
pretending it can, in two layers:

- **Always: the assumption is recorded, not hidden.** `ASSUMED_SINGLE_PAGE` propagates map → slice →
  export. The historical defect becomes queryable after the fact — "236k chunks report page 1 with
  origin `assumed_single_page`" is an answerable question, where "236k chunks report page 1" was
  not. Page resolution itself is unchanged: the same offsets resolve to the same page numbers as
  before.
- **On declaration: it is refused.** Only the operator knows whether a corpus is supposed to be
  paginated. `require_page_map` is how they say so, and a document that then arrives without page
  information is a `PARSE_ERROR` with the reason attached — the same visible-degradation posture
  ADR-0020 gave an undecodable file and ADR-0034 gave a chunk with no `PageSlice`.

Both are additive and default-off in behaviour: an existing pipeline with no sidecars resolves every
offset to exactly the page it resolved to before.

## Consequences

- `SCHEMA_VERSION` → **21**. Every addition is defaulted; a v20 record deserialises unchanged.
  Records now serialize one extra field on `PageMap` / `PageSlice` (`origin`), which is the point —
  the honesty has to be *in* the artifact.
- `benchmark/s5/pagemap.py`'s reconstruction keeps working and keeps its byte-equality assertion;
  it is now a *fallback* for corpora produced before this ADR rather than the only path. (Its spans
  exclude the provenance header and the inter-page separators, where the sidecar's tile the whole
  document as `PageMap` requires. The two agree on every offset that lands in real page text; they
  differ only on who owns the seams.)
- A regression test drives an OCR'd multi-page document through the real
  `LocalFolderSource → PlaintextParser → chunk` path and asserts that a chunk from page N reports
  page N. That test fails on the pre-ADR code, which is the bar this defect earned.

### Other silent-`page 1` paths found

Surveyed while fixing this. In scope and fixed: `parser.plaintext`.

- **`latence_benchmark.chunking.as_chunk`** (`packages/latence-benchmark/src/latence_benchmark/chunking.py`)
  wraps each labelled benchmark document as one chunk with a hand-built single-page `PageSlice`.
  Honest — those corpora are page-less plain text — but it was stating it in the same way the
  defect did, so it now stamps `ASSUMED_SINGLE_PAGE`. Fixed here; no behaviour change.
- **`parser.endpoint`, `parser.document`, `parser.pdfplumber`, `parser.glm`, `parser.lighton`,
  `parser.lighton_vllm`, `parser.render`** all build their maps from genuinely per-page text via
  `PageMap.from_page_texts` and correctly keep `origin=PARSER`. Not affected. Note, however, that
  `parser.endpoint` trusts whatever page split its remote endpoint returns: an endpoint that
  returns one blob yields a one-span map labelled `PARSER`, i.e. the same over-claim one layer out.
  Out of scope here (it needs the endpoint contract to state whether the split is real); listed so
  it is not rediscovered as a surprise.
- **`PageOffsetIndex.page_for_offset`** degrades an empty map to page 1 with a drift diagnostic.
  Left as is: that path is unreachable (`PageMap` forbids an empty page list) and it is explicitly
  counted, not silent.
