# 1 · Ingest and document understanding

*`source` → `intake_screen` → `parse` → `chunk` → `content_screen`. Five stages that turn an
opaque folder of bytes into screened, retrieval-sized passages which can still name the page
and the character range they came from. Everything downstream — extraction, the graph,
retrieval — inherits whatever this segment gets right or wrong, because nothing later
re-reads the source file.*

The [index](index.md) gave each of these a sentence. This chapter is the mechanism: what
happens to a byte in order, which record types cross each seam, which knobs exist, and what
breaks when they are set wrong. Every number, default and model id below was read out of the
implementation; where a doc disagreed, the code won.

---

## What actually arrived, and why every corpus parsed as plain text

The campaign ran five corpora that are genuinely different kinds of input:

| Corpus | What it really is | Where its markdown came from |
|---|---|---|
| `vidoseek` | 292 documents of page **images** | S2 OCR |
| `ohr_bench` | 1,261 OCR-oriented documents (textbooks, scanned admin forms) | S2 OCR |
| `uda` | 1,917 long enterprise documents (annual reports, filings, papers) | S2 OCR |
| `multihop_rag` | 609 news articles, born-digital JSON | local prep, no OCR |
| `wiki2multihop` | Wikipedia context paragraphs from a parquet dev split | local prep, no OCR |

And yet all five `s3-pipeline` YAMLs configure `parser.plaintext`. That is not the framework
being text-only; it is the campaign's stage boundary. S3's `source.local_folder` is pointed at
`input/<dataset>/markdown`, and by the time S3 runs, that directory already holds markdown —
produced by one of two upstream paths, both of which land in the same place:

- The three image/scan corpora go through the **s2-ocr** stage, which rasterises every page and
  reads it with `lightonai/LightOnOCR-2-1B` served by vLLM, then assembles the per-page markdown
  into one file per document. `benchmark/sota-campaign/stages/s2-ocr/metrics.json` records the
  run: 142,593 pages rendered, 132,351 pages OCR'd, **3,470 documents assembled**, at 2.2–2.5
  pages/second on the big pod.
- The two text corpora never touch OCR at all. `prepare_multihop_rag` and
  `prepare_wiki2multihop`
  (`packages/latence-benchmark/src/latence_benchmark/campaign/pipeline_stage.py:248` and `:260`)
  convert `corpus.json` and `dev.parquet` into content-addressed markdown files locally. The
  campaign literally names them apart: `INDUCED_DATASETS = ("ohr_bench", "vidoseek", "uda")`
  at `pipeline_stage.py:62`, and the stage's own error message tells you which producer was
  supposed to fill the directory — *"run/approve s2-ocr first"* for those three, *"the local
  corpus prep should have produced it (stage bug)"* for the other two (`:413`).

So the honest statement is: **the campaign measured the pipeline downstream of document
understanding, with the OCR cost paid once in a separate stage.** That is a benchmarking
choice — it makes the S3 runs re-runnable without re-paying 142,593 pages of GPU OCR, and it
keeps the text corpora and the scanned corpora comparable, since both arrive as markdown. It
is not a limitation of the Parse seam. Wire `parser.lighton_vllm` (or `parser.render`, or
`parser.glm`) into the `parse` stage of any of those YAMLs, widen the Source's `extensions` to
include `pdf`, and the same DAG ingests the raw PDFs directly. The Parser roster later in this
chapter is the proof: eight Providers, one Protocol, one method.

There is one consequence of the split worth stating plainly, because it is visible in the
shipped artifacts. The S2 assembly that produced the campaign's markdown ran *before* the
page-map sidecar convention existed, so it threw the page boundaries away in the join. The
exported corpus records that faithfully: `stages/s3-pipeline/exports/*/export/corpus.jsonl`
carries `schema_version: 20`, and on the OCR'd datasets every chunk resolved to page 1. The
fix — a sidecar the OCR worker writes beside the markdown, which the Source finds and the
Parser validates — is schema v21, and it is described below under *Parse*.

---

## Stage 1 · `source` — making bytes addressable

**Provider:** `source.local_folder`, in `latence-core`
(`packages/latence-core/src/latence_core/stages/source.py`).
**Out:** one `ParserInput` per file.
**Compute:** CPU, deterministic, no model
(`ProviderProfile` at `source.py:86`).

A Source is the only side of the Parse seam that holds a `Storage` handle, so *finding* a file
is its job and *interpreting* it is not. `LocalFolderSource.produce` walks the configured URI
and, per file, does exactly six things:

1. **Skips page-map sidecars.** A `<file>.pagemap.json` belongs *to* a document; it is not one.
   The check runs before the extension filter, so widening `extensions` to include `json` can
   never turn a document's own page boundaries into a second, contentless "document"
   (`source.py:149`).
2. **Filters by extension**, defaulting to `{"txt", "md", "markdown"}`. The campaign narrows
   this to `["md", "markdown"]`. Widen it to feed a PDF-capable Parser; there is no second
   Source Provider for binary formats.
3. **Enforces basename uniqueness across the whole tree.** Two files called `report.md` in
   different subfolders raise a `ValueError` rather than being ingested, because downstream
   gold-set keys bind annotations to `Provenance.file_name` and would silently merge two
   distinct documents. The check runs *before* the retraction exclusion, so a retracted
   document sharing a basename with a kept one still surfaces the ambiguity.
4. **Content-addresses the bytes**: `document_id = "sha256:" + sha256(data).hexdigest()`. Two
   copies of the same file under different names are the same document id — which is what makes
   an incremental delta able to skip unchanged documents, and what makes
   `exclude_document_ids` a working retraction mechanism (a purge re-runs the pipeline with the
   deleted ids excluded *here*, so document-level stages simply never see them).
5. **Stamps Provenance**: `source_uri`, `file_name`, `file_type` (lowercased extension, no dot),
   `file_size`, `document_id`.
6. **Stamps Classification**: `sensitivity` and `category` straight from config, and a naive
   `language` — but only for a text extension. For any other accepted type the Source hands the
   raw bytes on with `language = "und"`, because it cannot decode a PDF and will not guess. The
   authoritative Classification is attached at Parse, from the decoded text.

Language detection is a stop-word signature match over the lowercase alphabetic tokens
(`latence_core/language.py:32`) — deliberately naive, deterministic, and ties broken by a fixed
table order. It is a hint that survives into every downstream record, not a claim.

### Configuration

| Key | Default | Effect | What breaks if wrong |
|---|---|---|---|
| `path` | *required* | Storage URI or bare path of the folder | Empty or non-string raises at construction |
| `sensitivity` | `"unknown"` | Stamped onto every record's `Classification` | Under-stamping propagates: redaction policy and export gating read this field on every downstream record |
| `category` | `None` | Optional content category | — |
| `follow_symlinks` | `False` | Whether to follow symlinks out of the source root | Setting `True` lets a symlink planted in the folder read files outside it |
| `extensions` | `["txt","md","markdown"]` | Extensions read, no dot, lowercased | Too narrow and documents vanish with no error; too wide and binaries reach a Parser that cannot decode them (which degrades to `PARSE_ERROR`, not a crash) |
| `exclude_document_ids` | `[]` | Content-addressed ids to skip | A stale list silently drops documents |

Two properties of `Storage.walk_files` matter architecturally and are stated as interface, not
implementation, in `latence_core/storage.py:412`:

- **Symlink escape is checked here, not delegated.** With `follow_symlinks=False` (the default)
  a file whose `os.path.realpath` escapes the root's real path is dropped. fsspec's own walk
  follows symlinks, so relying on a walk flag would have been the wrong shape.
- **The listing is not streamed.** The whole recursive listing is accumulated and *sorted*
  before the first path is yielded, so peak memory is O(number of files) and the first record
  costs a full prefix walk. That is deliberate: the framework's determinism invariant needs a
  stable document order, and no globally sorted order can be produced without seeing every
  path. Point this at a bucket prefix with millions of objects and you will feel it; scope the
  prefix.

---

## Stage 2 · `intake_screen` — the security control

**Provider:** `screening.intake_signature`, in `latence-core`
(`packages/latence-core/src/latence_core/stages/screening.py`).
**In / out:** `ParserInput` → `ParserInput` (minus quarantined).
**Compute:** CPU, deterministic, no model, no network, no model download, no unsafe
deserialization.

This is the checkpoint that runs **before** anything decodes the bytes. It is the only stage in
the front segment whose entire purpose is refusal, and it checks three things.

### Size ceiling

```python
if len(data) > self._max_bytes:
    return ("oversized", f"file is {len(data)} bytes, over the {self._max_bytes} cap")
```
`stages/screening.py:155`

Default `25 * 1024 * 1024` (`screening.py:132`). The campaign sets `max_bytes: 26214400` — the
same 25 MiB, pinned explicitly in the YAML rather than inherited, so the config is
self-describing.

**Be precise about what this cap does and does not bound.** It rejects a document from the
pipeline; it does not bound the read. `LocalFolderSource.produce` has already called
`storage.read_bytes(uri)` and holds the full file in memory by the time the screener sees it.
The cap protects Parse, Chunk and every model downstream from a pathological document; it is
not a defence against a hostile filesystem handing you a 40 GB file. If that is your threat,
bound it at the Storage layer or the mount.

### File-type spoofing

A small, high-value magic-byte table (`screening.py:62`) covers `zip`, `docx`, `xlsx`, `pptx`,
`gz`, `png`, `jpg`, `jpeg` — the formats a Parser will try to decode as structured documents.
The signature must be at **offset 0**; a substring match anywhere in the first kilobyte was the
old behaviour and was removed because an adversarial file could smuggle the signature past it.

PDF gets its own, stricter check, because a bare header is cheap to forge:

```python
while offset < limit and data[offset] in b"\x00\r\n\t \xef\xbb\xbf\xff\xfe":
    offset += 1
```
`stages/screening.py:193`

Only an eight-byte window of BOM/whitespace is tolerated before `%PDF-`, matching how pypdf
itself locates the header — the screener is deliberately at least as strict as the reader it
protects. A version digit must follow the header, and at least one genuine structural marker
(`%%EOF`, `startxref`, `xref`, `trailer`, ` obj`) must appear somewhere in the body
(`screening.py:205`). A forged header with no PDF structure is quarantined.

The limit worth knowing: **the screener judges against the declared extension.** A file with an
extension the table does not cover returns *"no known signature for this type — not a spoof we
can judge"* (`screening.py:171`). A ZIP named `notes.md` is not caught here — but neither is it
parsed as a ZIP, because the Parser also dispatches on the declared type, and the OCR-capable
Parsers repeat the offset-0 guard themselves before rasterising. The defence is layered, not
centralised.

### Zip bombs

```python
ratio = total_uncompressed / on_disk
if ratio > self._max_zip_ratio:
    return f"zip compression ratio {ratio:.1f}x exceeds the {self._max_zip_ratio}x cap"
```
`stages/screening.py:235`

The inspection reads only the archive's **central directory** (`ZipInfo.file_size` /
`compress_size`) — it never decompresses a member, so the inspection itself cannot trigger the
bomb, and there is no path-traversal surface. Two independent bounds fire: an absolute cap on
total uncompressed size (default 512 MiB) catches a bomb whose ratio is modest, and the ratio
cap (default 100.0) catches the classic case. Applied to `zip`, `docx`, `xlsx`, `pptx` — the
Office OpenXML formats *are* ZIP containers.

An archive that is declared as a ZIP but unreadable as one returns `None` here, deliberately:
the spoof check already ran, and an unreadable archive is Parse's decode failure to record, not
Screening's quarantine to claim.

### Configuration

| Key | Default | Campaign | Effect |
|---|---|---|---|
| `max_bytes` | `26214400` (25 MiB) | `26214400` | Oversized quarantine threshold |
| `max_zip_ratio` | `100.0` | `100.0` | Uncompressed ÷ on-disk ceiling |
| `max_zip_uncompressed` | `536870912` (512 MiB) | *(default)* | Absolute uncompressed ceiling |
| `check_spoof` | `True` | *(default)* | Disable only if you feed the pipeline extensions you control end to end |

### What quarantine actually means

A refused document becomes a `QuarantineRecord` — Provenance intact, `checkpoint`, `detector`,
`category`, `reason`, and `quarantined_bytes` recorded, **no content**. The Runner persists two
sidecars atomically alongside the checkpoint
(`latence_core/runner.py:1682`, paths from `run_store.py:141` and `:146`):

```
<run_dir>/quarantine/<stage_name>.jsonl   ← the withheld records
<run_dir>/screening/<stage_name>.jsonl    ← one ScreeningFinding per input, pass or not
```

Two properties follow, and they are the ones an auditor cares about. First, the findings file
carries a verdict for **every** input, not just the refusals — a clean run is a file full of
`PASS` findings, which is what makes "nothing was silently dropped" checkable. Second, because
the sidecars are persisted, a resumed run rebuilds the Quality Report's screening rollup from
Storage without re-running Screening, so the audit trail on a resumed run is identical to the
one on a fresh run. The rollup itself is `ScreeningQuality` (`latence_core/quality.py:313`):
counts plus the verbatim findings list.

Note the runner's dispatch shape here: `intake_screening` declares
`input_plan.shape = "gathered"` (`docs/reference/stage-contracts.json`), so the stage's input is
materialised as a list before the screener runs, unlike `parse` and `chunk` which are streamed
one record at a time. `ScreeningOutcome` holds passed + quarantined + findings, so the
materialisation is inherent to the Capability's shape rather than an oversight — but it means
the intake stage's peak memory is proportional to the corpus slice it screens, and that slice
is *raw bytes*. It is the one place in this segment where corpus size, not document size, sets
the footprint.

---

## Stage 3 · `parse` — the swappable seam

**Capability:** `Parser`, one method:

```
parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]
```

That is the whole interface. No base class is required — a Capability is a structural
`typing.Protocol`, so a Provider is anything with that shape registered under the
`latence.providers` entry-point group. `AdapterBase` exists as a template (device gating, the
native-fault guard, batching, Provenance carrying) and the heavier Providers use it, but the
in-core reference ones do not, which is the seam proving it is a seam.

### The eight shipped Parser Providers

Model ids come from each package's `_DEFAULT_MODEL`; licences from the `ProviderProfile` the
Provider declares in code, which records **weights and code separately** with a citation and a
verification date.

| Provider | Package | Model (`_DEFAULT_MODEL`) | Weights / code licence | Compute | Deterministic | Source |
|---|---|---|---|---|---|---|
| `parser.plaintext` | `latence-parser-plaintext` | *none* | Apache-2.0 (framework code) | cpu | yes | `provider.py:75` |
| `parser.document` | `latence-parser-document` | *none* (`pypdf`, BSD-3-Clause) | Apache-2.0 (framework code) | cpu | yes | `provider.py:96` |
| `parser.pdfplumber` | `latence-parser-pdfplumber` | *none* (`pdfplumber`) | MIT / MIT | cpu | yes | `provider.py:151` |
| `parser.render` | `latence-parser-render` | *none* — delegates OCR | Apache-2.0; deps pypdfium2 Apache-2.0/BSD-3, Pillow HPND, python-docx/openpyxl/python-pptx MIT, LibreOffice MPL-2.0 as an external subprocess | either | **no** (OCR is delegated) | `provider.py:208` |
| `parser.lighton` | `latence-parser-lighton` | `lightonai/LightOnOCR-1B-1025` | Apache-2.0 / Apache-2.0, verified 2026-07-08 | gpu | **no** | `provider.py:243` |
| `parser.lighton_vllm` | `latence-parser-lighton-vllm` | `lightonai/LightOnOCR-2-1B` | Apache-2.0 / Apache-2.0, verified 2026-07-09 | either | **no** | `provider.py:221` |
| `parser.glm` | `latence-parser-glm` | `zai-org/GLM-OCR` | MIT / MIT, verified 2026-07-08 (its PP-DocLayoutV3 layout component is Apache-2.0) | gpu | **no** | `provider.py:251` |
| `parser.endpoint` | `latence-parser-endpoint` | `"paddleocr-vl"` (a *served* model name, `model_id=None` — the package ships no weights) | Apache-2.0 (framework code + `openai` client) | either | **no** | `provider.py:109` |

Three things in that table are worth reading twice.

**`deterministic=False` is a declaration, not a defect.** All four VLM paths decode greedily
(`do_sample=False`) by default, which is as reproducible as the model allows on fixed input. But
float decoding is not promised byte-identical across hardware and library versions, so the
profile says so rather than overclaiming. The conformance suite checks a Provider against its
*declared* determinism — a Provider that claims determinism and does not have it fails.

**Two LightOn packages pin two different checkpoints.** `latence-parser-lighton` runs
`LightOnOCR-1B-1025` in-process through `transformers`; `latence-parser-lighton-vllm` calls a
warm vLLM server running `LightOnOCR-2-1B`. Both were licence-verified independently, on
different dates. They are not two ways of saying the same thing.

**The AGPL discipline is enforced by a test, not a note.** `parser.render` is a clean-room port
of a converter that used PyMuPDF (`fitz`, AGPL-3.0). It never imports `fitz`; a test greps the
package to keep it that way. LibreOffice is invoked as an external subprocess with an explicit
argument list (no shell), a 120-second timeout and an isolated temp dir — external, not
vendored, so its copyleft never reaches the package. `soffice` is optional: a host without it
degrades to `python-docx`/`openpyxl`/`python-pptx` text extraction, never a crash.

### What `parser.plaintext` does — including the page problem

For a `.md`/`.txt` document the transform is nearly an identity: decode the bytes (the Parser
owns the bytes→text turn, not the Source), fill in `char_start = 0` / `char_end =
page_map.total_chars`, inherit Provenance and Classification, emit `media_type:
"text/markdown"`.

The interesting part is pages. Markdown carries no page structure, so this Provider used to
answer every offset→page question with "page 1" — correct for a note, and a silent lie for the
3,470 OCR'd documents whose per-page markdown had been assembled elsewhere and handed back as
one blob. It was measured: `page_start == page_end == 1` on all 236k exported chunks. A
populated, plausible, wrong number is strictly worse than an absent one.

There are now three outcomes (`PlaintextParser._page_map_for`, `provider.py:141`):

| Situation | Result | `PageMap.origin` |
|---|---|---|
| A `<file>.pagemap.json` sidecar sits beside the document | real multi-page map, cross-checked against `len(text)` | `SIDECAR` |
| No sidecar, `require_page_map: false` (default) | one span covering everything | `ASSUMED_SINGLE_PAGE` |
| No sidecar, `require_page_map: true` | `PARSE_ERROR` record, run continues | — |

`PageMapOrigin` is what makes the absence *representable*: `PARSER` means the Parser walked the
source's pages and measured them; `SIDECAR` means a producer measured them out of band and the
consumer cross-checked; `ASSUMED_SINGLE_PAGE` means there was no page structure and the single
span is a convention so offset→page resolution keeps one code path — **not** a claim that the
document has one page. A consumer needing real page citation treats the third value as
*unknown*.

The cross-check is the part that makes a sidecar trustworthy. `PageMap.from_sidecar_json`
(`latence_core/contracts.py:485`) ignores any `origin` the file claims (a file cannot vouch for
its own provenance), and refuses when the sidecar's `total_chars` disagrees with the decoded
text length — a stale sidecar's spans would validate, resolve, and be confidently wrong. That
raises `PageMapSidecarError`, which the Parser turns into a `PARSE_ERROR` record. The producer
half lives in `campaign/ocr_lib.py:357`: `assemble_document_with_pages` computes the spans in
the same function that builds the joined string, so there is no second implementation of the
join to drift from.

### What the OCR path does when it *is* in play

Take `parser.lighton_vllm`, the production path (vLLM at concurrency measured ~2.7× faster than
optimised in-process `transformers` on the same GPU). Per document:

1. **Rasterise** through the shared `PageRasteriser` — deliberately *shared* with
   `parser.lighton` so the two back-ends cannot silently diverge at the pixel boundary. The
   recipe is LightOn's published one: `DEFAULT_DPI = 200`, downscaled to
   `DEFAULT_MAX_LONGEST_DIM = 1540` px on the longest side with LANCZOS, aspect preserved
   (`rasterize.py:50`, `:55`). Feeding an A4 at 200 DPI un-downscaled is ~2,339 px tall, about
   2.3× the vision tokens — the single biggest throughput lever, and *not* a byte-identical
   no-op, which is why it is a knob and not a hard-coded step.
2. **Bound the raster.** This is the first framework Parser that reads *pixels*, and
   `max_bytes` does not bound pixels: a tiny PDF can declare enormous page dimensions, and an
   image's declared pixel count is independent of its byte size. So the render scale is clamped
   by a `MAX_RENDER_MEGAPIXELS = 25.0` budget (`rasterize.py:124`) — scale reduced by
   `sqrt(budget/pixels)` — and `Image.MAX_IMAGE_PIXELS` is tightened (and restored afterwards,
   no global side effect) so Pillow *raises* `DecompressionBombError` rather than allocating.
   That native fault maps through the adapter guard to a typed error and degrades to a graceful
   `PARSE_ERROR`.
3. **POST each page image** as a base64 PNG data URL to `{base_url}/v1/chat/completions`, with
   `max_tokens` defaulting to **4096**. That default is pod-corrected, not copied from the model
   card: at LightOn's example 1024, dense real banking pages truncated ~25% of the OCR text
   (19,905 → 14,951 characters) — silently dropping document content, which for enterprise input
   is worse than being slow. Page requests fan out concurrently under an
   `asyncio.Semaphore(max_concurrency)` (default 8) with exponential-backoff-with-jitter retry,
   and results are re-assembled in **input page order regardless of completion order**, so a
   shuffled completion order can never perturb the page map. A per-page endpoint failure fails
   open to a marker for *that page only* — a 40-page scan does not lose 39 good pages to one bad
   one.
4. **Assemble** through `PageMap.from_page_texts` (`contracts.py:443`), the same exact-offset
   seam every other Parser uses. Boundaries are exact by construction because the string and the
   spans are built together; the inter-page separator counts toward the *preceding* page so the
   spans tile `[0, len(text))` with no un-attributed gap. Here `origin` stays `PARSER` — these
   boundaries were measured.

The in-process `parser.lighton` differs only in step 3: it batches page images into one
`generate()` call (`ocr_batch_size`, default 4) with left-padding and a per-item prompt-width
slice, which under greedy decoding is record-for-record identical to running each page alone.
That prompt-token slice is load-bearing and both Providers document why: HF `generate` returns
the *full* prompt + continuation sequence, so decoding without slicing prepends the
chat-template scaffolding to the OCR markdown on **every page**, corrupting content, page-map
offsets, size and language at once.

One Parser deserves an explicit caveat. `parser.endpoint` asks a served model to delimit pages
with a literal token and then splits on it:

```python
page_texts = markdown.split(_PAGE_SEPARATOR)   # "\n\n---PAGE---\n\n"
```
`latence-parser-endpoint/.../provider.py:191`

Those page boundaries are *inferred from model output*, not measured from the source. They are
therefore weaker evidence than the rasterising Providers' boundaries, and the `PageMap` carries
`origin=PARSER` all the same — the origin enum distinguishes measured-from-source from
assumed, not measured-well from measured-badly. Worth knowing before you cite a page number
from an endpoint-parsed corpus.

### The failure contract every Parser shares

A document the Parser cannot handle — a bad codec hint, a corrupt or password-protected PDF, a
stale sidecar, an unsupported type, a terminal endpoint error, a decompression bomb — becomes a
`DocumentRecord` with `disposition = PARSE_ERROR`, empty content, no page map, Provenance
intact, and the reason recorded. The run continues; the failure is auditable in the Quality
Report (`ParseQuality.parse_errors`, `quality.py:261`); and the Chunk stage skips such records
entirely, so nothing content-free ever reaches the corpus. That contract is identical across all
eight Providers, which is what makes swapping one for another safe.

The typed error carries *provider + operation + native exception type name* and nothing else —
never input bytes, never extracted text, never OCR output. OCR output is untrusted and
PII-bearing; a stack trace is not a place to put it.

---

## Stage 4 · `chunk` — the algorithm, in full

**Provider:** `chunk.markdown` → `MarkdownChunker`
(`packages/latence-core/src/latence_core/stages/chunk.py`), wrapping the engine at
`packages/latence-core/src/latence_core/chunking.py`.
**In / out:** `DocumentRecord` → `ChunkRecord`.
**Compute:** CPU, deterministic, no model. Identical `text` + `budget` yields byte-identical
chunks.

The engine does three things in order: strip markup while tracking every deletion, budget the
*stripped* text into pieces, then map each piece's boundaries back through the strip map so its
recorded span points into the **original** markdown.

### Step 1 — 18-pattern markup stripping, with an offset map

You cannot map offsets through `re.sub`: it rewrites the string opaquely. So `strip_markup`
(`chunking.py:164`) builds a per-original-character **keep mask**. Every pattern marks the
positions it deletes; a replacement that backreferences a capture group keeps that group's
characters (their offsets survive), and a replacement that is a literal deletes the whole match.
The surviving characters, in order, are the stripped text, and each one remembers its original
offset in a `mapping` array with `len(stripped) + 1` entries — the sentinel so an exclusive end
offset maps cleanly.

**The order of the 18 patterns is inert.** They are listed in a reading order — code, then block
constructs, then inline emphasis, then residual HTML — but every pattern is matched against the
*original* text and contributes only deletions to that one shared mask, so the output is the
union of those deletions however the list is sorted. Shuffling it 200 times yields byte-identical
stripped text and offset maps (`test_strip_pattern_order_does_not_change_the_result`). Running
code first therefore also buys no protection for code: a fence's contents are re-interpreted like
any prose, so a fence holding `x = a | b ***c***` strips to `x = a b *c*`. The source comment
claimed both the opposite things until 2026-08-25, when they were checked.

**Table cells are separated by a real character, never welded.** Pattern 13 reads:

```python
(re.compile(r"\|[ \t]*$|(?:(?<=[ \t])|^)\|[ \t]*|(\|)", re.MULTILINE), r"\1"),
```
`chunking.py:132`

The constraint that shapes it is `_apply_match` (`chunking.py:192`): a replacement holding no
backreference is a **deletion** — deliberately, and documented in its own docstring, because an
injected literal has no original offset to map. So the separator between two cells has to be a
character that already exists in the source. A pipe with whitespace in front of it (or one that
opens a line) is dropped together with the padding that *follows* it, and the space in front
survives as the separator; a row-closing pipe goes with its trailing padding; a pipe with no
whitespace on its left — the compact form, where there is no space to promote — is kept, because
the grid character is then the only thing that can hold the two cell texts apart:

```
"| Revenue | 1,200 | 3,400 |"   →   "Revenue 1,200 3,400"
"|Revenue|1,200|3,400|"         →   "Revenue|1,200|3,400"
```

Until 2026-08-25 the pattern was `[ \t]*\|[ \t]*` with a literal `" "` replacement, which — by
the deletion rule above — removed the pipe *and the whitespace on both sides of it*, so the cell
texts abutted and `"| Revenue | 1,200 |"` became `"Revenue1,200"`. That was not only a cosmetic
degradation of tabular structure. A welded row **hid PII from the Redactor**: `| Ada Lovelace |
123 45 6789 | DE89370400440532013000 |` arrived as `Ada Lovelace123 45 6789DE89370400440532013000`,
where the no-dash SSN recogniser's letter-glue guard refused the digits and the IBAN pattern's
leading `\b` could not fire between `e` and `D`. The chunk came back with an empty `pii_spans`
and `masked_content == content` — the same values that were masked in a sentence were left in the
clear in a table (chapter 2, §4.4; pinned by
`test_chunk_provider.test_table_row_pii_is_masked_identically_to_prose`).

**Every measurement taken before that date saw the welded text.** Chunk text is what the embedder
and the extractor consume, so any number produced from a table-bearing corpus was measured on a
different corpus than this code now produces — 13 of the 40 enterprise-gold documents carry a
markdown table, and the OCR'd campaign corpora are table-heavy by construction. Those baselines
are *pre-fix* until they are re-run, and nothing in this repository was regenerated by the fix:
numbers must not be compared across it.

**HTML comments survive.** Pattern 18 strips `</?[a-zA-Z][^>]*>`. `<!--` does not start with a
letter, so comments are not matched — which is why the campaign's OCR provenance header is
still sitting in chunk 0 of every OCR'd document. Underscore italics are also deliberately left
intact (pattern 15 handles the asterisk form only), because `_` is ubiquitous in real text
(`my_var`, `a_b.txt`) and stripping it would corrupt legitimate content; doubled `__bold__` is
unambiguous and *is* handled by pattern 14.

### Step 2 — the token estimate

```python
tokens = _WORD_RE.findall(text)          # r"\w+|[^\w\s]"
total = 0
for tok in tokens:
    total += 1
    if tok.isalnum() and len(tok) > 6:
        total += (len(tok) - 1) // 6
```
`chunking.py:214`

Word/punctuation tokens, plus roughly one extra token per six characters of any word longer
than six — a crude BPE stand-in. No tokenizer is downloaded, nothing is fetched, and the count
is hash-stable, which is what lets a seeded test assert byte-identical chunk boundaries. A real
tokenizer is a drop-in Provider config override; the *default* must be deterministic.

`estimate_tokens("hello world") == 2`. `estimate_tokens("internationalization") == 4`.
`estimate_tokens("Data-driven, 2026.") == 6`.

The estimate is not the model's tokenizer, and the gap between them is a design constraint
severe enough to have its own section below.

### Step 3 — windows, breaks, overlap, and the short tail

`_budget_pieces` (`chunking.py:438`) walks the stripped text:

1. `_window_end` grows a window token by token until the next token would exceed `max_tokens`
   (measured in exactly the units `estimate_tokens` counts, so the ceiling means what it says).
2. `_best_break` (`chunking.py:488`) then backs the boundary up to the **last** clean break in
   that window, preferring **paragraph** (a blank line) → **sentence** (`.`, `!` or `?` followed
   by whitespace or end of text) →
   **any whitespace**. The break lands *after* the delimiter, so the boundary character stays
   with the left piece. If no break exists at all — one enormous unbroken token — it falls back
   to the hard token boundary, which is the only circumstance in which a chunk is split
   mid-word.
3. `_overlap_start` (`chunking.py:510`) walks a precomputed token index backward from the
   boundary until it has accumulated `overlap_tokens` worth of cost, and the next window starts
   there. The index is the fix for issue #26: the old implementation re-scanned the document
   from offset 0 on every chunk, making chunking O(n²) in document length — a genuine
   malicious-input DoS lever, where one large in-cap `.txt` drove a single core into minutes or
   hours. Byte-identical boundaries, `O(log n)` locate plus a bounded backward walk.
4. `_merge_short_tail` (`chunking.py:540`) folds a **final** piece below `min_tokens` into its
   predecessor.

Two consequences that surprise people, both easy to reproduce:

**`min_tokens` protects the tail, and only the tail.** It merges `pieces[-1]`. It does nothing
about a short chunk in the middle.

**Preferring the last paragraph break can produce a chunk far below the ceiling.** If the only
paragraph break inside the window sits early — a heading followed by one long unbroken
paragraph is the canonical case — the boundary lands there. On a document beginning
`# Title\n\nAlpha beta gamma. …`, with a 60-token budget, chunk 0 is `"Title\n\n"`: **one
token**, and `min_tokens` does not save it because it is not the tail. Heading-dense markdown
produces a long tail of small chunks. If that matters for your corpus, `chunk.sentence_window`
is the alternative shipped Provider: it packs whole sentences greedily up to the ceiling with
**no overlap**, so a chunk never ends mid-sentence unless a single sentence exceeds the budget
(and it ignores `overlap_tokens` entirely, forcing it to 0 so the `ChunkBudget` invariant holds
— `stages/chunk.py:172`).

### Step 4 — what the offsets guarantee

```python
orig_start = strip.mapping[start]
orig_end = strip.mapping[end - 1] + 1 if end > start else strip.mapping[start]
```
`chunking.py:280`

`char_start` and `char_end` are a half-open span in the **original parsed markdown**, derived
from the first and last *surviving* character of the piece. `content` is the **stripped** text.
These are different coordinate systems, and this is the single most important thing a developer
must internalise about a `ChunkRecord`:

> `document.content[chunk.char_start:chunk.char_end]` is a **superset** of `chunk.content`, not
> equal to it. It contains the markup that was stripped out.

That is why `char_start + local_offset` is only a *lower bound* on the true original position of
something found inside a chunk, and why every chunk carries an `OffsetMap`: a run-length list of
`(local_start, doc_start)` breakpoints, one per contiguous surviving run
(`_offset_segments`, `chunking.py:391`). A markup-free chunk collapses to a single identity
segment `(0, char_start)`, so on the plain-text path resolution is exactly `char_start + local`
and nothing changed. `latence_core/offsetmap.py` is the resolver, with an inverse
(`resolve_original`) for going the other way; both clamp rather than raise, because Provenance
resolution must never be the thing that fails a run.

Because overlap is real overlap, **adjacent chunks' original spans intersect**. With
`overlap_tokens: 80` a chunk ending at character 274 is followed by one starting at 232. Any
consumer counting distinct source characters, or deduplicating by span, has to account for
that.

### Step 5 — pages, per chunk

Each emitted chunk gets `page_start` / `page_end` resolved through the parent's
`PageOffsetIndex` (`latence_core/pagemap.py:69`) — a `bisect` over cumulative page starts,
`O(log n)` in page count, with a fuzzy drift-recovery fallback that clamps an out-of-range
offset to the nearest plausible page and *records the drift* rather than crashing. The drift
counters surface in the Quality Report (`ParseQuality.page_drift_*`); a clean parse has zero
recoveries.

Each chunk also carries its own `PageSlice` — exactly the document page spans overlapping its
own `[char_start, char_end)`, typically one, two when it straddles a page break
(`stages/chunk.py:233`). This is schema v19, and the design note in
`latence_core/page_index.py` is worth reading in full because it is a case study in removing a
failure mode rather than mitigating it. Before v19 the document's whole page map rode **one**
record — the document's first chunk — and a resolver adopted it from whichever record it saw
first. That made page resolution depend on two things that are not properties of the chunk being
resolved: **order** (any batching extractor resolves out of stream order) and **survival** (any
stage that removes chunks could remove the map-carrying one). Both degraded silently to a
plausible wrong page. Four audit rounds patched the consequences — a re-homing pass, then an LRU
retention window, then a fix to that window's justification — and never the shape. A slice
removes the shared state, so there is nothing left for an ordering or a dropped record to be
wrong about: the failure is not mitigated, it is **inexpressible**. The payload win survived
too — ~2 spans (~60 bytes) per chunk against the ~30 KB whole-document map a 519-page document
would otherwise duplicate onto ~399 records.

A chunk carrying no slice raises `PageSliceMissingError` rather than returning `None` for the
caller to degrade on, because "degrade" meant emitting a page that looked right and was not.

### Configuration

| Key | Engine default | Wizard default | Campaign | Effect |
|---|---|---|---|---|
| `max_tokens` | `512` (`chunking.py:240`) | `768` (`setup/recipe.py:206`) | `640` | Per-chunk ceiling in the estimator's units |
| `overlap_tokens` | `64` | `80` | `80` | Tokens of the previous chunk repeated at the start of the next. Must be in `[0, max_tokens)` or `ChunkBudget` raises |
| `min_tokens` | `16` | `8` | `8` | A **trailing** remnant below this merges into its predecessor |

Three different defaults for the same knob is not drift; they are three different authorities.
`ChunkBudget`'s dataclass defaults are what the engine does when nobody configures it. The setup
wizard emits `768/80/8` into a generated stack **and derives the downstream model window to
match**. The campaign pinned `640/80/8` because that is the largest budget that fits a *fixed*
768-token model window — which is the subject of the next section.

---

## The budget is calibrated against a tokenizer it does not use

This is the part of the front segment with the most leverage on final quality, and it is
invisible unless you go looking.

`chunk.markdown` counts in `chunking.estimate_tokens`. The extraction and redaction Providers
downstream run **mdeberta-v3**, whose tokenizer counts differently. The framework does not
pretend otherwise; it measured the discrepancy and encoded it:

```python
#: Pod-measured chunker-token -> mdeberta-token expansion (768 chunker tokens -> max 921 mdeberta,
#: 921/768 = 1.199). Used to size the model window from the chunk budget. Deliberately the MAX
#: observed ratio, not the median (666/768 = 0.87): the window is a truncation boundary, so it must
#: cover the worst chunk, not the typical one.
MDEBERTA_TOKEN_EXPANSION = 1.2
```
`latence_core/setup/recipe.py:210`

Read the numbers carefully. A 768-chunker-token cap measured a **median** of 666 mdeberta tokens
and a **maximum** of 921. The median is *below* the cap — the estimator over-counts on typical
text. The maximum is 20% *above* it. Sizing on the median would be sizing on the comfortable
case; the window is a truncation boundary, so it must cover the worst chunk.

What happens past the boundary is the reason this matters. `fused_entity_relation.gliner2`
threads a native `max_len` (default 768) into the model call, and its own docstring is blunt
about the alternative:

> `max_len: int` — the mdeberta-v3 model token window (default 768) … A chunk that tokenizes
> past it is **SILENTLY truncated** by the model (tail entities/relations lost, offsets
> unreliable); this bounds the window explicitly.
>
> `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:282`

At a 768-token chunk budget against a fixed 768 window, roughly **2 chunks in 7** measured past
the window. Not dropped, not errored — truncated, with the tail of the chunk's entities and
relations simply never extracted, and offsets past the window unreliable. Nothing in the output
says so.

Hence the derivation, which is pure arithmetic (`model_window_for`, `setup/recipe.py:237`):

```
window = ceil(chunk_max_tokens × 1.2) rounded up to a multiple of 64
```

| Chunk budget | Derived model window | Derived `microbatch_max_texts` |
|---|---|---|
| 512 | 640 | 4 |
| **640** | **768** | 4 |
| 768 | 960 | 3 |
| 1024 | 1280 | 2 |

Two things fall out. First, **the campaign's 640 was never a taste** — it is 768 ÷ 1.2, the
largest chunk budget that fits a fixed 768-token window, which is exactly why the campaign YAMLs
set no `max_len` at all and let the Provider default of 768 stand. The derivation reproduces the
historical pairing exactly, which is what makes it a generalisation of the calibration rather
than a replacement for it. Second, a wider window is not free: it costs VRAM quadratically in
attention and linearly in the padded batch, so `microbatch_max_texts_for` buys the extra
positions back out of the batch (`POD_SAFE_PADDED_POSITIONS = 4 × 768`, the point that peaked at
21.6 GiB and passed on a 32 GB pod).

**One precision worth stating.** It is tempting to say the downstream encoder's window "is
*derived* from this budget rather than fixed". That is
true of a stack the setup wizard emits — `model_window_for` runs at recipe-emission time and
writes an explicit `max_len` into the generated YAML. It is **not** true of a hand-written
pipeline file. `resolve_max_len` is a plain config read with a fixed default:

```python
return max(1, int(config.get("max_len", default)))    # default 768
```
`latence_core/providers/perf.py:1143`

So if you hand-edit a YAML and raise `chunk.max_tokens` past 640 without also raising
`extract.max_len` and `redact.max_len`, nothing warns you and the extractor starts silently
truncating. That is the single most expensive mistake available in this segment of the pipeline.

---

## Stage 5 · `content_screen` — flag, don't drop

**Provider:** `screening.content_keyword` → `KeywordContentScreener` (`stages/screening.py:295`).
**In / out:** `ChunkRecord` → `ChunkRecord` (some now carrying a `RiskMarker`).
**Compute:** CPU, deterministic, no model.

The second checkpoint runs *after* chunking, over each chunk's text, and its default disposition
is the opposite of intake's. Intake refuses; content screening **marks**:

```python
marker = RiskMarker(category=category, detector="screening.content_keyword",
                    score=min(score, 1.0), reason=reason)
flagged = chunk.model_copy(update={"risk_markers": [*chunk.risk_markers, marker]})
```
`stages/screening.py:367`

The marker is a field on the record contract, not an out-of-band note, so it survives into the
export and a RAG consumer can filter on `risk_markers`. That asymmetry is a deliberate ruling: a
legitimate document may *quote* an injection string — a security policy, an incident report, a
paper about prompt injection — and silently dropping it would be worse than marking it. A
configurable `quarantine_threshold` (default `1.01`, i.e. never, since a single hit scores
exactly `1.0`) escalates a hit to Quarantine; set it below `1.0` to make any hit a refusal.

Seven prompt-injection regexes and two harmful-content keywords, all lowercase-matched
(`screening.py:274`, `:283`) — deliberately small and high-precision, since the cost of a false
positive here is a filterable flag rather than lost data.

### The cross-boundary pass

Per-chunk scanning has an obvious evasion: split the phrase across two chunks. So a second pass
re-scans the **seam** of every consecutive same-document chunk pair
(`_flag_cross_chunk_injection`, `screening.py:386`), over a bounded 240-character window
(`_BOUNDARY_WINDOW`, `:292`) — the tail of the previous chunk plus the head of the next. Bounded
so the pass stays linear in chunk count and cannot be turned quadratic by a pathological corpus;
240 characters comfortably spans the longest pattern even when the split lands mid-phrase.

Three details make it correct rather than merely present. A match counts only if it genuinely
**straddles** the boundary — begins in the tail, ends in the head — so an injection wholly
inside one side is not double-reported. A chunker typically consumes the whitespace it splits
on, so when neither side carries boundary whitespace a single space is reconstructed at the seam
and accounted for in the straddle test, otherwise a phrase split exactly on a word gap would be
glued into one token and hidden. And when a hit fires, **both** chunks are flagged, so the
corpus carries the marker wherever the reader lands.

Note the interaction with `overlap_tokens`. With overlap configured, adjacent chunks already
share text, so many boundary-straddling phrases are caught by the ordinary in-chunk pass. The
seam pass is what covers the case where they are not — including `chunk.sentence_window`, which
has no overlap at all.

### The second Provider

`screening.content_fuzzy` (`FuzzyInjectionContentScreener`, `screening.py:600`) sits behind the
identical Protocol and trades a different way. It compacts chunk text to lowercase
alphanumerics only — a single linear pass, no regex, ReDoS-proof by construction — and matches
compacted signatures, so `i g n o r e   p r e v i o u s   i n s t r u c t i o n s` and
`ignore/previous.instructions` both fire where the keyword screener misses them. Its scan cap is
on the **compacted** output (8,192 alphanumeric characters), not on a raw prefix, specifically
because a raw-prefix cap could be defeated by padding an injection past the window with
cost-free separators; a secondary 1 MiB raw cap bounds the pathological all-punctuation chunk.
It does *not* implement the cross-boundary pass — pick per threat model.

### What lands in the Quality Report

Same two sidecars as intake, under this stage's name: every chunk gets a `ScreeningFinding`
(pass, flag, or quarantine, with the reason and the marker), and `ChunkQuality.chunks_flagged`
plus `ScreeningQuality` roll them up. `ChunkQuality.offset_preserving`
(`quality.py:290`) is the S3 round-trip invariant made checkable: it is true only when *every*
emitted chunk carried a valid source character span **and** page span.

---

## What this segment guarantees, and where the work happens

| Stage | Compute | Deterministic | Runner input shape | Peak memory scales with |
|---|---|---|---|---|
| `source` | CPU | yes | root (reads Storage) | number of files (the sorted listing) |
| `intake_screen` | CPU | yes | **gathered** | corpus slice, in raw bytes |
| `parse` | CPU with `plaintext`/`document`/`pdfplumber`; GPU or served endpoint with the OCR paths | yes on the CPU paths, **declared no** on every VLM path | streamed | one document (plus the model, if any) |
| `chunk` | CPU | yes | streamed | one document |
| `content_screen` | CPU | yes | **gathered** | corpus slice, in chunk text |

Four of the five stages run on CPU with no model and no network, and are byte-for-byte
reproducible: same input plus same config yields the same records, which is what makes a run
auditable and a regression attributable. The only stage that can need a GPU is Parse, and only
when the corpus needs pixels read — which is precisely why the campaign paid that cost once, in
S2, and ran S3 on markdown.

The two `gathered` stages are the memory shape to plan around. Both screening Capabilities
materialise their input, because `ScreeningOutcome` returns passed and quarantined together and
because the content screener's seam pass needs adjacency. Everything else in this segment
streams one record at a time.

---

## What the next stage receives

Out of `content_screen` comes a stream of `ChunkRecord`s. Each one carries:

- `record_id` of the form `<document_record_id>#chunk-<i>`, and `index`, its position within its
  parent document;
- `content` — markup-stripped, embeddable text, `media_type: "text/markdown"`;
- `token_count` — the estimator's count for exactly that text;
- `provenance` — the full source lineage (`source_uri`, `file_name`, `file_type`, `file_size`,
  the content-addressed `document_id`) narrowed by `char_start` / `char_end` in the **original**
  markdown and `page_start` / `page_end` in the source document;
- `page_slice` — this chunk's own page spans, so any sub-chunk offset resolves to a real page
  from the record in hand, in any order, with any sibling missing;
- `offset_map` — the run-length map from stripped-content offsets back to original-markdown
  offsets;
- `classification` — language, category, and the `sensitivity` the Source stamped, inherited
  unchanged;
- `risk_markers` — empty for a clean chunk, populated for a flagged one.

Everything after this point works on that record and never re-reads the source file. Extraction
finds a mention at a chunk-local offset and resolves it through `offset_map` and `page_slice` to
a true original offset and a true source page; redaction masks spans in the same coordinates;
the graph's edges cite them. Which is the subject of
[chapter 2 · Extraction and privacy](02-extraction-and-privacy.md) — schema induction, fused
entity and relation extraction, redaction, and profiling.
