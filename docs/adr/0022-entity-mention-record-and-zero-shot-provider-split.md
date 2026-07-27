# Entity mentions are first-class inter-Stage Records; the zero-shot label set lives in config; the deterministic extractor ships in core, the learned one is a separate package

S4 adds **Entity Extraction** (CONTEXT): the Stage that finds typed entity
mentions in chunked text (zero-shot NER). Four design questions had non-obvious
answers, recorded here.

## An EntityMention is a first-class inter-Stage Record, not a field on the chunk

A mention could be modelled as a list nested inside each `ChunkRecord`. It is not.
`EntityMention` is its own versioned core contract (`SCHEMA_VERSION` → 4), a
subclass of `Record` — so it carries Provenance and Classification like every other
inter-Stage carrier, is threaded through the DAG as a Stage output, is checkpointed
and resumed by the Runner uniformly, and is exported to the corpus directly.

**Why.** The downstream corpus-level Stages consume mentions, not chunks:
Relation Extraction (S5) relates *mentions*; Disambiguation (S8) links *mentions*
to canonical entities and attaches Evidence at *mention* offsets; Graph Assembly
builds nodes from *mentions*. Making the mention the record (with its own
Provenance span) means every downstream Stage receives the exact contract it needs
with offsets already resolved to the source document — no re-derivation from a
chunk-nested list, no second offset code path. It also keeps the Runner's
one-output-list-per-Stage invariant intact (the DAG threads `EntityMention` the
same way it threads `ChunkRecord`), and lets Export materialize mentions with zero
special-casing. Rejected: mentions-inside-chunks (couples every consumer to the
chunk shape, hides the mention's own Provenance, and forces each downstream Stage
to re-resolve offsets).

## A mention's offsets index the ORIGINAL document, resolved through the chunk's offset map

> **Correction (2026-07-07, superseded by ADR-0031).** As first shipped, this section
> claimed the shift `doc_offset = chunk.provenance.char_start + local_offset` yields the
> mention's original-document offset. **That is false for a chunk that contains markup.**
> Chunk `content` is markup-**stripped** (`chunking.strip_markup` deletes 18 constructs),
> so `local_offset` is measured over shorter text; because stripping only ever *deletes*
> characters, the shift is a **lower bound** on the true offset, exact **only** when no
> markup was stripped before the mention within its chunk. The S4 fixtures passed because
> they were all markup-free plain text. ADR-0031 corrects the root cause: Chunk now carries
> a stripped-`content`→original **offset map** (`OffsetMap`) on each `ChunkRecord`, and the
> extractor resolves a mention's local span through it (`OffsetIndex`) to the EXACT original
> offset. The description below is retained for history; read `char_start + local_offset` as
> "the lower-bound shift the offset map replaced". The page-resolution reasoning is unchanged
> — it simply now runs on the corrected offset.

> **C1 amendment (2026-07-27, schema v19 — a chunk describes its OWN pages).** The decision
> below originally read: "Chunk now carries the parent document's `page_map` forward on **each**
> `ChunkRecord` (optional field; absent ⇒ the mention degrades to the inherited chunk range)."
> **That mechanism was superseded twice, and the degradation contract it prescribed is a bug.**
>
> *First (schema v18, 2026-07-26).* `PageMap` is DOCUMENT-scoped state — contiguous spans covering
> the whole assembled markdown, invariant across every chunk of its document — so carrying it by
> value on all N chunks was pure duplication: on a 519-page document chunked at 640 tokens, ~30 KB
> of JSON on each of ~399 records, ~92% of every chunk checkpoint line, paid again on every
> checkpoint write, streaming read-back and decode. v18 kept that payload win by stamping the map
> on a document's **first** chunk only and having a document-scoped `PageIndexResolver` adopt it
> from the stream. **That shape is gone too**, and it is instructive why: it made page resolution
> depend on two things that are not properties of the chunk being resolved — that the map-carrying
> record was OBSERVED first (every batching extractor resolves out of stream order), and that it
> SURVIVED every filtering Stage (any chunk→chunk Capability may drop chunk 0). Both failures
> degraded silently to the chunk's inherited page range. Three patches (`rehome_page_maps`, an LRU
> retention window, then that window's justification) hardened the consequences, never the shape.
>
> *Now (schema v19).* No page map rides any chunk. The Chunk Stage stamps a **`PageSlice`** on
> **every** chunk (`stages/chunk.py`: `page_slice=PageSlice.for_span(doc.page_map, ch.char_start,
> ch.char_end)`) — exactly the document page spans overlapping that chunk's own
> `[char_start, char_end)`, in the document's original coordinates. One span for a chunk inside a
> page, two for a chunk straddling a break. That is the whole of what a chunk needs, because the
> only page question asked of it is *sub-chunk*: which page does an offset INSIDE it fall on.
> Order- and drop-independence are now properties of the record, so the v18 failure mode is not
> mitigated but **inexpressible**, and ~99.8% of the payload win is kept (~2 spans, ~60 bytes,
> against a ~30 KB whole-document map). `latence_core.page_index.PageIndexResolver` is
> correspondingly **stateless**: no `observe`, no per-document memoisation, no retention window;
> `rehome_page_maps` and `PageIndexResolver.from_documents` were deleted.
>
> **What this changes for a Provider author.** Do **not** read the page field off the record at
> all. Create **one** `PageIndexResolver` per Capability call (for the corpus-level drift roll-up,
> not for correctness) and stamp provenance from
> `resolver.pages_for_span(chunk, doc_start, doc_end)`, which returns `(page_start, page_end)` and
> raises `PageSliceMissingError` if the producing Chunker stamped no spans — there is no `None`
> to degrade on, because "degrade" meant emitting a page that looks right and is not (ADR-0034).
> For a relation's covering span, whose endpoints may sit in DIFFERENT chunks, use
> `pages_for_covering_span(head.provenance, tail.provenance)` instead; no single chunk can answer
> for it. The "absent ⇒ inherit the chunk range" fallback this section used to prescribe hands a
> mention the parent chunk's whole page range — a plausible-looking WRONG page, and exactly the
> ADR-0021/0031 violation audit R1 found in four shipped ecosystem Providers that had followed
> this ADR verbatim. See `docs/writing-an-adapter.md` and the schema-v19 CHANGELOG entry. The
> page-resolution *reasoning* below is unchanged: a mention still resolves its OWN page from its
> own document offset, never inherits the chunk's.

An extractor sees a *chunk's* text and returns *chunk-local* offsets. But a mention
must resolve back to its source **page** via `PageOffsetIndex`, which indexes the
**parent document's** assembled markdown (ADR-0020/0021). The Provider maps each
chunk-local match back to the parent document through the chunk's offset map
(ADR-0031; originally, and incorrectly for markup, a bare shift by the chunk's own
`provenance.char_start`): `doc_offset = resolve(local_offset)`. The mention's **page** is
then re-resolved from *that* document offset through `PageOffsetIndex` — the same
resolver the chunk Provider uses (ADR-0021) — **not** inherited wholesale from the
chunk's `page_start`/`page_end`. A default chunk budget (512 tokens) routinely spans
several short source pages, so a mention on a later page of a multi-page chunk must
cite *its own* page; inheriting the chunk's first page would violate the S4 core
requirement "Provenance resolving to the correct source page". To make this
resolvable at the mention Stage, Chunk stamps each `ChunkRecord` with its own `page_slice`
— the document page spans overlapping that chunk (**schema v19**), with the mention Stage
resolving through the stateless `PageIndexResolver` (C1 amendment above). The whole
document `page_map` no longer rides the chunk stream: it did on each `ChunkRecord` as first
shipped, and on the document's first chunk only under v18, and both shapes were replaced.
A boundary-straddling mention resolves `page_start` from its
start offset and `page_end` from its last included character, so it records both
pages. This completes the S4 round-trip (mention → offset → original page). The
`EntityMention` contract *enforces* the span: a mention with no char span or no page
span fails validation, so a Provider that forgets the shift cannot silently emit
un-resolvable mentions. This is the same subtle invariant ADR-0021 called out for
chunks — offsets index the original, never the local text — carried one Stage
further, and now the page is resolved at the same granularity as the offset.

## The zero-shot label set is Pipeline config; the Capability names no types

Entity Extraction is zero-shot: the entity types are a per-corpus choice, not a
framework constant. The `EntityExtractor` Capability names no labels, core hardcodes
none, and the label set is supplied entirely through the Stage `config`. The same
Provider extracts `person`/`organization` for one Pipeline and `gene`/`drug` for
another with no code change — the S4 acceptance criterion ("label set is
configurable per Pipeline; no label set hardcoded in core"). This is also what lets
the GLiNER Provider (a genuinely zero-shot model) and the gazetteer Provider share
one seam: both take their vocabulary from config.

## The deterministic extractor ships in core; the learned GLiNER Provider is its own package

Two reference Providers, split on the ADR-0007/0016 line:

* **`entity.gazetteer`** ships in `latence-core` (like the S3 Chunker/Screeners): a
  pure-Python, dependency-free, **deterministic** zero-shot gazetteer/regex
  extractor. It is what makes the end-to-end demo run on first `git clone` with no
  model download, and what makes the S4 tests hash-stable for the Baseline bar. It
  is not a weaker reinvention of a learned NER model — it is the CPU-zero-dep
  reference, and it is genuinely zero-shot (labels + recognisers from config).

* **`entity.gliner`** is the separate package `latence-ner-gliner`: a learned
  zero-shot Provider wrapping a GLiNER checkpoint via plain `transformers`/ONNX (CPU
  path, no vLLM). Its heavy stack (`gliner` → torch/transformers) is isolated in its
  own package so `pip install latence-core` stays lean (ADR-0016) and the laptop-
  first promise holds (ADR-0007). Swapping to it is a one-line provider/config
  change, no pipeline change. Licensing verified permissive both ways (ADR-0012,
  2026-07-06): the `gliner` library is Apache-2.0 (code) and the default checkpoint
  `urchade/gliner_multi-v2.1` is Apache-2.0 (weights) — v2.1 relicensed from the
  earlier CC-BY-NC-4.0, so the commercial-safe checkpoint is the default and a
  CC-BY-NC one is a deliberate opt-in, never shipped as the default.

Because the learned stack must stay out of the deterministic/offline dev+test env,
`latence-ner-gliner` is a workspace member but is **not** pinned into the dev
meta-project's dependencies; CI installs only its source (`--no-deps`) so its `src`
is strict-type-checked and its Provider tests (which monkeypatch the `gliner`
dependency with a fake to exercise the offset math offline) collect and run without
torch. Rejected: hardcoding GLiNER as THE reference (forces torch into first-run,
contradicts ADR-0007) and shipping only a rule stub (undercuts the NER quality
story — the learned Provider must exist behind the same seam, per ADR-0013's
empirical-choice principle, here applied to Entity Extraction).
