# Entity Extraction emits TRUE original offsets by carrying the chunk's strip offset map forward

S4 Entity Extraction (and the fused NER+RE Provider) emitted mention offsets in a
**hybrid coordinate system**: `doc_offset = chunk.provenance.char_start + local_offset`,
where `local_offset` indexes the chunk's markup-**stripped** `content` while
`char_start` is in **original** parsed-markdown coordinates. Chunk stripping
(`chunking.strip_markup`, 18 patterns) only ever *deletes* characters, so once any
markup is stripped **before** a mention within its chunk, the recorded offset is a
**lower bound**, not the true offset — and the mention's own page (resolved through
`PageOffsetIndex` against the true page map) can be wrong. ADR-0022 asserted this shift
yields original offsets; that claim was false for markup-containing chunks. It went
unnoticed because every S4 fixture was markup-free plain text (corrected in ADR-0022).

The same latent bug lived in the fused `latence-relation-gliner` Provider
(`chunk_char_start + local_*` for both mentions and, transitively, relation covering
spans) and in the two **standalone learned/cloud** EntityExtractor Providers that also
emit chunk-local model offsets: `entity.gliner` (`latence-ner-gliner`, the learned GLiNER
CPU path) and `entity.endpoint` (`latence-ner-endpoint`, the OpenAI-compatible NER
performance path) — the one-config swaps behind the same S4 seam. This ADR fixes the
**root**, so **all four** EntityMention-emitting Providers (the in-core `entity.gazetteer`,
the two standalone learned/cloud NER Providers, and the fused `relation-gliner`) resolve
their model/match offsets through `OffsetIndex(chunk.offset_map)`, and every downstream
consumer inherits correct offsets instead of each re-deriving them; the S5-relation
coordinate concern (issue #48 called it out explicitly) has the same root and is subsumed
here.

## Grill: three ways to recover the true offset

`entity.py` holds only the `ChunkRecord` (stripped `content` + the outer
`[char_start, char_end)`), not the per-character strip map — the chunking engine builds
that map (`_StripResult.mapping`) and then discards it. Three options were weighed
against CONTEXT (Chunk "preserves offsets") and ADR-0020/0021 (offsets index the
original, never the local text):

- **(c) Re-strip the chunk's original slice to rebuild the map.** *Rejected — fragile.*
  The 18 patterns are line-anchored / multiline / context-sensitive (headings, fenced
  code, table-separator rows); a chunk's slice does not reproduce `chunk.content` at its
  boundaries, so a re-strip can disagree with what Chunk actually emitted. It also
  re-derives, lossily, exactly what the engine already computed once.

- **(b) Thread the parsed `DocumentRecord` into S4 and relocate each surface**
  (`text.find(surface, lower, upper)`). *Rejected as the root fix.* It widens the
  `EntityExtractor` seam (the extractor would need the whole document to find its own
  output's position), and it is a **search, not an exact map**: a surface whose own span
  contains stripped markup (e.g. `Acme **Corp**`) is not found and silently degrades to
  the lower bound. Wrong for S4 mentions, which are the offset source of truth every
  later Stage trusts.

- **(a) Carry the stripped-`content`→original offset map forward on `ChunkRecord`.**
  *Chosen.* It completes the contract the engine already promises (chunking.py part 3,
  "offset preservation") at the layer that owns it, makes the mention offset **exact**
  (not a search), and needs no seam change — the extractor still sees only chunks, now
  chunks that carry a faithful map. Precedent: `ChunkRecord` already carried the parent
  `page_map` forward (ADR-0022) for exactly this shape of problem (sub-chunk offset → its
  own page); this is the same move one level finer (sub-chunk offset → its own original
  offset → its own page). *(That precedent's carriage was replaced twice and no longer
  exists: v18 narrowed it to a document's first chunk, and schema v19 removed the whole-map
  carriage entirely in favour of a per-chunk `page_slice` — see ADR-0022's C1 amendment. The
  `offset_map` decided here was always genuinely per chunk: chunk-scoped state, not
  document-scoped, so neither change touched it — and v19 makes the two fields the same
  shape, each describing only its own chunk.)*

## Decision: a compact run-length offset map + a resolver, mirroring PageMap/PageOffsetIndex

Because stripping only deletes spans, a chunk's surviving characters form contiguous
runs of the original; the local→original map is piecewise slope-1 with a jump at each
stripped gap. We store one breakpoint per run — `OffsetSegment(local_start, doc_start)` —
in an `OffsetMap` contract (`segments` + `length`), and resolve through `OffsetIndex`
(`resolve` / `resolve_span`), a live index built once per chunk exactly as
`PageOffsetIndex` is built from `PageMap`. Properties:

- A **markup-free** chunk has a single `(0, char_start)` segment, so `resolve(k) ==
  char_start + k` — the plain-text path is byte-identical to before (no fixture churn,
  determinism preserved).
- Weight is proportional to the number of stripped gaps in the chunk (typically a
  handful), far smaller than the whole-document `page_map`, which at the time rode every
  chunk (that comparison no longer holds: v18 narrowed the carriage to a document's first
  chunk and schema v19 replaced it with a per-chunk `page_slice` of ~2 spans — ADR-0022's
  C1 amendment — so both fields are now chunk-scoped and comparably small).
- The engine stays contract-free: `chunking.Chunk` carries the raw `(local_start,
  doc_start)` tuples; the reference `MarkdownChunker` Provider wraps them in the
  `OffsetMap` contract (the same engine/Provider split the module already uses).
- `resolve_span` uses the exclusive-end convention the engine uses for a chunk's own
  `char_end` (`resolve(local_end - 1) + 1`), so a mention span whose interior contained
  stripped markup still covers the full original extent.

The field is **optional and additive**: a chunk without an offset map (a pre-v11 record,
or a document with no map) degrades to the old `char_start + local` shift. This is a
breaking contract change only in the sense that mention offsets now *mean* the true
original offset; **`SCHEMA_VERSION` is bumped 10 → 11** and the change is recorded in the
contract history.

Downstream inheritance (the point of fixing at source): Relation Extraction's covering
spans (min/max of its endpoints' now-correct offsets), Disambiguation Evidence offsets,
Graph Assembly, and the fused `relation-gliner` all become correct with no per-Stage
offset code. The mention `text` field remains the clean **stripped** surface; only the
Provenance span is widened to the exact original extent.

## Scope note

The in-core `PatternRelationExtractor` reconstructs per-document text by writing stripped
chunk `content` at original offsets (`_document_texts`) to test trigger phrases *between*
mentions — a separate latent markup concern that this fix does **not** address (chunk-level
offsets were always correct; only sub-chunk *mention* offsets were the lower bound). This
ADR's change leaves that path unaffected and only improves that extractor's mention-derived
covering spans (which are now min/max over corrected endpoint offsets). Verify Redaction
scans the original `DocumentRecord.content` directly and is unaffected — it never consumed
the chunk-local offset, so it stays consistent with the now-corrected mention offsets.
