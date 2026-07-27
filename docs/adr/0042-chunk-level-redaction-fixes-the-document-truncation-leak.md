# Redaction is a chunk→chunk transform (fixes the document-level truncation PII leak)

**Status: accepted** — W16-chunk-level-redaction. Changes the `PIIDetector` Capability from
DOCUMENT-level (`redact(documents) -> Iterator[RedactionRecord]`) to CHUNK-level
(`redact(chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]`): the Redactor now consumes the run's
chunks and yields the SAME chunks with the additive `ChunkRecord.masked_content` + `pii_spans` +
`redaction_disabled_for_sensitive` fields populated. Bumps `SCHEMA_VERSION` 15→16. Builds on ADR-0004
(Capability protocols + Provider plugins), ADR-0007 (CPU-first default), ADR-0016 (thin core, zero
extra deps), ADR-0017 (Export writes the RAG-ready corpus), ADR-0022/0031 (chunk `content` is
markup-stripped with a carried `offset_map` from stripped→original coordinates), ADR-0024 (the S6
Redaction policy/masking machinery), ADR-0036 (Provider ecosystem: profile, conformance, bake-off),
ADR-0038 (the additive-annotation record precedent — `induced_labels`, and the `config ∪ induced` label
seam), ADR-0039 (the chunk→chunk transform shape — `context_enrichment` — and the additive
`context_header` field this mirrors).

## Context

Redaction was DOCUMENT-level: a Provider scanned a whole parsed `DocumentRecord`'s assembled markdown
and emitted one `RedactionRecord` per document. The learned PII Providers (`redaction.gliner_pii`,
`redaction.gliner2`, `redaction.presidio`) run a fixed-window encoder (mdeberta-v3, **768 tokens**).
Feeding a whole document into that window means **any PII past token 768 is silently truncated by the
model and leaks unmasked into the corpus** — a real PII-safety bug. This is not hypothetical: the
provider `max_len` guard (W12) only *bounds* the window explicitly; it does not stop the truncation, it
documents it.

The bug is also an **architectural inconsistency**. After the Chunk Stage, *everything else the pipeline
learns is chunk-level* — Content Screening, Schema Induction, Entity/Relation Extraction, Context
Enrichment all consume chunks (each ≤ the model window). Redaction was the lone Stage still reaching
back to the whole document, which is exactly why it hit the window ceiling the chunker was built to
respect.

The old design justified scanning the document (not chunks) on a coordinate-correctness argument: a
chunk's `content` is markup-stripped with offsets remapped, so `len(content) != char_end - char_start`,
and a naive reconstruction from chunks would fabricate a broken coordinate system. But that argument
predates the S4 `offset_map` (ADR-0031): a chunk now carries a run-length map from its stripped
`content` offsets back to the original markdown, so a chunk-local span **does** resolve to its true
original source page. The coordinate objection is obsolete.

The fix must preserve the invariants Redaction is built on:

1. **No truncation.** Each unit scanned must fit the model window, so all PII is seen.
2. **Extraction still sees full entities.** Entity/Relation Extraction reads the chunk `content`; that
   text must stay UNMASKED (masking it would blind extraction to the very entities it exists to find).
3. **The RAG corpus is PII-safe.** The exported corpus text must be the masked variant — the
   stack-validate rag-scan reads every exported cell.
4. **Counts-only discipline.** No raw PII value in any report/log/span (S6 AC).
5. **Genuinely additive + byte-identical corpus.** A stack with no Redaction Stage is unchanged.
6. **Don't silently break the world.** The document-level `RedactionRecord` path stays available,
   deprecated.

## Decisions

### 1. The `PIIDetector` Capability becomes a chunk→chunk transform

`redact(chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]`. A Provider consumes the run's chunks
(each ≤ the model window) and yields the SAME chunks, 1:1 and order-stable, with the redaction fields
populated. This is structurally identical to the existing `ContentScreener` / `LabelInducer` /
`ContextEnricher` chunk→chunk transforms (ADR-0038/0039), so the Runner threads it with zero new carrier
machinery: `redaction`'s `_output_type` and `_export_carrier` are `ChunkRecord`, exactly like
`context_enrichment`. The Redaction Stage now `depends_on` the Chunk (or Content Screening / Schema
Induction) Stage, **not** Parse.

### 2. Three additive `ChunkRecord` fields; the clean `content` stays UNMASKED

* `masked_content: str | None` — the PII-handled variant of the chunk text (every non-`TAG` PII span
  replaced by its placeholder). Default `None` ⇒ no Redaction Stage ran.
* `pii_spans: list[PIISpan]` — the chunk-local detected spans. `char_start`/`char_end` are offsets into
  the chunk's `content` (the coordinate system `masked_content` is masked in); `page_start`/`page_end`
  resolve to the ORIGINAL source page through the chunk's `offset_map` and the document-scoped
  `PageIndexResolver` (the ADR-0031 round trip; since schema v18 the `page_map` itself rides a
  document's first chunk only — ADR-0022's C1 amendment). Counts-only: a span never stores the raw value.
* `redaction_disabled_for_sensitive: bool` — the H-C1 §4 no-op floor, carried per chunk.

The chunk's `content` is **byte-unchanged** — extraction (upstream, on the same chunk stream) still sees
full entities, and any later audit/re-read sees the real text. `masked_content` is the PII-safe variant
the Export materializes. All three fields default to nothing (`None`/`[]`/`False`), mirroring the
`risk_markers` / `induced_labels` / `context_header` annotation precedent, so prior-version JSON still
validates and a run with no Redaction Stage is byte-identical for its corpus text.

### 3. Extract-on-unmasked, mask-for-RAG — the stage ordering

The blessed chunk stream is:

```
chunk → content_screen → [induce] → extract (reads UNMASKED content)
                                   → redact (chunk → masked_content)
                                   → [enrich (context_header on the masked chunk)]
                                   → export_corpus (masked + header)
```

`extract`'s mentions feed `disambiguate → graph`. **The KG is built from the UNMASKED chunks** (so it
keeps the real entities), while **the RAG corpus text is masked**. Because Redaction now sets
`masked_content` on the chunk (rather than emitting a separate `RedactionRecord`), the enriched chunk
can carry BOTH `masked_content` (PII-safe) AND `context_header` (KG coherence) — so the corpus record is
masked AND enriched. This resolves the pre-W16 constraint (documented in `stacks/context-enrichment.yaml`)
where the RAG export had to bypass enrichment to avoid leaking the un-redacted chunk `content`.

### 4. Export materializes `masked_content` as the corpus text

At the single dump-through point, the Export substitutes `content := masked_content` when a chunk carries
a non-`None` `masked_content` — so the exported `content` column, the JSONL row, and the embedding input
are all the PII-safe variant; the raw `content` never reaches the corpus (the rag-scan reads every cell).
The W13 `context_header` still prepends to the EMBEDDING input only. A chunk with no `masked_content`
(no Redaction Stage) exports its `content` unchanged — byte-identical to today.

### 5. Shared chunk-redaction machinery; a Provider owns only detection

`latence_core.redaction_policy` gains `plan_chunk_redaction` (policy resolution + the §4 no-op floor +
the chunk `offset_map` resolver + the call-scoped `PageIndexResolver`) and `finalize_redacted_chunk` (build the chunk-local spans
with original-resolved pages, mask right-to-left, `model_copy` the chunk). All four Providers
(`redaction.hybrid_rule`, `.gliner_pii`, `.gliner2`, `.presidio`) share this tail and differ only in
detection — the gliner-family keep their `quantize`/`compile`/`max_len` guard and label-set batching, and
now read `config ∪ induced` PII labels **directly off each `chunk.induced_labels`** (no
chunk→document bridge — the induced schema already rides the chunk, ADR-0038).

### 6. The document-level path is retained but DEPRECATED

The `RedactionRecord` contract stays (prior-version JSON, and any external consumer). The in-core
Provider keeps a deprecated `redact_documents(documents) -> Iterator[RedactionRecord]` secondary method;
the Runner's `redaction` Stage and every blessed stack now flow chunks. The Quality Report's
`RedactionQuality` rolls up from the redacted chunks (per-type/per-action span counts; documents-redacted
/ documents-with-PII / disabled-for-sensitive deduped to the DOCUMENT level so the report reads the same
as before), and still tallies a `RedactionRecord` stream for the deprecated path.

## Consequences

* **The truncation leak is fixed.** Each chunk is ≤ the model window, so PII anywhere in the document is
  scanned in full and masked. A control test proves the old doc-level path misses PII past the window
  while the chunk-level path catches it.
* **Redaction is now consistent with the post-chunk architecture** — one chunk→chunk transform among
  peers, threaded by the Runner with no bespoke carrier.
* **The corpus is masked AND enriched** — the export substitution + the enrich-after-redact ordering let
  one chunk carry both, which the pre-W16 design could not.
* **Byte-identical when off** — no Redaction Stage ⇒ `masked_content` `None` ⇒ the export falls back to
  `content`; the corpus text is unchanged.
* **A per-page-map cost** on very large chunks (building the `OffsetIndex`/`PageOffsetIndex` per chunk),
  bounded and negligible vs the model forward pass.
* **Follow-ups:** the deprecated `RedactionRecord`/`redact_documents` path can be removed once no
  consumer depends on it; and a future Provider could fold Redaction into the fused extraction pass
  (one encoder forward for NER+RE+PII) now that all three read the same chunk.
