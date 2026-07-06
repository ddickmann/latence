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

## A mention's offsets index the ORIGINAL document, resolved by shifting the chunk span

An extractor sees a *chunk's* text and returns *chunk-local* offsets. But a mention
must resolve back to its source **page** via `PageOffsetIndex`, which indexes the
**parent document's** assembled markdown (ADR-0020/0021). So the Provider shifts
each chunk-local match by the chunk's own `provenance.char_start` (the span the
chunk occupies in the parent document, preserved losslessly by Chunk, ADR-0021):
`doc_offset = chunk.provenance.char_start + local_offset`. The mention's **page** is
then re-resolved from *that* document offset through `PageOffsetIndex` — the same
resolver the chunk Provider uses (ADR-0021) — **not** inherited wholesale from the
chunk's `page_start`/`page_end`. A default chunk budget (512 tokens) routinely spans
several short source pages, so a mention on a later page of a multi-page chunk must
cite *its own* page; inheriting the chunk's first page would violate the S4 core
requirement "Provenance resolving to the correct source page". To make this
resolvable at the mention Stage, Chunk now carries the parent document's `page_map`
forward on each `ChunkRecord` (optional field; absent ⇒ the mention degrades to the
inherited chunk range). A boundary-straddling mention resolves `page_start` from its
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
