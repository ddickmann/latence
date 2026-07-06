# A RelationMention is a first-class inter-Stage Record; a fused Stage emits both carriers in one pass, threaded by a mixed-carrier checkpoint

S5 adds **Relation Extraction** (CONTEXT): the Stage that finds typed relations
between entity mentions within a document. Three design questions had non-obvious
answers; they extend ADR-0013 (the empirical-choice + fused-Capability decision)
with the concrete contract and Runner mechanics, mirroring how ADR-0022 did for S4.

## A RelationMention is a first-class inter-Stage Record, not a field on a mention

A relation could be a list nested inside each `EntityMention` (its outgoing edges).
It is not. `RelationMention` is its own versioned core contract (`SCHEMA_VERSION` →
5), a subclass of `Record` — so it carries Provenance and Classification like every
other inter-Stage carrier, is threaded through the DAG as a Stage output,
checkpointed/resumed by the Runner uniformly, and exported to the corpus directly.

**Why.** Graph Assembly (S8) builds *edges* from relations and attaches per-edge
Evidence at the relation's offsets; Disambiguation (S8) relates *canonical* entities
via the mention-level relations. Making the relation the record (with its own
covering-span Provenance and head/tail mention refs) means the downstream KG Stages
receive the exact contract they need — a directed, typed, offset-anchored edge — with
no re-derivation from a mention-nested list, and keeps the Runner's
one-output-list-per-Stage invariant intact. Rejected: relations-as-fields-on-mentions
(couples every consumer to the mention shape, hides the relation's own Provenance,
and cannot express a relation whose endpoints live in different chunks).

## A relation's Provenance span covers BOTH endpoints and re-resolves its own page

An `EntityMention` anchors one span; a relation anchors *two*. The relation's
Provenance char span is the covering `[min(head.start, tail.start),
max(head.end, tail.end))` in the parent document's assembled markdown, and its page
span is re-resolved from that covering span through the parent's page map via
`PageOffsetIndex` — so a relation whose endpoints straddle a page boundary records
*both* pages (the S5 round-trip: relation → offset → original page), degrading to the
union of the two mentions' own pages only when no page map is available. The
`RelationMention` contract *enforces* the span (no char span or no page span ⇒
validation failure) and *enforces distinct head/tail* (no self-relation), so a
Provider that forgets to resolve the endpoint offsets, or hallucinates a
self-reference, cannot emit an un-resolvable record. Relation Extraction reconstructs
the document text from the chunks (each chunk carries its own document-level span,
S3) so a text-based Provider can test a trigger phrase against the *real*
between-mention text at document offsets — the same coordinate system the covering
span uses.

## A fused Stage emits BOTH carriers in one pass; the Runner threads them via a mixed-carrier checkpoint

ADR-0013 requires the Pipeline to permit one Provider fulfilling both Entity
Extraction and Relation Extraction (gliner-relex, the custom span-predictor) — the
Pipeline must not force one Provider per Stage. Concretely:

* A new Capability `FusedEntityRelationExtractor` returns a `FusedExtraction`
  (`mentions` + `relations`) from one pass over the chunks, and a new
  `CapabilityKind.FUSED_ENTITY_RELATION` binds a Stage to it. The Runner runs the
  fused Stage **instead of** a separate Entity-Extraction Stage — the S5 "routes
  correctly without double-extracting" requirement — and writes both carriers into
  the *one* Stage output list (`[*mentions, *relations]`), so both flow downstream
  from the single Stage.
* That output list is a **mixed carrier** (EntityMention + RelationMention). Its
  checkpoint therefore cannot be read back with a single record type. The Runner
  signals a mixed carrier by returning `Record` from `_output_type`/`_export_carrier`
  and reads the checkpoint **polymorphically**, discriminating each line by a field
  unique to the relation contract (`head_mention_id` present ⇒ RelationMention). Both
  are `Record`s, so a resumed fused run re-materialises both without a second Stage
  and without a discriminator field baked into the wire format. The Quality Report's
  entity/relation sections gather from the fused Stage as well as the dedicated ones,
  filtering each list to its own carrier, so a fused run reports both.
* A **separate** Relation-Extraction Stage (the in-core `relation.pattern`, the LLM
  Provider) is the alternative shape: it is multi-input, wiring BOTH a Chunk (text +
  page map) parent and an Entity-Extraction (the mentions) parent via `depends_on`.
  The Runner partitions the gathered inputs by carrier and raises a clear typed error
  if either side is missing — a forgotten edge fails loudly rather than silently
  extracting nothing.

Rejected: a per-line `record_kind` discriminator baked into the contract (redundant
with the type's own fields, and noise in every serialized record); two separate
checkpoints for one fused Stage (breaks the one-output-list-per-Stage invariant and
the atomic checkpoint discipline); and forcing the fused model behind two Stages that
each re-run it (the double-extraction ADR-0013 explicitly forbids).

## Providers shipped (empirical choice, ADR-0013; licenses verified 2026-07-06)

* **`relation.pattern`** ships in `latence-core`: a pure-Python, dependency-free,
  deterministic zero-shot **trigger-phrase** relation extractor (head/tail entity
  labels + trigger phrases from config gate and type each relation). Not a bare
  co-occurrence stub (ADR-0013 rejects that) — the trigger both types and justifies
  the edge, so the demo produces typed KG edges on first `git clone`, hash-stable for
  the Baseline bar.
* **`relation.llm`** (`latence-relation-llm`): an LLM extractor over any
  OpenAI-compatible endpoint incl. local Ollama (framework code Apache-2.0, `openai`
  client Apache-2.0, **no bundled weights**; a hallucinated mention index / unknown
  label / self-relation is dropped, never emitted).
* **`relation.gliner_relex`** (`latence-relation-gliner`): the **fused** Provider over
  `knowledgator/gliner-relex-multi-v1.0` (weights Apache-2.0, `gliner` code
  Apache-2.0), joint zero-shot NER+RE in one pass. Both gliner packages and the llm
  one keep their heavy/endpoint deps out of the thin core (ADR-0016): workspace
  members, type-checked from source, exercised with the dep monkeypatched by a fake
  in tests, never installed into the deterministic dev/test env.
