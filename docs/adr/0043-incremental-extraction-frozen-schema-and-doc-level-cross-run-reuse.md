# Incremental extraction: a frozen induced schema + doc-level cross-run reuse (the O(changed) delta)

**Status: accepted** — W17-incremental-extraction (resolves the issue-#42 boundary). Makes a delta's
EXTRACTION COMPUTE incremental: an UNCHANGED document skips its doc-level Stages
(parse/chunk/extract/redact) and REUSES its committed records; only NEW/CHANGED documents are
extracted, and the corpus-level Stages run over the UNION. Builds on ADR-0018 (the affected-set
delta / `DeltaProcessor` / `CorpusStore` / WAL Corpus Version transaction), ADR-0003 (the Runner's
DAG + per-Stage checkpoints), ADR-0033/#69 (the streamed corpus-level accumulator inputs this
overlays), ADR-0038 (the induced-labels contract + `config ∪ induced` seam, and `granularity:
corpus`), ADR-0029 (the retract/purge erasure model this extends to the new residue surface), and
ADR-0012/0007 (content-addressed document ids, CPU-first determinism). No `SCHEMA_VERSION` bump —
the change is additive files under the Corpus Version + additive `DeltaChurn` fields.

## Context

Before W17 the `DeltaRunner` re-ran the whole pipeline over the whole current source on every delta
(a fresh `run_id`, no cross-run checkpoint reuse), then the `DeltaProcessor` scoped only the
*committed corpus-level record set* to the affected neighborhood. So the *stored* corpus honoured
ADR-0018's "recompute only the affected set", but the *extraction cost* — parse, chunk, the GPU
entity/relation extractors, the PII redactor — was still O(corpus). Adding one document to a
1000-document corpus re-extracted all 1001. That was the honest boundary called out in the
`delta_runner` module docstring and issue #42.

Two things make naive skipping unsound, and both must be handled:

1. **The corpus-level coupling.** `disambiguation` must resolve the *union* of all mentions — a
   bridging document merges with an existing cluster only if both are disambiguated together — so we
   cannot just extract the new document and disambiguate it in isolation. The `DeltaProcessor`
   already relies on `current_records` being the fully-reconciled union.
2. **The schema coupling.** With `schema_induction` at `granularity: corpus` (ADR-0038), ONE shared
   vocabulary is induced from ALL chunks. If adding a document shifted that vocabulary, the
   unchanged documents — extracted under the OLD labels — would be stale. Reuse is only valid if the
   labels are stable across the delta.

## Decision

### 1. Freeze the induced corpus schema in the Corpus Version

On the first run (or an explicit `--reinduce`), induction runs and the resulting corpus schema
(`InducedLabels`: entity/relation/pii types) is PERSISTED in the committed Corpus Version
(`_latence/corpus/wal/v<N>/schema.json`, staged under the same WAL transaction). On a subsequent
`apply_delta` the frozen schema is reused by default: the `DeltaRunner` sets the `schema_induction`
Stage's `frozen_induced_labels` config, and `LLMLabelInducer.induce` STAMPS that single schema on
every chunk and makes NO LLM call. So the freshly-extracted documents use the SAME labels as the
reused ones — extraction labels are stable across the delta, so an unchanged document's extraction
stays valid. `--reinduce` omits the freeze: induction re-runs over the full current corpus and every
document is re-extracted (the correct-by-construction O(corpus) fallback). Only `granularity: corpus`
is frozen (its single shared vocabulary is the coupling); per-document/`chunk` granularity rides in
each document's reused chunk records with no corpus-level freeze.

### 2. Cross-run doc-level reuse

Each committed Version persists every LIVE document's doc-level Stage outputs
(parse→`DocumentRecord`, chunk/redact→`ChunkRecord`, entity→`EntityMention`,
relation→`RelationMention`) in `wal/v<N>/doclevel.jsonl`, keyed by the content-addressed
`provenance.document_id` (sha256 of the bytes). On `apply_delta`:

* the current source document ids are computed (a cheap read-and-hash of the source bytes — NOT
  extraction); a current id that matches a committed id is byte-identical, i.e. UNCHANGED;
* the unchanged documents are EXCLUDED at Source (so their doc-level Stages never run) and their
  persisted records are handed to the Runner as a **reuse overlay**;
* NEW/CHANGED documents run the doc-level Stages as usual.

`LocalRunner.run(reuse=…)` indexes the overlay by carrier type and streams it into the corpus-level
accumulators (`_stream_inputs_of_type` / `_stream_all_records_of_type` / the `records_in` count and
wiring checks) and a doc-level Export ALONGSIDE the freshly-extracted records — so those Stages see
the whole corpus (`reused ∪ fresh`). The overlay is NEVER fed to a doc-level Stage (those gather via
the overlay-blind `_gather_inputs`), so a reused document is never re-parsed or re-extracted. The
`DeltaProcessor` affected-set recompute is UNCHANGED — it consumes the same union `current_records`
it always did.

### 3. Honest boundary shift + telemetry

`DeltaChurn` gains `documents_reused` / `documents_extracted` (the perf witness, surfaced in the
Quality Report). The `delta_runner` module docstring now states: extraction is O(changed documents),
not O(corpus), when the schema is frozen — with the ONE caveat that `--reinduce` (or a first run / a
schema change) re-extracts all, by design.

## Why this is correct (byte-identical), not just cheaper

The committed corpus is byte-identical to a full from-scratch run of the whole source, by
construction:

* **Content-addressed ids.** Every doc-level record id derives from the document bytes (+ chunk/
  mention index); a reused record is byte-identical to re-extracting that document under the same
  frozen schema. Doc-level extraction is per-document, so a document's records never depend on
  another document's presence.
* **Order-independent corpus-level fold.** `disambiguation.cascade` folds mentions in sorted
  `mention_id` order (Union-Find over a sorted vocabulary), so the union `{fresh} ∪ {reused}`
  produces byte-identical clusters regardless of the overlay's interleave position; graph assembly is
  a deterministic function of those clusters. Hence `current_records` is byte-identical to a full
  run, and the `DeltaProcessor`'s carry-forward + affected-set splice is unchanged.

This is proven end-to-end: adding 1 document to a committed 3-document corpus yields committed
records for the 3 unchanged documents byte-identical to a from-scratch run of all 4, while the run's
doc-level checkpoints hold ONLY the added document (`documents_reused == 3`).

## The walls we checked (and cleared)

Both risks flagged for this work are cleared, not hacked around:

* *"The checkpoint can't key by document_id."* Every doc-level carrier already carries
  `provenance.document_id`; the reuse store keys on it directly.
* *"Corpus-level Stages can't consume a mixed reused+fresh stream without a rewrite."* They can —
  the overlay is additive and the accumulators are order-independent (above), so the mixed stream is
  consumed by the exact same code path, byte-identically.

## Scope boundary (deliberately deferred)

Reuse falls back to full extraction (correct by construction) for a pipeline containing a
`context_enrichment` Stage: a chunk's `context_header` summarizes the MUTABLE assembled graph, so a
reused document's enriched output can go stale when a delta changes the graph, and its enriched
chunks would collide with the doc-level Export overlay. Rather than a fragile partial reuse that
could corrupt the exported corpus, `DeltaRunner._reuse_supported` disables reuse for such a pipeline.
Combining reuse with enrichment (re-enriching reused chunks over the current graph) is a clean
follow-up. The retract/purge path also re-extracts the survivors from scratch (a split-on-delete must
re-resolve them) — its reuse is out of scope here; it still refreshes the reuse store.

## Consequences

* A delta over a large stable corpus is now dominated by the changed documents' extraction, not the
  whole corpus — the headline #42 win, with the churn reporting `documents_reused`/`_extracted`.
* **New residue surface:** `doclevel.jsonl` holds raw chunk/mention text, so a GDPR Purge now sweeps
  it too (by `document_id` reach, exactly like the corpus-level log, for committed AND abandoned WAL
  dirs) — nothing raw survives a purge.
* Determinism, idempotency (the `CorpusStore` fingerprint is over corpus-level records only, so the
  doc-level store never perturbs it), and WAL crash-resume are preserved.
* The reuse store costs storage proportional to the live corpus's doc-level records — the standard
  space-for-compute trade an incremental engine makes.
