# R1 — ADR-0053 SPLADE composition is never applied at index time (asymmetric query/document sparse space)

- **Severity:** major
- **Dimension:** spec / AC / ADR compliance
- **Branch:** feat/retrieval-tooling
- **Tickets:** #160 (02-index-time-signal-seam), #162 (04-splade-composition), ADR-0053
- **Status:** fixed

## What is wrong

ADR-0053 specifies the sparse-retrieval signal as a *structured composition*:

    sparse = model_weight · SPLADE(text)  ⊕  entity_boost · confidence-gated entity_terms
                                          ⊕  metadata_boost · selected metadata_terms

and it is explicit that this is *one composition applied to two clocks* — the **same**
composition must run at index time (over each chunk) and at query time (over the query
string) so the query and corpus land in the same sparse space.

At index time this never happens. The Export → `SparseEmbedder` seam is text-only
(`packages/latence-core/src/latence_core/capability.py` `SparseEmbedder.encode_sparse(texts)`),
and `packages/latence-core/src/latence_core/stages/export.py` (`_attach_signals`) calls
`self._sparse.encode_sparse(content_texts)`. `SpladeCompositionEmbedder.encode_sparse`
(`packages/latence-splade/src/latence_splade/provider.py`) composes the **model leg only** on
that text-only path — no entity injection, no metadata injection. So the emitted
`sparse_indices` / `sparse_values` corpus columns are plain `SPLADE(text)`.

The query side, however, DOES inject: `SpladeQueryEncoder.encode`
(`packages/latence-retrieval/src/latence_retrieval/sparse.py`) calls `encode_composed` with
the query's entities + metadata, emitting the boosted entity/metadata term ids.

The result is an **asymmetric query/document sparse space**: injected metadata term ids and
boosted entity term ids exist only in the query vector. A hybrid/sparse search on an injected
metadata term matches zero documents (no corpus vector ever carried it), and the entity boost
is asymmetric. The ADR-0053 differentiator is inert at index time — this defeats #162 AC-2
("the sparse vector = model term-weights unioned with confidence-gated entity surface terms and
selected metadata terms") on the corpus side, and no index-time/export test asserted the
injection.

## Failure scenario

Wire `SpladeCompositionEmbedder` as the Export `sparse_embedder` with
`metadata_fields=["category"]`, `entity_boost=5`, `metadata_boost=7`. A chunk
"the plaintiff filed a motion" with a canonical entity "plaintiff" and `category="litigation"`
emits, on the old code, `SPLADE("the plaintiff filed a motion")` with NO entity boost and NO
"litigation" metadata term. At query time `SpladeQueryEncoder.encode(same text,
entities=[("plaintiff",0.99)], metadata={"category":"litigation"})` emits the model leg PLUS a
boosted "plaintiff" id AND a "litigation" metadata id — ids that exist only in the query
vector. A sparse search on the metadata id matches zero documents.

## What to build

Make the index-time seam carry the chunk's structured signals so the emitted corpus vector is
the ADR-0053 composition, symmetric with the query side:

1. A **core-owned** composing seam (no `latence-core` → `latence-splade` dependency): a
   `SparseSignals` per-record carrier (confidence-bearing entity surfaces + selected metadata)
   and a `ComposingSparseEmbedder` Protocol with `encode_sparse_composed(texts, signals)`.
2. `SpladeCompositionEmbedder` implements `encode_sparse_composed` by building a `SpladeInput`
   per `(text, signals)` and running the SAME `compose` the query side runs.
3. Export detects the composing Capability and passes each row's signals — entity surfaces from
   the chunk's `ContextHeader` (already confidence-gated by `context_enrichment`, offered at
   confidence 1.0) and selected metadata from the row's `Classification`. The Provider's own
   redaction gate (`redact=True` default) drops any entity surface not present in the redacted
   corpus text, so no masked PII surface is re-injected (REDACTION-BY-DEFAULT preserved).

An embedder that does NOT implement the composing Capability (the deterministic `sparse.hashing`
reference) is driven text-only, byte-identically to today.

## Acceptance criteria

- [x] When a composing `SparseEmbedder` is wired, the emitted corpus `sparse_indices`/`sparse_values`
      carry the boosted confidence-gated entity terms and the selected metadata terms (not just the
      model leg).
- [x] The index-time composed vector equals the query-side composition for the same
      `(text, entities, metadata)` — one composition, two clocks.
- [x] A masked/redacted entity surface is not injected at index time (REDACTION-BY-DEFAULT).
- [x] With no sparse embedder wired, or with a non-composing reference (`sparse.hashing`) wired,
      the export is byte-identical to today (ADR-0017/0049).
- [x] A non-vacuous regression test fails on the old code (text-only `encode_sparse`) and passes on
      the fix; the composition is asserted on the *observable emitted vector*, not on a config flag.
