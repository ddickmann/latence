# Incremental ingest: exact corpus-level recompute over an order-preserving reuse union (enrichment pipelines included)

**Status: accepted** — wayfinder-enterprise-serving ticket 03. Makes document churn
(add/update/remove against an existing run's store) a first-class, *verified* capability: the
increment path now covers the flagship enrichment profile, and the incremental result is proven
equivalent to a full rebuild of the same final corpus on real data (the s3 `multihop_rag` export,
609 documents, zero LLM calls). Builds on ADR-0018 (affected-set delta / CorpusStore / WAL
Version transaction), ADR-0043 (frozen schema + doc-level cross-run reuse — the reuse overlay this
ADR re-orders and extends), ADR-0054 (seeded canonical type vocabulary), ADR-0039 (context
enrichment), ADR-0057 (hyperedges), ADR-0017 (export-attached embeddings), ADR-0048/#164 (BM25
artifact). No `SCHEMA_VERSION` bump — no contract changes; a non-delta run is byte-identical.

## Context

The delta machinery already made *extraction* incremental (ADR-0043): an unchanged document's
doc-level records (parse/chunk/extract/redact) are reused from the committed Corpus Version, only
new/changed documents run the doc-level Stages, and the corpus-level Stages run over the union via
the Runner's reuse overlay. Three gaps kept churn from being an enterprise answer:

1. **The flagship profile was excluded.** `DeltaRunner._reuse_supported` refused cross-run reuse
   for any pipeline containing a `context_enrichment` Stage — precisely the enterprise-SOTA s3
   profile — so on that stack every "incremental" delta silently degenerated to a full O(corpus)
   re-extraction (and, on induced corpora, re-ran induction over every unchanged document's
   chunks). The refusal was honest: the overlay never reached the per-carrier streams Context
   Enrichment reads, so reused chunks would simply have gone missing from the enriched corpus.
2. **The overlay was appended, not merged.** Reused records were yielded *after* the fresh ones,
   so a corpus-level accumulator saw the union in a different arrival order than a full rebuild
   would deliver. Most accumulators fold in sorted-id order and don't care, but three named
   dependences remained: the resolver's first-seen canonical **display spelling**
   (`display.setdefault` — `canonical_name` could flip between `"IBM"` and `"I.B.M."` on a delta),
   the hyperedge dedup's first-arrival **tie-break** (`merged_sources` order, γ-tie primary), and
   the profiler's last-document language + capped pair table. ADR-0043's equivalence claim was
   therefore honest-but-narrow: ids and cluster structure, not bytes.
3. **Nothing was proven at scale.** The equivalence property had unit tests on toy corpora; no
   real export had ever been built incrementally and compared against its full rebuild.

The ticket's named tensions, resolved below: the per-stage invalidation cascade, exact-recompute
vs bounded-staleness for corpus statistics (BM25 idf/avgdl, profiling), the scope of
re-disambiguation, and the graph-feature refresh policy.

## Decision

### 1. The invalidation cascade is the stage-level partition — nothing else

The increment unit is the **document** (content-addressed, so add/update/remove are manifest
diffs). What a churn invalidates is read off the Capability descriptors' existing
document-level/corpus-level partition (`capability_descriptor.CORPUS_LEVEL_KINDS`), not a new
hand-maintained table:

| Layer | Stages | On churn |
| --- | --- | --- |
| Doc-level | source, screening, parse, chunk, schema induction, entity/relation/fused extraction, redaction | **Reused** for unchanged documents (per-document reproducible under the frozen schema / seeded vocabulary); run only for new/changed documents. |
| Corpus-level | type consolidation, disambiguation, graph assembly (+ features + hyperedges), graph completion, context enrichment, profiling | **Exact recompute over the union** (reused ∪ fresh), every churn. |
| Export-attached | embeddings, BM25 stats/postings, sparse/multi-vector signals, all export files | **Exact recompute**, every churn — they are pure functions of the exported row set. |

### 2. Corpus-level statistics: exact recompute; bounded staleness is REJECTED

BM25 `idf`/`avgdl`/`df`, profiling's corpus record, graph features and context headers shift on
every churn. We recompute them exactly, every delta, and reject bounded-staleness caching for all
of them. Rationale, with the reference measurement (multihop_rag, 609 docs, 3 264 chunks, M1 Pro
CPU): the corpus-level phase is a small, model-free fraction of the pipeline; the expensive phase
is extraction (GLiNER2 forward passes), and that is already O(changed). A staleness window on
`idf` or on community ids would buy minutes of CPU at the price of a silently divergent retrieval
state — forbidden by the operating contract ("no silent divergence"). If a future corpus makes the
corpus-level phase the bottleneck, the escape hatch is a *measured* ADR amending this one, not a
config knob.

Two deliberate consequences are documented rather than hidden:

* **Dense embeddings are recomputed per export run** (they are export-attached, ADR-0017). On a
  GPU-less host this is the dominant churn cost. It is still exact-recompute by decision: an
  embedding cache keyed by embedding-input text would be correct (the input is
  content + context-header, both deterministic), but it is an optimization with real invalidation
  subtleties (the header changes whenever the chunk's KG neighborhood changes) — deferred until
  measured need, recorded in the ticket-03 report.
* **The committed Corpus Version stays affected-set-scoped** (ADR-0018): untouched clusters are
  carried forward verbatim in `records.jsonl` as the audit/transaction artifact. The *exported
  retrieval state* is never that splice — it is the run's full-union recompute. So
  re-disambiguation scope is: **in-run = full union (exact); committed store = affected-set
  (audit)**. New surfaces can flip a merge decision, and the export reflects the flip immediately.

### 3. Graph features are refreshed every churn

Degree/community/centrality are global by construction (community id = lexicographically smallest
node id of the component; degree centrality divides by n−1), so *any* node or edge churn
invalidates all of them. They are recomputed inside Graph Assembly every run — there is no
incremental path and none is pretended.

### 4. Reuse extends to context-enrichment pipelines

`DeltaRunner._reuse_supported`'s `context_enrichment` refusal is removed. It is sound now — and
only now — because of §5: Context Enrichment is a corpus-level Stage that re-derives **every**
chunk's header from the *current* assembled graph each run, and with the overlay reaching its
per-carrier chunk stream it sees reused ∪ fresh chunks exactly as a rebuild would. The reuse store
continues to persist the **redacted (pre-enrichment) chunk variant** (the corpus-level kinds are
skipped when collecting `doclevel.jsonl`), so no KG-derived artifact is ever carried across runs —
the stale-header hazard is structurally gone, not tolerated.

### 5. The overlay folds in document order, at the first doc-level producer

One rule replaces the append + the `TYPE_CONSOLIDATION` suppression special-case:

> The reuse overlay of carrier `T` is folded into the stream of the **first document-level Stage**
> (in the consumer's reading order) that produces `T`, **document-interleaved in source order**. A
> corpus-level parent's re-emission is never overlaid — it already carries the union.

Document-interleaved: a doc-level checkpoint holds its records grouped per document, in the
Source's globally-sorted URI order; the overlay's records are grouped per document with within-doc
order preserved from their original run. Merging the two doc-sorted subsequences by
`(provenance.source_uri, document_id)` reproduces exactly the arrival order a full rebuild
delivers — so the first-seen display spelling, the hyperedge tie-breaks, the profiler's
last-document fields, the enriched chunk stream (and therefore the export row order and the
embedder's batch composition) are all rebuild-identical, not merely set-identical. Memory cost is
O(overlay) — which the delta path already pays to hold the overlay at all — plus O(1) fresh
records. The same rule applies to a doc-level Export's materialization: the overlay is *merged*
(not appended) when the nearest producing ancestor is doc-level, and suppressed entirely when it
is corpus-level (e.g. an export fed by Context Enrichment, whose checkpoint already carries the
enriched union). Multi-Source pipelines whose URI ranges interleave across Sources fall outside
the byte-order guarantee (sets and rankings still hold); single-Source pipelines — every reference
stack — get the full guarantee.

### 6. Deletion semantics: two paths, one incremental

* **File-removal churn** (`latence delta` after files disappeared from the source) is the
  incremental remove: survivors are reused, the manifest diff classifies the gone ids DELETED, and
  the DeltaProcessor retracts by provenance.
* **Explicit `retract`/`purge`** (erasure of documents that may still exist at the source)
  deliberately re-extract the survivors from scratch — split-on-delete through the real pipeline,
  the correctness-first erasure path. Their O(corpus) cost is documented, unchanged here, and
  acceptable for what is a compliance operation, not a churn loop.

### 7. Induced-schema corpora never silently re-induce on churn

Unchanged documents are excluded at Source, so their chunks are never re-induced; a
`granularity: corpus` induction Stage runs under the frozen schema (ADR-0043 §1); per-chunk
induction consistency rides the seeded canonical type vocabulary (ADR-0054/T4). With §4 this now
holds for enrichment pipelines too — before, the reuse refusal silently re-ran induction over the
whole corpus on every delta of the SOTA stack. Enforced by test: a counting fake inducer proves
the induce Stage touches only the changed documents' chunks on an enrichment-pipeline delta.

### 8. The gate: incremental == full rebuild, proven on real data

The equivalence property is verified two ways, permanently:

* **CI-scale**: an e2e test builds a base corpus, applies add + update + remove deltas through an
  enrichment + export pipeline, full-rebuilds the same final corpus in a fresh store, and asserts
  identical exported row sets and graph topology (and byte-identical rows where the profile is
  deterministic).
* **Real-scale (the ticket-03 gate)**: the s3 `multihop_rag` corpus (609 documents, hand-authored
  schema, zero LLM calls) built as ~90% base + 10% incremental add + an update batch + a remove
  batch, compared against a full rebuild of the same final corpus: exported row sets, graph
  topology (nodes/edges/hyperedges), BM25 statistics, and retrieval rankings on the real query
  set; embeddings bitwise where the device's determinism permits, else a measured, documented
  tolerance. Results live in `docs/evidence/enterprise-serving-03-incremental-ingest.md`; every
  deliberate divergence is measured there — none is silent.

## Consequences

* The enterprise churn story is uniform: `latence delta` / `retract` / `purge` cover
  add/update/remove on **every** reference profile, with extraction O(changed) and an exported
  retrieval state that is exactly the full-rebuild state.
* ADR-0043's equivalence claim upgrades from "ids and cluster structure" to "arrival-order
  identical union" — display names, hyperedge provenance order and export row order included.
* The corpus-level phase remains O(corpus) per churn by decision (§2): exactness is the product;
  the measured costs and the deferred optimizations (embedding cache, incremental BM25 df
  maintenance — both reconstructible from existing artifacts) are recorded in the ticket report.
* `LocalRunner._parent_reemits_overlay` is superseded by the fold rule and removed; the
  behavior it guarded (no double-fold through Type Consolidation) is a special case of §5.
