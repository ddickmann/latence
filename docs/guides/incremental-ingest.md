# Incremental ingest: document churn without a full rebuild

> **Status: shipped, equivalence-tested.** The machinery is
> `latence_core.delta_runner.DeltaRunner` behind the `latence delta` / `retract` / `purge` CLI
> verbs; the design is [ADR-0061](../adr/0061-incremental-ingest-exact-corpus-recompute-over-an-order-preserving-reuse-union.md)
> (over [ADR-0018](../adr/0018-incremental-corpus-deltas.md) /
> [ADR-0043](../adr/0043-incremental-extraction-frozen-schema-and-doc-level-cross-run-reuse.md)).
> The guarantee is that a corpus built by incremental churn exports the same files as a full
> rebuild of the same final corpus. A test holds it on every build:
> `packages/latence-core/tests/test_e2e_delta_churn_equals_rebuild.py` applies an add, an
> update and a remove, and asserts that the final exported corpus (`records.jsonl`) and the
> assembled-graph checkpoint are byte-identical to a batch rebuild.
>
> The guarantee was also measured at scale on 2026-08-13, on a 609-document corpus. That run
> used the pipeline as it was then, which included Context Enrichment, hyperedges, graph
> features and a BM25 export. All four were removed by
> [ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md),
> and the measurement has not been repeated on the current pipeline. The dated record is the
> [ticket-03 report](../evidence/enterprise-serving-03-incremental-ingest.md).

Enterprises do not re-run pipelines over frozen corpora — documents arrive, get edited, and get
deleted every day. Latence treats that churn as a first-class operation: the **document** is the
increment unit (content-addressed by the sha256 of its bytes), and every churn commits an
immutable, auditable **Corpus Version**.

## The three operations

All three take the same stack/pipeline YAML that `latence run` takes. Each commits the next
Corpus Version and folds the churn counters into the run's Quality Report. A delta that finds
nothing changed commits no new Version.

```bash
# Preview: what WOULD a delta do? Runs no Stage, loads no model, commits nothing.
# Add --verbose to list the document ids in each class.
latence delta stack.yaml --dry-run

# Add/update/remove — you changed files in the source; this detects everything by content hash.
latence delta stack.yaml

# Explicit soft delete (auditable, rollback-able): derived records tombstoned, clusters split.
latence retract stack.yaml --doc sha256:...

# Hard GDPR erasure: source + every derived record + Evidence physically removed, everywhere.
latence purge stack.yaml --doc sha256:... --yes
```

* **Add** — drop new files into the source; the delta extracts only them.
* **Update** — edit files in place; a content-addressed edit mints a new document id (the old id
  is retired), so only the edited documents re-extract. The churn line reports an update as one
  document added and one deleted.
* **Remove** — delete files from the source and run `latence delta` (the incremental path:
  survivors are reused, the gone ids are retracted), or use `retract`/`purge` for documents that
  must be erased even though the source still holds them (these deliberately re-extract the
  survivors — split-on-delete through the real pipeline — the compliance path, not the churn
  loop).

## What a churn recomputes (and what it never recomputes)

The invalidation cascade is exactly the framework's document-level / corpus-level Stage
partition (ADR-0061 §1), read from the Capability descriptors
(`capability_descriptor.CORPUS_LEVEL_KINDS`):

| | On churn |
| --- | --- |
| **Doc-level** (source, screening, parse, chunk, schema induction, extraction, redaction) | **Reused** for unchanged documents; run only for new/changed ones. Extraction cost is O(changed). |
| **Corpus-level** (type consolidation, disambiguation, graph assembly, graph completion, profiling) | **Exactly recomputed over the union** (reused ∪ fresh) every churn. |
| **Export** (the corpus and knowledge-graph files, and the embeddings when an `embedder` is wired) | **Exactly recomputed** — pure functions of the exported row set. |

Two properties make this safe to rely on:

1. **No silent staleness, anywhere.** Entity clusters, canonical names, the graph's nodes and
   edges and the profiler's corpus record shift when the corpus shifts — so they are recomputed
   exactly, every delta. Bounded-staleness caching was considered and rejected (ADR-0061 §2):
   the expensive phase is model extraction, which is already incremental.
2. **Order-preserving reuse.** Reused records are not appended after the fresh ones — they are
   merged back document-interleaved in source order
   (`reuse_overlay.merge_overlay_doc_ordered`), so every corpus-level Stage (and the Export, and
   the embedder's batching) consumes the union in exactly the order a full rebuild would
   deliver. That is what upgrades "equivalent" to *byte-identical* on deterministic hardware.
   The byte-order guarantee covers single-Source pipelines; with several Sources whose URI
   ranges interleave, set equivalence still holds but row order may differ (ADR-0061 §5).

Induced-schema corpora never silently re-induce on churn: unchanged documents are excluded at
Source (their induced chunks ride the reuse store), a `granularity: corpus` induction Stage runs
under the frozen committed schema, and per-chunk induction canonicalizes against the committed
type vocabulary (append-don't-churn — existing canonical types never rename). `--reinduce` is
the explicit full rebuild for when you *want* the vocabulary re-elected.

## Reading the churn

Every delta prints (and persists, in the `delta` block of `quality-report.json`) its witnesses.
This is one added document on a four-document corpus:

```text
Corpus Version v0 -> v1 (incremental)
  documents: +1 ~0 -0 (=4 unchanged)
  entities:  +0 created, 3 merged, 0 split
  edges:     +1 -0
  records:   0 retracted, 0 purged (retraction)
  affected set: 21/30 corpus-level records recomputed
  extraction: 1 extracted, 4 reused (unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)
  drift: 0.200
```

`documents_reused` / `documents_extracted` (the `extraction:` line) prove the O(changed)
extraction; `affected_set_size` vs `corpus_record_count` (the `affected set:` line) proves the
committed corpus-level splice stayed scoped (ADR-0018). If `documents_reused` is unexpectedly 0,
the base Version is missing (first run), `--reinduce` was passed, or the operation was a
`retract` or `purge`, which re-extract the survivors by design.

## Cost model, honestly

* Extraction (the model-heavy phase): **O(changed documents)**.
* Corpus-level resolution and graph assembly: **O(corpus)**, exact by decision. This phase is
  model-free with a deterministic resolver such as `disambiguation.exact_surface`; a resolver
  that uses an embedder or a linking model runs that model over the union on every churn.
* Export embeddings: **O(corpus rows)** per export run. With a learned embedder this is the
  dominant churn cost on a host without a GPU (ADR-0061 §2). Exact-recompute by decision; an
  embedding-input-keyed cache is the documented deferred optimization.
* `retract`/`purge`: **O(corpus)** re-extraction of survivors — deliberate (erasure correctness
  over speed).
