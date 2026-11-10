# Incremental ingest: document churn without a full rebuild

!!! warning "Written before the retrieval tooling was removed"
    This page predates [ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md), which removed the query-time retrieval
    tooling together with Context Enrichment (`enrich`), graph features, hyperedges, the
    BM25 / sparse / multi-vector export signals and the Qdrant sink. The measured pipeline included Context Enrichment and Hyperedge stages; the incremental-ingest mechanism itself is unchanged.
    The page is kept until it is rewritten.

> **Status: production capability, equivalence-verified.** The machinery is
> `latence_core.delta_runner.DeltaRunner` behind the `latence delta` / `retract` / `purge` CLI
> verbs; the design is [ADR-0061](../adr/0061-incremental-ingest-exact-corpus-recompute-over-an-order-preserving-reuse-union.md)
> (over [ADR-0018](../adr/0018-incremental-corpus-deltas.md) /
> [ADR-0043](../adr/0043-incremental-extraction-frozen-schema-and-doc-level-cross-run-reuse.md)).
> The headline guarantee is **proven on real data**: building a 609-document corpus as 90% base
> + incremental add/update/remove churn produces an exported retrieval state equivalent to a
> full rebuild of the same final corpus (identical row sets, graph topology, BM25 statistics and
> retrieval rankings — see the ticket-03 equivalence report).

Enterprises do not re-run pipelines over frozen corpora — documents arrive, get edited, and get
deleted every day. Latence treats that churn as a first-class operation: the **document** is the
increment unit (content-addressed by the sha256 of its bytes), and every churn commits an
immutable, auditable **Corpus Version**.

## The three operations

All three take the same stack/pipeline YAML that `latence run` takes, and each commits Version
N+1 with the churn counters folded into the run's Quality Report.

```bash
# Preview: what WOULD a delta do? Runs no Stage, loads no model, commits nothing.
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
  is retired), so only the edited documents re-extract.
* **Remove** — delete files from the source and run `latence delta` (the incremental path:
  survivors are reused, the gone ids are retracted), or use `retract`/`purge` for documents that
  must be erased even though the source still holds them (these deliberately re-extract the
  survivors — split-on-delete through the real pipeline — the compliance path, not the churn
  loop).

## What a churn recomputes (and what it never recomputes)

The invalidation cascade is exactly the framework's document-level / corpus-level Stage
partition (ADR-0061 §1):

| | On churn |
| --- | --- |
| **Doc-level** (parse, chunk, screening, induction, extraction, redaction) | **Reused** for unchanged documents; run only for new/changed ones. Extraction cost is O(changed). |
| **Corpus-level** (type consolidation, disambiguation, graph + features + hyperedges, context headers, profiling) | **Exactly recomputed over the union** (reused ∪ fresh) every churn. |
| **Export-attached** (embeddings, BM25 stats/postings, sparse signals, export files) | **Exactly recomputed** — pure functions of the exported row set. |

Two properties make this safe to rely on:

1. **No silent staleness, anywhere.** BM25 `idf`/`avgdl`, graph degree/community/centrality and
   every chunk's KG context header shift when the corpus shifts — so they are recomputed
   exactly, every delta. Bounded-staleness caching was considered and rejected (ADR-0061 §2):
   the expensive phase is model extraction, which is already incremental; the corpus-level phase
   is model-free and cheap.
2. **Order-preserving reuse.** Reused records are not appended after the fresh ones — they are
   merged back document-interleaved in source order, so every corpus-level Stage (and the
   Export, and the embedder's batching) consumes the union in exactly the order a full rebuild
   would deliver. That is what upgrades "equivalent" to *byte-identical* on deterministic
   hardware.

Context-enrichment pipelines are fully supported: the cross-run store keeps only the
pre-enrichment chunk variant, and every chunk's `context_header` is re-derived from the current
assembled graph each run — a reused document's header can never go stale.

Induced-schema corpora never silently re-induce on churn: unchanged documents are excluded at
Source (their induced chunks ride the reuse store), a `granularity: corpus` induction Stage runs
under the frozen committed schema, and per-chunk induction canonicalizes against the committed
type vocabulary (append-don't-churn — existing canonical types never rename). `--reinduce` is
the explicit full rebuild for when you *want* the vocabulary re-elected.

## Reading the churn

Every delta prints (and persists, in `quality-report.json`) its witnesses:

```text
Corpus Version v3 -> v4
  documents: +61 ~0 -0 (548 unchanged)
  reused:    548 documents (extraction ran on 61)
  ...
  affected set: 812 of 24913 records
```

`documents_reused` / `documents_extracted` prove the O(changed) extraction;
`affected_set_size` vs `corpus_record_count` proves the committed corpus-level splice stayed
scoped (ADR-0018). If `documents_reused` is unexpectedly 0, the base Version is missing (first
run) or `--reinduce` was passed.

## Cost model, honestly

* Extraction (the model-heavy phase): **O(changed documents)**.
* Corpus-level resolution + graph + headers: **O(corpus)**, model-free, exact by decision.
* Export embeddings: **O(corpus rows)** per export run — the dominant churn cost on CPU-only
  hosts. Exact-recompute by decision; an embedding-input-keyed cache is the documented deferred
  optimization (ADR-0061 §2).
* `retract`/`purge`: **O(corpus)** re-extraction of survivors — deliberate (erasure correctness
  over speed).
