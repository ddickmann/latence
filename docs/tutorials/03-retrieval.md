# Tutorial 3 — Retrieval: four signals, fused

**What you will build.** A working query path over the files Tutorial 1 emitted — dense vectors,
learned-sparse term weights, BM25 and knowledge-graph traversal — fused into one ranking and packed
into a token budget. Then you will take it apart and see what each signal actually contributes, and
where each one is blind.

**Prerequisites.** [Tutorial 1](01-your-first-pipeline.md) and
[Tutorial 2](02-understanding-the-output.md). You will re-run the pipeline once, with one line added.

**Time.** About 30 minutes.

---

## 0. The one architectural fact to hold on to

The retrieval layer is **strictly stateless**. It never holds an index and never runs first-stage
search ([ADR-0048](../adr/0048-retrieval-tooling-is-strictly-stateless.md)). It is a set of pure
transforms over a candidate list *you* fetched from *your* engine — Qdrant, Elastic, Weaviate,
Azure AI Search, whatever you already run.

That splits retrieval into two clocks:

* **Index time** — the dense embedder, the sparse embedder, the BM25 tokenizer and the graph
  features are Providers *inside* the pipeline. They turn chunk text into vectors, term weights and
  statistics, materialised as files during the run.
* **Query time** — fuse, expand, rerank and pack are a downstream library. They hold nothing and
  fetch nothing.

In this tutorial your corpus is four documents, so *you* are the engine: the candidate list is
simply every row of `records.parquet`. Every component below behaves identically when the candidates
come from a real store instead — that is the whole point of the seam.

---

## 1. Add the sparse signal

Tutorial 1's stack already emits a dense `embedding` column and BM25 statistics. Add the third
index-time signal by putting one more line in the `export_corpus` config:

```yaml
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
    config:
      basename: records
      context_columns: true
      embedder: {provider: embedding.hashing, config: {dimension: 256}}
      bm25: {provider: tokenizer.regex, config: {}}
      sparse_embedder: {provider: sparse.hashing, config: {vocab_size: 1024}}   # new
```

```bash
uv run latence stack check latence.stack.yaml
uv run latence run latence.stack.yaml --run-id run-0003
uv run python -c "
import pyarrow.parquet as pq
print(pq.read_schema('latence-out/_latence/runs/run-0003/export/records.parquet').names)
"
```

✓ **look for** — two new columns, `sparse_indices` and `sparse_values`, and nothing else changed:

```
['record_id', 'schema_version', 'provenance', 'classification', 'content', 'media_type',
 'embedding', 'sparse_indices', 'sparse_values', 'context_header', 'kg_node_ids',
 'neighbor_node_ids', 'kg_node_community', 'kg_node_centrality']
```

Parallel `list<int64>` / `list<double>` columns are the shape a store's sparse index ingests
directly. Like every signal, it is generated over the **redacted** corpus text, so the sparse index
can never re-leak the PII the corpus removed.

You also need the embedded graph engine the multi-hop expander uses (MIT, serverless, nothing to
stand up), if you did not install it during the quickstart:

```bash
uv pip install -e 'packages/latence-retrieval[graph-duckdb]'
```

---

## 2. Build the four legs

Save this as `search.py`:

```python
"""Four retrieval signals over the emitted files, fused, graph-expanded and packed."""

import math
import sys

import pyarrow.parquet as pq
from latence_core.capability import ProviderRegistry
from latence_retrieval import (
    Bm25Rescorer,
    Candidate,
    DuckDBGraphSource,
    MultiHopExpander,
    Packer,
    RrfFuser,
)

EXPORT, QUERY = sys.argv[1], sys.argv[2]
rows = pq.read_table(f"{EXPORT}/records.parquet").to_pylist()
registry = ProviderRegistry()


def candidate(row, score):
    return Candidate(
        id=row["record_id"],
        score=score,
        text=row["content"],
        metadata={"kg_node_ids": list(row["kg_node_ids"] or [])},
    )


def show(label, cands, n=4):
    print(f"\n-- {label}")
    for c in cands[:n]:
        print(f"   {c.score:8.4f}  {c.text.splitlines()[0][:44]}")


def rank(scored):
    return sorted(scored, key=lambda c: (-c.score, c.id))


# 1 - dense: the SAME Embedder Provider the Export wired, resolved by name
dense_model = registry.load("embedding.hashing")({"dimension": 256})
qvec = next(iter(dense_model.embed([QUERY])))


def cosine(a, b):
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / den if den else 0.0


dense = rank(candidate(r, cosine(qvec, r["embedding"])) for r in rows)
show("dense - cosine over the emitted `embedding` column", dense)

# 2 - sparse: the SAME SparseEmbedder Provider, dot product in term space
sparse_model = registry.load("sparse.hashing")({"vocab_size": 1024})
qs = next(iter(sparse_model.encode_sparse([QUERY])))
qterms = dict(zip(qs.indices, qs.values))
sparse = rank(
    candidate(
        r,
        sum(qterms.get(i, 0.0) * v for i, v in zip(r["sparse_indices"], r["sparse_values"])),
    )
    for r in rows
)
show("sparse - dot product over `sparse_indices`/`sparse_values`", sparse)

# 3 - BM25: exact rescoring from the emitted corpus statistics
lexical = Bm25Rescorer.from_export(EXPORT).process(QUERY, [candidate(r, 0.0) for r in rows])
show("BM25 - from bm25-stats.json, no re-tokenisation", lexical)

# 4 - fuse: three incomparable scales, so fuse by rank alone
fused = RrfFuser().fuse([dense, sparse, lexical])
show("fused - reciprocal rank over the three lists", fused)

# 5 - graph: edge-path proximity to the top hit
expanded = MultiHopExpander(
    {"seed_top_k": 1, "max_hops": 2, "hop_weight": 0.01},
    graph_source=DuckDBGraphSource(f"{EXPORT}/graph-edges.parquet"),
).process(QUERY, fused)
show("after 2-hop graph expansion", expanded)

# 6 - pack: relevance minus redundancy, under a token budget
show("packed @ 300 tokens", Packer({"budget": 300}).process(QUERY, expanded))
```

Two things in there are not incidental.

`ProviderRegistry().load("embedding.hashing")` resolves the **same Provider by the same name** the
Export used. That is how a query vector is guaranteed to land in the same space as the corpus
vectors — you do not re-implement the embedder on the query side, you resolve it. The identical
discipline applies to the sparse encoder, and to BM25: `Bm25Rescorer.from_export` builds the
rescorer straight from the emitted `bm25-stats.json`, mapping its `avgdl` and per-term idf onto the
rescorer's statistics. **No re-tokenisation of the corpus happens at query time.** Note that it is a
classmethod, not a bare constructor — a bare `Bm25Rescorer()` falls back to using the candidate pool
as its own corpus, which is a different and much weaker statistic.

---

## 3. Run it

```bash
uv run python search.py latence-out/_latence/runs/run-0003/export "who signed for Globex"
```

✓ **look for** — six blocks. The first three are the raw signals:

```
-- dense - cosine over the emitted `embedding` column
     0.4291  Master Services Agreement
     0.3313  Travel and Expense Policy
     0.3162  Incident Report 2024-07
     0.1596  Acme Corporation Employee Handbook

-- sparse - dot product over `sparse_indices`/`sparse_values`
     3.0000  Master Services Agreement
     2.0000  Incident Report 2024-07
     2.0000  Travel and Expense Policy
     1.0000  Acme Corporation Employee Handbook

-- BM25 - from bm25-stats.json, no re-tokenisation
     1.8207  Master Services Agreement
     0.5277  Travel and Expense Policy
     0.4981  Incident Report 2024-07
     0.0000  Acme Corporation Employee Handbook
```

All three put the right document first, and all three are on **completely incomparable scales** —
a cosine in `[-1, 1]`, an unbounded dot product, an unbounded BM25 sum. That is the problem fusion
exists to solve.

---

## 4. What each signal is actually for

Before you fuse them, be clear about what you just measured — and about one thing you did **not**.

**BM25** is exact lexical matching with corpus-aware weighting. A rare term (`IBAN`, a status code,
a header field name) carries high idf and dominates; a chunk sharing none of the query's terms
scores exactly `0.000`. It is unbeatable on precise lookups and blind to paraphrase.

**Learned sparse** produces term weights in a vocabulary space, so it can expand a query into
related terms while staying in the interpretable, invertible world of terms. In production this is
SPLADE; the query side reuses the *same composition class* the pipeline ran at index time, so query
and corpus land in the identical sparse space.

**The graph** is the only signal that follows *edges* rather than similarity — the subject of the
next section.

**Dense** is supposed to be the semantic signal. Here it is not:

!!! warning "`embedding.hashing` cannot demonstrate semantics — and neither can `sparse.hashing`"
    Both in-core signal generators are **deterministic references**. They hash terms into buckets.
    They exist so the seam, the emission and the entire offline test suite run with zero downloads
    and byte-reproducible output, and they do that job perfectly.

    What they cannot do is retrieve on meaning. Query this corpus with a paraphrase that shares no
    words with the answer and the "dense" leg scores it near zero, because there is nothing semantic
    inside it. **Do not conclude anything about dense-vs-lexical retrieval from these numbers** —
    you would be comparing a lexical signal to a lexical signal.

    To see the real behaviour, swap in a real model. It is a one-line config change on the
    `embedder` block — `{provider: embedding.sentence_transformers, config: {model: ibm-granite/granite-embedding-97m-multilingual-r2, dimension: 384, device: cpu}}`
    — plus `uv pip install -e packages/latence-embedder-st`. The query side changes the same
    one line, because it resolves the Provider by name. That the swap is one line in two places is
    the point of the seam; that the reference is honest about being a reference is the point of
    this box.

---

## 5. Fusion

```
-- fused - reciprocal rank over the three lists
     0.0492  Master Services Agreement
     0.0481  Travel and Expense Policy
     0.0479  Incident Report 2024-07
     0.0469  Acme Corporation Employee Handbook
```

`RrfFuser` sums `weight / (k + rank)` across the input lists. It looks only at **ranks**, never at
scores, which is exactly right when the scores have no common unit: a cosine of `0.43` and a BM25
of `1.82` cannot be added, but "first in both lists" is meaningful in any unit.

Notice the consequence: the fused scores are all around `0.05`, in a band about `0.002` wide.
Reciprocal-rank scores are small and tightly clustered by construction. Remember that number — the
next step depends on it entirely.

Two variations worth knowing:

* `RrfFuser().fuse([dense, sparse, lexical], weights=[2.0, 1.0, 1.0])` weights a leg you trust more.
* `WeightedFuser` preserves magnitudes (min-max normalised) when you actually do want score
  differences rather than rank differences.

Fusion is skipped entirely when only one leg fired, so a single store's own scores are never
discarded and re-derived.

---

## 6. The graph leg, and how to get it wrong

```
-- after 2-hop graph expansion
     0.0581  Travel and Expense Policy
     0.0579  Incident Report 2024-07
     0.0569  Acme Corporation Employee Handbook
     0.0492  Master Services Agreement
```

That is **wrong**, and instructively so. The correct answer went from first to last.

Here is what happened. `MultiHopExpander` anchors on the top `seed_top_k` candidates, reads their
`kg_node_ids`, traverses `graph-edges.parquet` out to `max_hops`, and gives every *other* candidate
a hop-decayed bonus for the closest hop at which one of its own nodes lands in that neighbourhood.
A one-hop neighbour earns the full `hop_weight`; a *k*-hop neighbour earns
`hop_weight × hop_decay^(k-1)`.

Two things went wrong at once:

1. **The bonus was scaled to the wrong range.** `hop_weight: 0.01` is a sensible nudge next to raw
   BM25 scores of ~5. Next to fused reciprocal-rank scores spread over `0.002`, it is five times the
   entire score range — so the bonus stopped being a tiebreaker and became the ranking.
2. **On a four-document corpus every document is one hop from every other.** The graph has five
   nodes and everything touches Acme or Globex. A uniform bonus to everything-but-the-seed is,
   precisely, an inversion.

Fix the first by scaling the weight to the signal it is added to:

```python
expanded = MultiHopExpander(
    {"seed_top_k": 1, "max_hops": 2, "hop_weight": 0.0005},
    graph_source=DuckDBGraphSource(f"{EXPORT}/graph-edges.parquet"),
).process(QUERY, fused)
```

✓ **look for** — the seed holds first place and the graph acts as a tiebreaker among the rest:

```
-- after 2-hop graph expansion
     0.0492  Master Services Agreement
     0.0486  Travel and Expense Policy
     0.0484  Incident Report 2024-07
     0.0474  Acme Corporation Employee Handbook
```

The second problem is not tunable — it is a property of a four-document corpus, and there is no
honest way to make a toy graph produce a multi-hop win. What a real corpus looks like is the next
section.

!!! note "Two more traps worth knowing before you meet them"
    **Node ids are content hashes, not labels.** `graph-nodes.parquet` node ids look like
    `96987fab1d4dd7e2`. Seeding a traversal with the literal string `"Acme"` resolves to nothing and
    returns an empty neighbourhood — which looks exactly like a dead graph. Resolve the label to its
    `node_id`(s) via `canonical_name` first.

    **A large `seed_top_k` can absorb the whole graph.** A candidate whose entity nodes are already
    *in the seed set* earns no bonus. On this corpus, `seed_top_k: 2` covers four of the five nodes
    and expansion becomes a complete no-op — zero bonuses, no error, no warning. Start at `1` and
    raise it as the corpus grows.

---

## 7. What the graph is actually for

The mechanism is best stated as the thing single-hop retrieval structurally cannot do.

Take a question framed in one document and answered in another: *"the cipher suites for the TLS
protocol referenced by RFC 7230."* The query is about RFC 7230 (HTTP); the answer lives in RFC 5246
(TLS). No lexical signal puts RFC 5246's chunk near this query — its text is not about HTTP. No
semantic signal does either, for the same reason. Fusing them cannot help: fusion makes each signal
cover the other's blind spots, and here they share the blind spot. The evidence is one reference
edge away, and neither signal follows edges.

The expander does. It takes the RFC 7230 chunks the first stage *did* find, reads their entity node
ids, traverses the `7230 → 5246` reference edge in the emitted graph, and rescores RFC 5246's chunk
up into the returned window.

That has been measured, on 80 IETF RFCs run through the enterprise-SOTA pipeline against a live
Qdrant instance:

| retrieval configuration | recall@10 · all 90 questions | recall@10 · the 7 multi-hop cases |
|---|---|---|
| dense only (Granite → Qdrant) | 0.644 | **0.000** |
| BM25 only | 0.656 | **0.000** |
| **fused — dense + BM25 + KG 2-hop + rerank** | **0.678** | **1.000** |

Read that honestly, which is how the [deep dive](../DEEP-DIVE.md) reports it. Across the broad
90-question set the aggregate lift is **modest** — three points — because most questions are
answerable by a single signal and fusion merely ties them. The advantage lands exactly where it
should: on the seven questions that require chaining evidence across documents, where single signals
score **zero** and the fused stack scores **perfect**. Seven of ninety is about 8%, and that is the
true rate, not a padded one.

The defensible claim is not "fused is several times better everywhere". It is that **the
knowledge-graph stack unlocks a class of queries single retrievers structurally cannot answer.**

One detail from that run is worth carrying into your own corpus: the induced graph initially
recovered only **14%** of the corpus's real cross-reference edges, because the header-only extractor
could not see inline body citations. Adding the deterministic `inline_refs` extractor lifted
coverage to **~112%** of the gold graph. Those recovered edges are precisely what the expander
traverses. **No edges, no multi-hop win** — the retrieval result is only ever as good as the graph.

---

## 8. Packing

With the corrected `hop_weight`, the packed block is the whole reordered list — four short chunks
fit comfortably in 300 tokens:

```
-- packed @ 300 tokens
     0.0492  Master Services Agreement
     0.0486  Travel and Expense Policy
     0.0484  Incident Report 2024-07
     0.0474  Acme Corporation Employee Handbook
```

`Packer` solves a quadratic knapsack: it maximises **relevance minus redundancy** under a token
budget, greedily adding the candidate with the best
`(value − λ · similarity-to-already-picked) / cost` that still fits. Naive top-k spends the budget
on three near-identical passages; the packer prefers diverse, non-redundant evidence.

It **selects, it does not re-score** — the returned candidates are the originals, unchanged, ordered
by descending first-stage score. So the order is: rerank first, then pack.

Useful knobs:

* `budget` — required, in tokens.
* `redundancy_weight` — the λ on pairwise similarity (default `0.5`; `0` gives a pure top-value
  knapsack).
* `value_signals` — a `metadata-key → weight` map of additive richness signals. This is how you feed
  a chunk's peak entity centrality into the packing decision, using the `kg_node_centrality` column
  the Export already wrote.

The token counter is itself a seam: a zero-dependency heuristic by default, real `tiktoken` when you
install `latence-retrieval[pack]`.

✓ **look for** — drop `budget` to `100` and re-run: the packed block loses its lowest-value member
and returns three chunks instead of four. The budget is a hard constraint — every candidate costs at
least one token, so a selection never exceeds it.

---

## 9. Reranking, and the same story with a real engine

Two pieces this tutorial has left out on purpose.

**Reranking** is the precision pass between fusing and packing: a cross-encoder scores each
candidate against the query *jointly*, rather than as independent embeddings, and reorders. The
`Reranker` component pins **no checkpoint** — you name one, and its license is your explicit choice.
Install `latence-retrieval[rerank]` for the real `sentence-transformers` cross-encoder adapter.

**A real engine.** Everything above works unchanged when the candidate list comes from a store
instead of a Parquet file. Load the export:

```bash
uv run latence-sink-qdrant load latence-out/_latence/runs/run-0003/export \
  --collection first-pipeline --url http://localhost:6333
```

`load` is idempotent — re-run it and it upserts only what changed and sweeps what the Export no
longer contains, which is what makes a `retract` or `purge` take effect in the store. Its
`--vector dense|sparse|both` flag decides which signal it writes; under `both` the sparse column
pair becomes a queryable **named sparse vector** rather than inert payload.

On the query side, `QdrantBackend` declares its honest capabilities from the closed set
`{dense, sparse, bm25, filter, multivector}` and exposes exactly one method, `search(query)` —
**the only fetch in the entire library, and it runs on your engine.** The key it reads for chunk
text is the key the sink writes, so a collection this framework loaded needs no mapping
configuration. Point the seam at your own store and nothing else in this tutorial changes.

A capability the backend cannot serve is a **loud error**, never a silent drop, and the result
reports which modes actually ran.

---

## What you have

A query path built from the emitted files alone, and a clear-eyed view of what each signal does:
BM25 for exact terms, sparse for term expansion, dense for meaning (once you wire a real model), the
graph for evidence that similarity cannot reach — and the two ways to get the graph leg wrong.

**Next:** [Bringing your own Provider](04-bringing-your-own-provider.md) — implement a Capability,
register it as a plugin, and have the conformance suite check you.
