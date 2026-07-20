# latence-retrieval

The **stateless, query-time** half of the Latence framework's retrieval-quality tooling
(ADR-0048/0049). It gives an adopter's existing retrieval engine SOTA quality **without being one**:
it holds no index, runs no first-stage search, and never fetches anything — it only **transforms
candidate lists** the caller's own engine already returned.

This package is the tested foundation (#159) every later query-time capability (`fuse`, `rerank`,
`pack`, `expand`, the MCP server) builds on. It is a *separate downstream library* from
`latence-core` — an adopter who wants only candidate transforms installs this without pulling the
whole pipeline. Its core is **pure-Python** (only `pydantic`); heavy query-time deps land behind
extras in later tickets.

## What ships here

| Piece | What it is |
|---|---|
| `Candidate` | The retrieved-hit contract: `id`, `score`, `text`, `metadata`. Frozen; round-trips losslessly through dict + JSON. |
| `Capability` | The closed set a store declares: `dense`, `sparse`, `bm25`, `filter`, `multivector`. |
| `RetrievalQuery` | A first-stage request handed to *your* engine — validated so a mode/payload contradiction fails at the boundary. |
| `RetrievalBackend` | The bring-your-own-engine seam: declares `capabilities` and runs one `search`. Never replaced by the tooling (ADR-0052). |
| `FakeBackend` (`latence_retrieval.testing`) | The faithful offline adapter every query-time test drives through — rejects **exactly** what a real store rejects (capability + dimension mismatches). |
| `QdrantBackend` | The **reference production** `RetrievalBackend` — the real store adapter an integrator copies for their own engine (`{dense, sparse, filter}`, behind the `[backend-qdrant]` extra). Declares its capabilities, guards them before the wire, translates one `RetrievalQuery` into a single Qdrant `query_points`, maps `ScoredPoint`s back to `Candidate`s. |
| `CandidateProcessor` + `IdentityProcessor` | The query-time transform seam and its trivial no-op reference. |
| `Reranker` + `CrossEncoderModel` | The `rerank` transform (`processor.rerank`): re-score one candidate list with a config-selected cross-encoder and reorder it. `CrossEncoderModel` is a two-adapter seam — `SentenceTransformersCrossEncoder` (real, behind the `[rerank]` extra) + `FakeCrossEncoder` (shipped faithful fake). |
| `Packer` + `TokenCounter` | The `pack` transform (`processor.pack`): select a token-budgeted subset maximising coverage/relevance minus redundancy — the ported Apache-2.0 Knapsack QKP context-packer. `TokenCounter` is a two-adapter seam — `HeuristicTokenCounter` (pure-Python, the day-one default, no extra) + `TiktokenCounter` (exact, behind the `[pack]` extra). Optional richness signals (centrality, recency, roles, clusters, similarity) degrade gracefully. |
| `Bm25Rescorer` | The `bm25` transform (`processor.bm25`): **exact** Okapi BM25 rescoring of a caller-supplied candidate set — IDF from supplied corpus `stats` or the candidate pool itself, query-term-driven so it stays fast on large sets. Pure-Python, deterministic, no extra. **Never** a first-stage index (ADR-0048). |
| `SpladeQueryEncoder` + `SparseQueryEncoder` | Query-side sparse generation (`query_sparse.splade`, via `resolve_sparse_encoder`): `query text → SparseVector`, **reusing** the index-time ADR-0053 SPLADE composition (`latence-splade`) so the query lands in the corpus's sparse space. `latence-splade` rides the `[sparse]` extra, imported lazily; offline tests inject the faithful `FakeSpladeModel`. |
| `MaxSimReranker` + `MultiVectorQueryEncoder` + `max_sim` | The **experimental** MaxSim (late-interaction) rerank (`processor.maxsim`, ADR-0050, **off by default**): re-score one candidate list by **exact** MaxSim over the raw `multivectors.parquet` sidecar the caller loaded onto each candidate, against the query's multi-vector. `max_sim` is the real (not faked) reference scorer; `MultiVectorQueryEncoder` is a two-adapter seam — `HashingMultiVectorQueryEncoder` (real, **reuses** the index-time `multivector.hashing` embedder so the query matches the corpus contract, behind the `[multivector]` extra, via `resolve_multivector_encoder` as `query_multivector.hashing`) + `FakeMultiVectorEmbedder` (shipped faithful fake). A numpy/SIMD MaxSim kernel is the deferred accelerator (ADR-0049/0050). |
| `RetrievalRegistry` + `resolve_processor` / `resolve_fuser` / `resolve_sparse_encoder` / `resolve_multivector_encoder` | Discovery + config-resolution over the new `latence.retrieval` entry-point group (separate from `latence.providers`). |

## The stateless boundary (ADR-0048)

Every component is one of two shapes — a **signal generator** (index-time, in the pipeline) or a
**candidate processor** (query-time, here). A candidate processor is a pure transform
`(query, candidates) → candidates`: the caller supplies the candidates (their engine ran the
first-stage search), and the processor returns a new list. Nothing here persists an index, stores a
vector, or runs an ANN/sparse/BM25 lookup — that always stays on the customer's engine.

## Usage

```python
from latence_retrieval import Candidate, RetrievalRegistry, resolve_processor

# Resolve a query-time processor by config (same {"provider", "config"} shape as the pipeline).
processor = resolve_processor({"provider": "processor.identity", "config": {}})

# Candidates your own engine already fetched:
candidates = [Candidate(id="a", score=0.9, text="…"), Candidate(id="b", score=0.4, text="…")]
result = processor.process("my query", candidates)
```

### Rerank with a cross-encoder (`[rerank]` extra)

```python
from latence_retrieval import resolve_processor

# Pick any cross-encoder your team has license-cleared (weights AND code) — no default is pinned.
rerank = resolve_processor(
    {"provider": "processor.rerank", "config": {"model": "your-org/cleared-cross-encoder", "top_k": 5}}
)
best = rerank.process("my query", candidates)  # re-scored, reordered, top-5
```

The real model rides `pip install 'latence-retrieval[rerank]'`. Offline tests inject the shipped
faithful `FakeCrossEncoder` (`latence_retrieval.testing`) — one score per text, string-only inputs,
rejecting exactly what a real cross-encoder rejects — so `rerank` is covered without weights.

### Pack a token-budgeted context (`pack`)

```python
from latence_retrieval import resolve_processor

# Select the most useful, least-repetitive subset that fits a 2 000-token context window.
pack = resolve_processor({"provider": "processor.pack", "config": {"budget": 2000}})
context = pack.process("my query", candidates)  # a budgeted, redundancy-penalised subset
```

`pack` runs **day one** on the pure-Python `HeuristicTokenCounter` (no extra). It maximises
value − redundancy under the token budget with a deterministic greedy Knapsack QKP solver, and
accepts optional richness signals from candidate `metadata` that degrade gracefully when absent:

```python
resolve_processor({"provider": "processor.pack", "config": {
    "budget": 2000,
    "redundancy_weight": 0.5,                 # λ on pairwise similarity (0 disables the penalty)
    "value_signals": {"centrality": 0.3, "recency": 0.1},  # additive numeric signals
    "cluster_key": "cluster",                 # same cluster ⇒ redundant
    "similarity_key": "sims",                 # metadata["sims"] = {other_id: similarity}
    "token_key": "n_tokens",                  # exact index-time token counts, when present
    "tokenizer": {"encoding": "cl100k_base"}, # opt-in exact counter (needs the [pack] extra)
}})
```

The exact counter rides `pip install 'latence-retrieval[pack]'` (`tiktoken`, MIT). No encoding is
pinned — you name a license-cleared one (ADR-0012). Offline tests drive the heuristic and a faithful
fake `tiktoken` module, so `pack` is covered without any heavy dep.

### Rescore exactly with BM25 (`bm25`)

```python
from latence_retrieval import resolve_processor

# Exact Okapi BM25 over the candidates your engine already fetched — no second index.
bm25 = resolve_processor({"provider": "processor.bm25", "config": {
    "k1": 1.5, "b": 0.75, "top_k": 10,
    # Optional: pass your engine's TRUE corpus stats to reproduce its global BM25 exactly.
    # Omit `stats` to derive a self-contained, pool-local BM25 from the candidate set itself.
    "stats": {"doc_count": 1_000_000, "avg_doc_len": 240.0, "doc_freq": {"revenue": 8123}},
}})
scored = bm25.process("quarterly revenue", candidates)  # re-scored, reordered, top-10
```

`bm25` is pure-Python, deterministic, and **never** builds a first-stage index (ADR-0048): it scores
exactly the candidate set you hand it. Scoring is query-term-driven (only documents containing a
query term are touched), so it stays fast on large candidate sets. With no `stats` it uses the
candidate pool as its own corpus (a documented pool-local IDF); pass real corpus `stats` to
reproduce your engine's global BM25.

### Query-side sparse: mirror the index-time SPLADE signal (`[sparse]` extra)

```python
from latence_retrieval import resolve_sparse_encoder

# Reuses the SAME ADR-0053 composition the pipeline ran at index time, so the query and the corpus
# land in one sparse space. Same config as the index-time SparseEmbedder; no model is pinned.
encoder = resolve_sparse_encoder({"provider": "query_sparse.splade", "config": {
    "model": "your-org/cleared-multilingual-splade",   # weights AND code license verified (ADR-0012)
    "metadata_fields": ["classification"],
}})
sparse_vector = encoder.encode(
    "quarterly revenue growth",
    entities=[("acme corp", 0.94)],                    # confidence-gated query entities (optional)
    metadata={"classification": "public"},             # composed in exactly as at index time
)
# Hand `sparse_vector` (indices + values) to YOUR store as the sparse leg of a hybrid query.
```

The composition is **reused**, not re-implemented — `SpladeQueryEncoder` wraps `latence-splade`'s
`SpladeCompositionEmbedder`, so the query vector is byte-identical to what the corpus was built with.
The real model rides `pip install 'latence-retrieval[sparse]'` (which pulls `latence-splade`, and
only when a real model is wired the `transformers` stack). Offline tests inject the shipped faithful
`FakeSpladeModel`, so query-side sparse is covered without weights.

### Bring your own store: the reference `QdrantBackend` (`[backend-qdrant]` extra)

The tooling never runs first-stage search — your store does (ADR-0048). A `RetrievalBackend` is the
thin adapter over the store you already run; `QdrantBackend` is the **reference example you copy** for
your own engine (Databricks, Azure AI Search, Elastic, pgvector, Weaviate — seam-compatible
follow-ons). It is the seam's real second adapter alongside the offline `FakeBackend`.

```python
from latence_retrieval import Capability, QdrantBackend, RetrievalQuery

# Declare only what your collection can serve; connection kwargs are threaded to QdrantClient.
backend = QdrantBackend({
    "collection": "docs",
    "capabilities": ["dense", "filter"],   # this reference adapter serves {dense, sparse, filter}
    "dimension": 768,                       # the dense index width (required for a dense mode)
    "vector_name": "content",              # the named dense vector to search (None ⇒ unnamed default)
    "text_key": "content",                 # payload field → Candidate.text (default: "content")
    "url": "http://localhost:6333",        # (+ api_key/prefer_grpc/… — built lazily on first search)
})

# One first-stage lookup on YOUR engine → the Candidates the tooling then reranks/packs/fuses.
hits = backend.search(
    RetrievalQuery(mode=Capability.DENSE, vector=my_query_vector, filter={"tenant": "acme"}, top_k=10)
)
```

**Where the text comes from.** `text_key` defaults to `"content"` — the framework's own Export
column name, which the `latence-sink-qdrant` loader writes verbatim into every point's payload. So a
collection *this framework loaded* is queryable with no override: `Candidate.text` is the chunk text
the pipeline exported. Point the adapter at a collection somebody else built and set `text_key` to
whatever key they used; a key the payload does not carry raises `BackendSearchError` naming the keys
that *are* there — never a page of Candidates with silently empty `text` (ADR-0034).

**The pattern to copy** (`src/latence_retrieval/qdrant.py`): declare `capabilities` honestly (never
advertise a mode you cannot serve); enforce the seam invariants yourself **before the wire** — reject
a mode/filter outside `capabilities` (`BackendCapabilityError`) or a wrong-width vector
(`BackendDimensionError`), exactly as your store would; translate one `RetrievalQuery` into a single
store call; map results back to `Candidate`s; and map any native store fault to `BackendSearchError`
(no raw-store-text leak). Hold no index and open no connection until `search`.

The real client rides `pip install 'latence-retrieval[backend-qdrant]'` (`qdrant-client`,
Apache-2.0 — code; a vector store ships **no** model weights, ADR-0012). Protocol conformance is
unit-tested against a faithful fake `qdrant_client` module; live-store validation against a running
Qdrant is done out of band (`LATENCE_QDRANT_URL`).

## Licensing

Apache-2.0. The core has no non-permissive or heavy dependency (only `pydantic`). Heavy query-time
deps sit behind extras: the `[rerank]` extra pulls **`sentence-transformers`** (Apache-2.0 — code:
<https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE>). That library ships **no**
model; a cross-encoder **checkpoint** is config-selected and its weights AND code license verified
separately before it is wired — nothing is hardcoded (ADR-0012 / research-diligence). No model
weights are attested here.

The `[pack]` extra pulls **`tiktoken`** (MIT — code:
<https://github.com/openai/tiktoken/blob/main/LICENSE>) for the *optional* exact token counter; the
`pack` transform runs day one on the pure-Python `HeuristicTokenCounter` with no extra. `tiktoken`
ships its encoding data and **no** model weights; the **encoding** is config-selected and no default
is pinned (ADR-0012). The Knapsack QKP packer itself is ported Apache-2.0 (the maintainer's own
`colsearch`; provenance in `docs/retrieval-tooling.md`) — pure-Python, deterministic on CPU, no
third-party or non-permissive code. The `bm25` rescorer is pure-Python (no extra) — an exact Okapi
BM25 with no third-party code.

The `[sparse]` extra pulls **`latence-splade`** (Apache-2.0), which owns the ADR-0053 SPLADE
composition the query-side encoder reuses. It ships **no** model; a multilingual SPLADE **checkpoint**
is config-selected and its weights AND code license verified separately before it is wired — nothing
is hardcoded (ADR-0012 / research-diligence). A compiled-Cython BM25 accelerator over the same exact
formula is a documented follow-on (like the SPLADE torch-reduce), not shipped here.
