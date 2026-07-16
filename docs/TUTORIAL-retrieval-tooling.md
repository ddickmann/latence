# Tutorial — testing the retrieval-quality tooling (every piece, hands-on, offline-first)

This walkthrough drives **every piece** of the retrieval-quality tooling — the index-time signal
generators and the query-time stateless library — and, for each, gives you three things:

1. **the exact command or snippet** to invoke it,
2. **what output to expect** (copied from a real run), and
3. **how to know it worked** — the verification/test angle, including the audited invariants.

The whole **offline spine** runs today on a laptop with **no GPU, no model weights, no vector store,
no graph database**: the reference signal generators, the faithful fakes, the pure-Python one-liners,
and DuckDB-over-parquet are all self-contained. Anything that needs real infrastructure is clearly
marked **(requires real weights / live store — deferred)** and is never on the offline path.

> **What this tooling is (and is not).** latence provides the *machinery and intelligence for SOTA
> retrieval quality — not the retrieval engine itself* (`docs/retrieval-tooling.md`, ADR-0048). It is
> **strictly stateless**: it *generates signals* (index-time, emitted as files) and *transforms
> candidate lists* (query-time, over hits your own engine already fetched). It holds no index, no
> stored vectors, and runs no first-stage search — that always stays on your store (Qdrant, Weaviate,
> Azure AI Search, Elastic, …). Two clocks (ADR-0049): **index-time** signal generation is part of
> `latence process`; **query-time** tooling is the separate `latence-retrieval` library.

---

## 0. The map — what you will test

| # | Piece | Home | Offline? |
|---|---|---|---|
| §2.1 | SPLADE sparse composition | `sparse.hashing` ref / `sparse.splade` real | ref: yes · real: deferred |
| §2.2 | MUVERA FDE + raw multi-vectors *(experimental)* | `fde.reference`/`fde.muvera` + `multivector.hashing` | yes |
| §2.3 | Graph features (centrality / community) | Graph Assembly `graph_features.*` | yes |
| §2.4 | BM25 term-stats artifact *(optional)* | Export `bm25:` block | yes |
| §2.5 | The two audited index-time invariants | byte-identical-when-unwired · no-PII-re-leak | yes |
| §3 | à-la-carte: fuse / rerank / pack / expand / bm25 / query-sparse | `latence-retrieval` Level 1 | yes (real model deferred) |
| §4 | multi-hop expand (DuckDB / networkx / Neo4j) | `latence-retrieval` multihop | DuckDB/networkx: yes · Neo4j: deferred |
| §5 | the `retrieve()` orchestrator | `latence-retrieval` Level 2 | fake backend: yes · Qdrant: deferred |
| §6 | the MCP server + its hardening | `latence-retrieval.mcp` | yes |
| §7 | the stateless invariant + the full test suite | — | yes |

---

## 1. Setup

The repo is a **`uv` workspace**, so every package is already wired together — you run snippets and
the CLI through `uv run`, which resolves the workspace without a manual install.

```bash
cd /path/to/latence-framework

# sanity: the query-time library, the SPLADE composition package, and core all import
uv run python -c "import latence_retrieval, latence_splade, latence_muvera, latence_core; print('imports OK')"
# → imports OK

# the CLI (from latence-core) is on the uv path
uv run latence --help | head -5
```

**Which extra enables which piece.** The `latence-retrieval` core is **pure-Python** (only
`pydantic`); every heavy query-time dependency sits behind an extra, so you install only what a
transform needs. The whole **offline spine below needs none of them** — it runs on the pure core plus
the shipped *faithful fakes*. Install an extra only to swap in a real model or engine:

| Extra | Enables | Needed for the offline spine? |
|---|---|---|
| `latence-retrieval[rerank]` | real cross-encoder (`sentence-transformers`) | no — offline uses `FakeCrossEncoder` |
| `latence-retrieval[pack]` | exact `tiktoken` token counter | no — offline uses `HeuristicTokenCounter` |
| `latence-retrieval[sparse]` | query-side SPLADE (`latence-splade`) | no — offline injects a fake `SpladeModel` |
| `latence-retrieval[multivector]` | query-side multi-vector (`latence-core` ref) | no — offline uses `FakeMultiVectorEmbedder` |
| `latence-retrieval[graph-duckdb]` | zero-ops multi-hop over parquet (`duckdb`) | **§4 only** — `pip`/`uv add duckdb` |
| `latence-retrieval[graph-networkx]` | pure-Python multi-hop reference (`networkx`) | §4 alt |
| `latence-retrieval[graph-neo4j]` | optional Neo4j multi-hop | no — deferred |
| `latence-retrieval[mcp]` | the `mcp` SDK wire face | no — §6 drives the tool-call boundary directly |
| `latence-retrieval[backend-qdrant]` | reference Qdrant backend | no — offline uses `FakeBackend` |

> Every snippet below is copy-pasteable and was **run to produce the exact output shown**. If you run
> them yourself and get a byte-different number, that is a signal worth investigating — these are all
> deterministic.

For §4's DuckDB path only, add the one embedded engine (MIT, no server, no weights):

```bash
uv pip install duckdb        # or: uv run --with duckdb python ...
```

---

## 2. Index-time — generating retrieval signals as files

Index-time signal generation **extends the pipeline**: each generator is wired **additively** onto
the RAG-corpus Export (or Graph Assembly) and emits more AI-ready columns/files during
`latence process`. Because a full pipeline run needs a corpus + config, the smallest self-contained
path that exercises the real emission end-to-end is **`latence stack validate`**, which stages the
bundled deterministic corpus, runs a named stack, asserts the G1 checks (contracts, KG, RAG, PII,
determinism, resume, device-honesty), and — when you pin `--storage-root` — **leaves the exported
artifacts on disk for you to inspect**.

Two stacks ship pre-wired with the signal generators:

- `stacks/retrieval-signals.yaml` — the default spine **plus** `sparse.hashing` + `multivector.hashing`
  → `fde.reference` + the raw multi-vector sidecar, all on `export_corpus`.
- `stacks/graph-features.yaml` — the default spine **plus** `graph_features.degree` on Graph Assembly
  and a `context.kg_header` enrich stage.

Run the signals stack and keep the artifacts:

```bash
rm -rf /tmp/sig
uv run latence stack validate stacks/retrieval-signals.yaml --storage-root file:///tmp/sig
```

Expected tail — every check **PASS**:

```text
# Stack validation — `retrieval-signals.yaml` — **PASS**
| check | result | detail |
|---|---|---|
| contracts | PASS | completeness 48/48 prov, 48/48 class, offsets_aligned=True, drift=clean |
| kg | PASS | nodes=6, edges=33, evidence_coverage=1.00, exported=True |
| rag | PASS | 4 corpus rows across 1 JSONL export(s), no PII leak |
| ...
| determinism | PASS | 7 export artifact(s) byte-identical across two fresh runs |
| resume | PASS | 7 export artifact(s) byte-identical after same-run_id resume |
```

The stack runs **twice** (a `stack-validate` run and a `stack-validate-b` run) for the determinism
check, so the artifacts live under both. Point a shell variable at the first run's export dir:

```bash
EX=/tmp/sig/_latence/runs/stack-validate/export
ls "$EX"
#   corpus.jsonl  corpus.parquet  graph-nodes.parquet  graph-edges.parquet  multivectors.parquet ...
```

### 2.1 SPLADE sparse composition → `sparse_indices` / `sparse_values`

**How it's wired** (`stacks/retrieval-signals.yaml`, on `export_corpus`):

```yaml
      sparse_embedder:
        provider: sparse.hashing          # the CPU reference; the real model is sparse.splade
        config: {vocab_size: 1024}
```

**Inspect the emitted columns** — two parallel columns a store's sparse index ingests directly:

```bash
uv run python - "$EX/corpus.parquet" <<'PY'
import sys, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
print("columns:", t.column_names)
r = {k: t.column(k)[0].as_py() for k in t.column_names}
print("sparse_indices[0][:8]:", r["sparse_indices"][:8])
print("sparse_values[0][:8]:", r["sparse_values"][:8])
PY
```

Expected:

```text
columns: ['record_id', 'schema_version', 'provenance', 'classification', 'content', 'media_type', 'embedding', 'sparse_indices', 'sparse_values', 'fde_embedding']
sparse_indices[0][:8]: [7, 37, 154, 158, 160, 175, 264, 333]
sparse_values[0][:8]: [2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0]
```

**How to know it worked:** `sparse_indices` are **strictly ascending** (the canonical `SparseVector`
order) and the same length as `sparse_values`; the columns appear only because the `sparse_embedder`
was wired (see §2.5 for the byte-identical-when-unwired proof).

> **The real model — (requires real weights — deferred).** `sparse.hashing` is the deterministic,
> zero-dep **reference** that exercises the seam + emission offline. The blessed real signal is the
> ADR-0053 **structured composition** — `model_weight · SPLADE(text) ⊕ entity_boost · confidence-gated
> entity terms ⊕ metadata_boost · metadata terms` — in the pluggable `latence-splade` package
> (`sparse.splade`), which pins **no** checkpoint until its weights+code license is cleared
> (ADR-0012/0053). Swapping is a one-line config change (`provider: sparse.splade`,
> `config: {model: <cleared-multilingual-splade-id>}`). You test that composition *offline* with a
> fake model in §3.6.

### 2.2 MUVERA-FDE + raw multi-vectors *(experimental, off by default)*

**How it's wired** (same `export_corpus`):

```yaml
      multivector_embedder:
        provider: multivector.hashing
        config: {dimension: 8}
      fde_converter:
        provider: fde.reference             # the real MUVERA is fde.muvera (also pure-Python)
        config: {input_dimension: 8, buckets: 4}
      keep_multivectors: true               # also emit the raw multi-vector sidecar
```

**Inspect the FDE dense column** (lands in your customer's ordinary **dense** ANN index) — its width
is `buckets × input_dimension = 4 × 8 = 32`:

```bash
uv run python - "$EX/corpus.parquet" <<'PY'
import sys, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
r = {k: t.column(k)[0].as_py() for k in t.column_names}
print("fde_embedding width:", len(r["fde_embedding"]))
PY
# → fde_embedding width: 32
```

**Inspect the raw multi-vector sidecar** — a separate parquet keyed by `record_id`, one row per chunk,
each a `list<list<float>>` (one token vector per token):

```bash
uv run python - "$EX/multivectors.parquet" <<'PY'
import sys, pyarrow.parquet as pq
for r in pq.read_table(sys.argv[1]).to_pylist():
    mv = r["multivector"]
    if mv:
        print("record_id …", r["record_id"][-12:], "| token-vectors:", len(mv), "| width:", len(mv[0]))
        break
PY
# → record_id … 2cfc#chunk-0 | token-vectors: 39 | width: 8
```

**How to know it worked:** the FDE width equals `buckets × input_dimension`; the sidecar column is
`multivector` (a short/empty chunk legitimately has 0 token-vectors, a longer one has one per token).

> **The real MUVERA is *also* offline-runnable** — `fde.muvera` (`latence-muvera`) is **data-oblivious
> and training-free**, so it carries **no weights** (Apache-2.0, pure-Python). It emits the *same*
> shape as the reference but its dot-product genuinely **approximates MaxSim**. Drive it directly,
> offline, and watch a single dense dot recover the late-interaction score (`max_sim` is the exact
> reference):

```bash
uv run python - <<'PY'
from latence_muvera import MuveraFdeConverter
from latence_retrieval.testing import FakeMultiVectorEmbedder
from latence_retrieval.maxsim import max_sim

enc  = FakeMultiVectorEmbedder(dimension=8)
conv = MuveraFdeConverter({"input_dimension": 8, "buckets": 4})
print("FDE dimension:", conv.dimension)                    # 1 rep × 4 buckets × 8 = 32

qmv     = enc.encode("alpha beta")
dmv_rel = enc.encode("alpha beta gamma")
dmv_irr = enc.encode("delta")
(fq,)   = list(conv.encode_query([qmv]))                   # query FDE: sum-per-cell
(fr,)   = list(conv.convert([dmv_rel]))                    # doc FDE:  mean + fill-empty
(fi,)   = list(conv.convert([dmv_irr]))
dot = lambda a, b: round(sum(x*y for x, y in zip(a, b)), 4)
print("relevant:  ", dot(fq, fr), "vs exact MaxSim", max_sim(qmv, dmv_rel))
print("irrelevant:", dot(fq, fi), "vs exact MaxSim", max_sim(qmv, dmv_irr))
PY
```

Expected — the FDE dot tracks the exact MaxSim, ranking the relevant doc far above the irrelevant:

```text
FDE dimension: 32
relevant:   2.0 vs exact MaxSim 2.0
irrelevant: 0.5573 vs exact MaxSim 0.55728027
```

> **Experimental (ADR-0050):** the multi-vector / FDE path is **real and evaluable but off by default**
> and blesses no per-corpus quality guarantee. It appears *only* when you wire the three keys — the
> default export never carries `fde_embedding` or the sidecar (§2.5).

### 2.3 Graph features — centrality / community

**How it's wired** (`stacks/graph-features.yaml` — on Graph Assembly, then projected by `enrich`):

```yaml
  - name: graph
    provider: graph.canonical
    config:
      corpus_id: latence-graph-features-stack
      graph_features:
        provider: graph_features.degree     # or graph_features.pagerank
        config: {}
```

Run it and inspect the promoted **graph-nodes.parquet** columns:

```bash
rm -rf /tmp/gf
uv run latence stack validate stacks/graph-features.yaml --storage-root file:///tmp/gf
GF=/tmp/gf/_latence/runs/stack-validate/export
uv run python - "$GF/graph-nodes.parquet" <<'PY'
import sys, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
print("columns:", t.column_names)
for r in t.to_pylist()[:3]:
    print(" ", r["canonical_name"], "| centrality:", r["centrality"], "| community:", r["community"])
PY
```

Expected — `centrality` + `community` are first-class node columns:

```text
columns: ['node_id', 'entity_id', 'type', 'canonical_name', 'kb_id', 'confidence', 'member_mention_ids', 'source_document_ids', 'properties', 'evidence', 'centrality', 'community']
  Carol | centrality: 0.6 | community: 4182b585a62932e3
  Acme | centrality: 1.0 | community: 4182b585a62932e3
  Alice | centrality: 0.6 | community: 4182b585a62932e3
```

The same features are **projected onto each chunk's `context_header`** — the columns the query-time
`expand` (§3.4) consumes as candidate metadata:

```bash
uv run python - "$GF/corpus.jsonl" <<'PY'
import sys, json
for line in open(sys.argv[1]):
    ch = json.loads(line).get("context_header")
    if ch:
        for k in ("kg_node_ids", "neighbor_node_ids", "kg_node_centrality", "kg_node_community"):
            print(f"{k}: {ch.get(k)}")
        break
PY
```

Expected:

```text
kg_node_ids: ['472c4fba9ce11e84', '6577c6614a8f440a', 'f2be8e59ceec975c', '4182b585a62932e3', 'c80c81d3cddaeae9', 'dae78ac7c3428fee']
neighbor_node_ids: []
kg_node_centrality: [1.0, 0.6, 0.6, 0.6, 1.0, 1.0]
kg_node_community: ['4182b585a62932e3', '4182b585a62932e3', '4182b585a62932e3', '4182b585a62932e3', '4182b585a62932e3', '4182b585a62932e3']
```

**How to know it worked:** `centrality` ∈ (0, 1] and `community` is a content-addressed id on every
node; the chunk `context_header` carries `kg_node_ids` + `kg_node_centrality` **aligned per entity**.
This is the exact metadata `expand`/`multihop_expand` read at query time — the index→query bridge.

> **The audit fixed a real wired-but-dead gap here** (R4, `e59a89c`): ADR-0051 graph features were
> computed and stamped but never surfaced as Export columns, so the whole graph-augmented seam was
> dead. The end-to-end column path is now covered by
> `packages/latence-retrieval/tests/test_export_graph_columns_e2e.py`.

### 2.4 BM25 term-stats artifact *(optional)*

An **artifact, never a first-stage index** (ADR-0048): index-time corpus statistics you feed to the
query-time BM25 rescorer (§3.5) so its IDF is exact-global instead of pool-local. Wire it by adding a
`bm25:` block to `export_corpus` (its varying seam is a Tokenizer):

```yaml
      bm25:
        provider: tokenizer.regex           # or tokenizer.whitespace
        config: {}
```

To test it without editing a committed stack, append the block to a copy and validate the copy:

```bash
cp stacks/retrieval-signals.yaml /tmp/bm25-stack.yaml
cat >> /tmp/bm25-stack.yaml <<'YAML'
      bm25:
        provider: tokenizer.regex
        config: {}
YAML
rm -rf /tmp/bm25
uv run latence stack validate /tmp/bm25-stack.yaml --storage-root file:///tmp/bm25
STATS=$(find /tmp/bm25 -name bm25-stats.json | head -1)
uv run python - "$STATS" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
print("keys:", sorted(d))
print("doc_count:", d["doc_count"], "| avgdl:", d["avgdl"], "| total_tokens:", d["total_tokens"])
print("sample terms:", d["terms"][:3])
PY
```

Expected — `bm25-stats.json` (corpus stats + precomputed Robertson IDF) and a
`bm25-postings.parquet` sidecar are emitted:

```text
keys: ['artifact', 'avgdl', 'doc_count', 'terms', 'tokenizer', 'total_tokens', 'version']
doc_count: 4 | avgdl: 30.5 | total_tokens: 122
sample terms: [{'df': 2, 'idf': 0.69314718, 'term': 'a'}, {'df': 4, 'idf': 0.10536052, 'term': 'acme'}, {'df': 1, 'idf': 1.2039728, 'term': 'acquisition'}]
```

**How to know it worked** — the artifact feeds the query-time rescorer for *exact-global* BM25:

```bash
uv run python - "$STATS" <<'PY'
import sys, json
from latence_retrieval import Candidate, Bm25Rescorer
d = json.load(open(sys.argv[1]))
stats = {"doc_count": d["doc_count"], "avg_doc_len": d["avgdl"],
         "doc_freq": {e["term"]: e["df"] for e in d["terms"]}}
out = Bm25Rescorer({"stats": stats}).process("acme ibm", [
    Candidate(id="x", score=0.0, text="acme partnered with ibm"),
    Candidate(id="y", score=0.0, text="a note about lunch"),
])
print("exact-global rescore:", [(c.id, round(c.score, 4)) for c in out])
PY
# → exact-global rescore: [('x', 1.3111), ('y', 0.0)]
```

### 2.5 The two audited index-time invariants — verify them yourself

These two properties are *load-bearing* and were both breached-and-fixed during the hostile audit, so
they are the first things to check on any change.

**(a) Byte-identical export when a generator is unwired (ADR-0017/0049).** Run the plain default stack
and confirm its corpus carries **none** of the signal columns — the schema is exactly the base export:

```bash
rm -rf /tmp/base
uv run latence stack validate stacks/default.yaml --storage-root file:///tmp/base
uv run python - /tmp/base/_latence/runs/stack-validate/export/corpus.parquet <<'PY'
import sys, pyarrow.parquet as pq
print("default columns:", pq.read_table(sys.argv[1]).column_names)
PY
```

Expected — the six base columns only; `embedding` / `sparse_*` / `fde_embedding` are **absent**:

```text
default columns: ['record_id', 'schema_version', 'provenance', 'classification', 'content', 'media_type']
```

So the signals in §2.1–§2.3 are strictly **additive** — wiring a generator adds columns and changes
nothing else; unwiring it returns a byte-identical export. The `determinism` + `resume` PASS lines in
every `stack validate` run assert this byte-identity across two fresh runs directly. The unit proof is
`packages/latence-core/tests/test_retrieval_signal_export.py::test_no_generator_wired_is_byte_identical`.

**(b) Signals never re-leak redacted PII (ADR-0053).** Every generator runs over the **redacted**
corpus text (`export_corpus` depends on `redact`), so a masked name/IBAN can never re-enter a sparse
term, an FDE cell, or the context header. Each `stack validate` run's `rag` and `report` checks scan
**every** exported artifact — the JSONL/Parquet corpus, the new signal columns, the sidecar, and the
KG — for the planted raw PII and PASS only when none appears:

```text
| rag | PASS | 4 corpus rows across 1 JSONL export(s), no PII leak |
| report | PASS | schema v19, 13 stage metrics, profile_columns=True, no PII leak |
```

> The audit fixed **two** real PII re-leaks on exactly these paths: the sparse signal once encoded the
> *un-redacted* context header (R1, `16efa09`), and the Export once wrote the raw ContextHeader
> (canonical names + relation triples) into inspectable corpus cells (R5, `ee8b2d2`). Both are now
> covered by the redaction-gate tests in `test_retrieval_signal_export.py` and
> `test_context_enrichment_export.py`.

---

## 3. Query-time — à-la-carte one-liners (Level 1)

Everything below is **pure-Python, offline, and stateless**: you build `Candidate`s by hand (standing
in for what *your* engine returned), call one transform, and read the result. A `Candidate` is frozen
(`id`, `score`, `text`, `metadata`) — a processor never mutates it, and identical inputs always yield
an identical output (ties broken by ascending `id`).

### 3.1 fuse — merge N ranked lists into one (RRF + weighted)

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, RrfFuser, WeightedFuser

dense = [Candidate(id="a", score=0.91), Candidate(id="b", score=0.83), Candidate(id="c", score=0.55)]
bm25  = [Candidate(id="c", score=12.4), Candidate(id="a", score=9.1), Candidate(id="d", score=2.0)]

print("RRF:     ", [(c.id, round(c.score, 5)) for c in RrfFuser().fuse([dense, bm25])])
print("weighted:", [(c.id, round(c.score, 4)) for c in WeightedFuser().fuse([dense, bm25], weights=[2.0, 1.0])])
PY
```

Expected:

```text
RRF:      [('a', 0.03252), ('c', 0.03227), ('b', 0.01613), ('d', 0.01587)]
weighted: [('a', 2.6827), ('b', 1.5556), ('c', 1.0), ('d', 0.0)]
```

**What to assert:** `a` and `c` (in *both* lists) rise to the top; disjoint ids (`b`, `d`) still
appear. RRF reads rank only (scale-free, so the incomparable 0.55 and 12.4 scales fuse cleanly);
weighted min-max-normalises each list then sums, so the dense list's 2× weight pulls `a` up.

### 3.2 rerank — re-score with a cross-encoder (faithful fake offline)

The reranker takes a `CrossEncoderModel` by injection. Offline you drive the shipped **`FakeCrossEncoder`**
(a deterministic lexical-overlap proxy that mirrors the real API's contract — one score per text, in
order, string-only, empty→empty):

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Reranker
from latence_retrieval.testing import FakeCrossEncoder

cands = [
    Candidate(id="a", score=0.4, text="the annual report covers revenue and tax"),
    Candidate(id="b", score=0.9, text="a recipe for sourdough bread"),
    Candidate(id="c", score=0.5, text="quarterly revenue grew and tax fell"),
]
out = Reranker(model=FakeCrossEncoder()).process("revenue and tax", cands)
print("rerank:", [(c.id, c.score) for c in out])
PY
# → rerank: [('a', 3.0), ('c', 3.0), ('b', 0.0)]
```

**What to assert:** the two revenue/tax chunks (`a`, `c`) leapfrog the higher-first-stage-scored but
irrelevant `b` — the reranker *replaces* the score with the cross-encoder's. `a` and `c` tie at 3.0
and break by ascending `id`.

> **The real model — (requires real weights — deferred).** In production you pass no `model=` and
> instead configure `{"model": "<cleared-cross-encoder-id>"}` (ADR-0012: **no checkpoint is pinned**);
> `pip install 'latence-retrieval[rerank]'` pulls `sentence-transformers`. Construction stays cheap —
> weights load lazily on first `process`.

### 3.3 pack — token-budgeted subset (the Knapsack QKP context-packer)

`pack` selects the subset maximising `value − redundancy` under a token budget. Day one it uses the
dependency-free `HeuristicTokenCounter` (no config, no download):

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Packer
cands = [
    Candidate(id="a", score=0.4, text="the annual report covers revenue and tax"),
    Candidate(id="b", score=0.9, text="a recipe for sourdough bread"),
    Candidate(id="c", score=0.5, text="quarterly revenue grew and tax fell"),
]
print("pack(budget=12):", [c.id for c in Packer({"budget": 12}).process("q", cands)])
PY
# → pack(budget=12): ['b', 'c']
```

**What to assert:** the packer respects the budget (it did not take all three) and the returned subset
is ordered by descending first-stage score. Note `pack` **selects, it does not re-score** — run
`rerank` *before* `pack` if you want a stronger relevance signal.

**Graph-augmented pack** — a per-entity `kg_node_centrality` list (the §2.3 column) is reduced to its
**peak** and added as a value signal (this is the exact shape the audit's R5 fix made packable):

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Packer
cands = [
    Candidate(id="a", score=0.5, text="alpha", metadata={"kg_node_centrality": [0.9, 0.2]}),
    Candidate(id="b", score=0.5, text="beta",  metadata={"kg_node_centrality": [0.1]}),
]
out = Packer({"budget": 50, "value_signals": {"kg_node_centrality": 1.0}}).process("q", cands)
print("pack + centrality:", [c.id for c in out])
PY
# → pack + centrality: ['a', 'b']
```

> **The exact tokenizer — (optional).** `{"tokenizer": {"encoding": "<name>"}}` selects the real
> `tiktoken` counter (`pip install 'latence-retrieval[pack]'`, MIT). Omit it for the heuristic.

### 3.4 expand — 1-hop graph-augmented rescoring (no graph DB)

`expand` reads the denormalised KG columns the pipeline projected onto each chunk (§2.3) and lifts
candidates that are 1-hop-linked to the list's own top hits — a **pure metadata operation**, no graph
database, works on any store:

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Expander
cands = [
    Candidate(id="a", score=0.9, text="x", metadata={"kg_node_ids": ["n1"], "neighbor_node_ids": ["n2"]}),
    Candidate(id="b", score=0.8, text="y", metadata={"kg_node_ids": ["n2"], "neighbor_node_ids": ["n9"]}),
    Candidate(id="c", score=0.7, text="z", metadata={"kg_node_ids": ["n5"], "neighbor_node_ids": []}),
]
out = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", cands)
print("expand:", [(c.id, c.score) for c in out])
PY
# → expand: [('b', 1.3), ('a', 0.9), ('c', 0.7)]
```

**What to assert:** the seed is `a` (top score). `b`'s node `n2` sits in `a`'s `neighbor_node_ids`, so
`b` earns `hop_weight` (0.8 → 1.3) and is *lifted above* the seed; `c` (unconnected) is unchanged.
Feed a list with **no** KG columns and `expand` is an order-preserving no-op (graceful degradation); a
*malformed* KG column fails **loudly** at the boundary (the anti-false-green contract).

### 3.5 bm25 — exact Okapi BM25 rescoring (never a first-stage index)

Rescore the candidate set you already have — no second index stood up. With no `stats`, the IDF is
*pool-local* (the candidate pool is its own corpus); feed the §2.4 artifact's stats for *exact-global*:

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Bm25Rescorer
cands = [
    Candidate(id="a", score=0.4, text="the annual report covers revenue and tax"),
    Candidate(id="b", score=0.9, text="a recipe for sourdough bread"),
    Candidate(id="c", score=0.5, text="quarterly revenue grew and tax fell"),
]
print("bm25:", [(c.id, round(c.score, 4)) for c in Bm25Rescorer().process("revenue tax", cands)])
PY
# → bm25: [('c', 0.94), ('a', 0.8744), ('b', 0.0)]
```

**What to assert:** the two term-bearing docs score positive and `b` (no query term) stays exactly
`0.0`. The IDF is the **non-negative** Lucene/Robertson form — the audit fixed a `df > doc_count`
path that produced a *negative* IDF and silently inverted ranking (R2, `4a3b6f2`); a nonsensical
`stats` block now fails loudly at construction.

### 3.6 query-signal-gen — the query-side SPLADE composition (offline, fake model)

At query time the **same** ADR-0053 composition must run over the query so it lands in the corpus's
sparse space (one composition, two clocks). `SpladeQueryEncoder` *reuses* the index-time embedder, and
offline you inject a torch-free fake `SpladeModel` (the same shape `latence-splade`'s tests use):

```bash
uv run python - <<'PY'
import hashlib, re
from latence_retrieval import SpladeQueryEncoder

_TOKEN = re.compile(r"[a-z0-9]+"); _VOCAB = 30000
def _tid(t): return int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big") % _VOCAB

class FakeSpladeModel:
    """A torch-free SpladeModel double: term-frequency weights + subword-ish id folding."""
    vocab_size = _VOCAB
    def term_weights(self, texts):
        out = []
        for t in texts:
            m = {}
            for tok in _TOKEN.findall(t.lower()): m[_tid(tok)] = m.get(_tid(tok), 0.0) + 1.0
            out.append(m)
        return out
    def term_ids(self, term):
        seen, ids = set(), []
        for tok in _TOKEN.findall(term.lower()):
            i = _tid(tok)
            if i not in seen: seen.add(i); ids.append(i)
        return ids

enc = SpladeQueryEncoder({"entity_boost": 3.0}, model=FakeSpladeModel())
vec = enc.encode("revenue tax filing", entities=[("tax", 0.95)])
terms = dict(zip(vec.indices, vec.values))
print("query sparse terms:", terms)
print("'tax' id weight (1.0 model + 3.0 entity boost):", terms[_tid("tax")])
PY
```

Expected — the confidence-gated `tax` entity boosts its own term id by `entity_boost`:

```text
query sparse terms: {8516: 1.0, 17741: 4.0, 29627: 1.0}
'tax' id weight (1.0 model + 3.0 entity boost): 4.0
```

**What to assert:** the returned `SparseVector` is what you hand your store as the sparse leg of a
hybrid query; the entity term (confidence 0.95 ≥ the 0.5 gate) added its boost, while a weak entity
would have been dropped. You wire it by config with `resolve_sparse_encoder({"provider":
"query_sparse.splade", "config": {...}})` — the name is kept distinct from the index-time
`sparse.splade` Provider so the two seams never collide.

### 3.7 maxsim — experimental late-interaction rerank

Re-score by exact MaxSim over the raw `multivectors.parquet` sidecar (§2.2) the caller loaded onto each
candidate. Offline, drive the faithful `FakeMultiVectorEmbedder` (a token → deterministic unit-vector
map, so MaxSim over a doc sharing *k* query tokens ≈ *k*):

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, MaxSimReranker
from latence_retrieval.testing import FakeMultiVectorEmbedder
enc = FakeMultiVectorEmbedder(dimension=8)
docs = [
    Candidate(id="a", score=0.1, text="", metadata={"multivectors": enc.encode("alpha beta gamma")}),
    Candidate(id="b", score=0.9, text="", metadata={"multivectors": enc.encode("delta")}),
]
print("maxsim:", [(c.id, c.score) for c in MaxSimReranker(encoder=enc).process("alpha beta", docs)])
PY
# → maxsim: [('a', 2.0), ('b', 0.55728027)]
```

**What to assert:** the doc sharing both query tokens scores ≈ 2.0 and overtakes the higher-first-stage
`b`. A candidate **missing** its sidecar, or one whose token width mismatches the query, raises a typed
`MaxSimError` — never a fabricated score (ADR-0050 anti-false-green). Experimental, off by default.

### 3.8 The config-driven registry — swap any component by name

Every Level-1 component is discoverable and resolvable by a `{"provider", "config"}` block through the
`latence.retrieval` registry (parallel to, and disjoint from, the pipeline's `latence.providers`):

```bash
uv run python - <<'PY'
from latence_retrieval import RetrievalRegistry, resolve_fuser, resolve_processor
print("registered:", RetrievalRegistry().names())
print("fuser  ->", type(resolve_fuser({"provider": "fuser.rrf"})).__name__)
print("proc   ->", type(resolve_processor({"provider": "processor.bm25", "config": {"k1": 1.2}})).__name__)
PY
```

Expected:

```text
registered: ['fuser.rrf', 'fuser.weighted', 'processor.bm25', 'processor.expand', 'processor.identity', 'processor.maxsim', 'processor.multihop_expand', 'processor.pack', 'processor.rerank', 'query_multivector.hashing', 'query_sparse.splade']
fuser  -> RrfFuser
proc   -> Bm25Rescorer
```

**What to assert:** `resolve_*` seam-checks the built component — a component that does not implement
its seam (e.g. a processor handed to `resolve_fuser`) fails at **wiring** time, never silently at query
time.

---

## 4. Query-time — multi-hop expand (the graph, queried read-only)

Where 1-hop `expand` (§3.4) needs no graph, *multi-hop* rewards edge-path closeness only a real
traversal reveals — so it reads the emitted `graph-edges.parquet` through the `GraphSource` seam. The
load-bearing decision (ADR-0051): it stays **zero-ops** — the graph is queried read-only, per call,
over the parquet the pipeline already emitted, with **no server**.

### 4.1 The faithful fake (always offline) — plant a graph, assert the lift

`FakeGraphSource` *is* the canonical traversal semantics (undirected BFS, minimum hop distance, seeds
excluded) — the reference every real engine is proven equal to:

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, MultiHopExpander
from latence_retrieval.testing import FakeGraphSource

gs = FakeGraphSource([("n1", "n2"), ("n2", "n3"), ("n3", "n4")])   # a path n1-n2-n3-n4
print("neighbors(n1, max_hops=2):", gs.neighbors(["n1"], 2))

cands = [
    Candidate(id="seed", score=0.9, text="", metadata={"kg_node_ids": ["n1"]}),
    Candidate(id="one",  score=0.5, text="", metadata={"kg_node_ids": ["n2"]}),
    Candidate(id="two",  score=0.4, text="", metadata={"kg_node_ids": ["n3"]}),
    Candidate(id="far",  score=0.3, text="", metadata={"kg_node_ids": ["n4"]}),
]
mh = MultiHopExpander({"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 2, "hop_decay": 0.5}, graph_source=gs)
print("multihop:", [(c.id, round(c.score, 4)) for c in mh.process("q", cands)])
PY
```

Expected — a 1-hop neighbour earns `hop_weight`, a 2-hop `hop_weight × hop_decay`, a 3-hop nothing
(bounded at `max_hops=2`):

```text
neighbors(n1, max_hops=2): {'n2': 1, 'n3': 2}
multihop: [('one', 1.0), ('seed', 0.9), ('two', 0.65), ('far', 0.3)]
```

### 4.2 The zero-ops DuckDB path — over the emitted parquet

Install the one embedded engine (`uv pip install duckdb`), write a `graph-edges.parquet` in the exact
shape the pipeline emits, and traverse it read-only — proving equivalence to the reference:

```bash
uv run python - <<'PY'
import pyarrow as pa, pyarrow.parquet as pq
from latence_retrieval import Candidate, MultiHopExpander, DuckDBGraphSource
from latence_retrieval.multihop import _traverse

edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n1")]   # cyclic + high-degree
pq.write_table(pa.table({"source_node_id": [s for s,_ in edges],
                         "target_node_id": [t for _,t in edges]}), "/tmp/edges.parquet")

src = DuckDBGraphSource("/tmp/edges.parquet")
print("duckdb   :", src.neighbors(["n1"], 2))
print("reference:", _traverse(edges, ["n1"], 2))     # must agree

mh = MultiHopExpander({"seed_top_k": 1, "max_hops": 2,
                       "graph": {"source": "duckdb", "edges_uri": "/tmp/edges.parquet"}})
cands = [Candidate(id="seed", score=0.9, text="", metadata={"kg_node_ids": ["n1"]}),
         Candidate(id="one",  score=0.5, text="", metadata={"kg_node_ids": ["n2"]}),
         Candidate(id="two",  score=0.4, text="", metadata={"kg_node_ids": ["n3"]})]
print("duckdb multihop:", [(c.id, round(c.score,4)) for c in mh.process("q", cands)])
PY
```

Expected — DuckDB agrees with the reference traversal (order aside), and drives the same rescoring:

```text
duckdb   : {'n3': 2, 'n2': 1}
reference: {'n2': 1, 'n3': 2}
duckdb multihop: [('one', 1.0), ('seed', 0.9), ('two', 0.65)]
```

**How to know it worked:** the DuckDB result set equals the reference `_traverse` on the same graph.

> **The audit hardened the traversal against a DoS** (R1, `042688a`): real KGs are cyclic and
> high-degree, and a naive recursive CTE with `UNION ALL` enumerates every path (≈`degree**max_hops`
> work). The engine uses `UNION` (set) so each `(node, hop)` is expanded once — bounded, polynomial
> work. The `max_hops` ceiling (64) rejects an unbounded traversal at the boundary.

> **networkx alternative (offline):** `{"source": "networkx", "edges_uri": ...}` with
> `pip install 'latence-retrieval[graph-networkx]'` — the pure-Python reference for small graphs, same
> contract. **Neo4j — (requires a live store — deferred):** `{"source": "neo4j", "uri":
> "bolt://…", …}` runs a `shortestPath` Cypher query against a Neo4j you already operate (never one the
> tooling stands up); `pip install 'latence-retrieval[graph-neo4j]'`.

---

## 5. Query-time — the `retrieve()` orchestrator (Level 2)

The optional Level-2 orchestrator chains `fetch-per-modality → fuse → (your processors) → pack`
around *your* engine, composing **only the legs the backend declares and the query has a signal for**,
dropping the rest gracefully (ADR-0052). Offline you drive the faithful **`FakeBackend`** — a
capability-declaring double that rejects **exactly** what a real store rejects (an undeclared mode, a
wrong-width vector).

```bash
uv run python - <<'PY'
from latence_retrieval import (Candidate, Capability, QueryBundle,
                               OrchestrationPipeline, retrieve, Packer)
from latence_retrieval.testing import FakeBackend

corpus  = [Candidate(id=f"d{i}", score=1.0 - i/10, text=f"doc {i} about revenue") for i in range(6)]
backend = FakeBackend(corpus, capabilities={Capability.BM25, Capability.DENSE, Capability.FILTER}, dimension=4)

# The bundle carries text (a bm25 leg) AND a dense vector — but NO sparse_terms.
bundle = QueryBundle(text="revenue", vector=(0.1, 0.2, 0.3, 0.4))
pipe   = OrchestrationPipeline(processors=(Packer({"budget": 40}),), final_top_k=3)
res    = retrieve(bundle, backend, pipe)
print("fetched_modes:", [m.value for m in res.fetched_modes])
print("results:", [(c.id, round(c.score, 4)) for c in res.candidates])

# A text-only bundle: the dense leg is dropped (no vector) — only bm25 fires.
res2 = retrieve(QueryBundle(text="revenue"), backend, OrchestrationPipeline())
print("text-only fetched_modes:", [m.value for m in res2.fetched_modes])
PY
```

Expected — two legs fire and are RRF-fused (hence the small fused scores); dropping the vector drops
the dense leg:

```text
fetched_modes: ['bm25', 'dense']
results: [('d0', 0.0328), ('d1', 0.0323), ('d2', 0.0317)]
text-only fetched_modes: ['bm25']
```

**How to know it worked:** `fetched_modes` is the observable proof of the capability-adaptive
composition — a mode the store lacks, or one the query has no signal for, is *absent* from it. Note a
**sparse** leg never fired even though the backend could serve one, because the bundle carried no
`sparse_terms`. A filter the backend cannot serve is a loud `OrchestrationError` (never silently
dropped → that would return unfiltered hits).

> **The reference production backend — (requires a live Qdrant — deferred).** `QdrantBackend`
> (`pip install 'latence-retrieval[backend-qdrant]'`) is the real seam adapter over a running Qdrant
> ≥1.10. It imports without the client (lazy) and satisfies the identical `RetrievalBackend` Protocol,
> so the orchestrator/MCP code above is *unchanged* against it — only the fetch reaches a live store.
> Validate it out of band against a real Qdrant.

---

## 6. The MCP server — start it, call a tool, run the security checks

The MCP server is Level 2 with an MCP face: capability-adaptive, config-driven agent copy, and the
full `memory_engine` hardening (bearer auth, rate-limit, leak-free envelopes) — **local/loopback +
bearer only** in v1 (ADR-0052). Its **tool-call boundary** (`call` / `list_tools`) is the whole test
surface, so you exercise all of it offline without the `mcp` wire SDK.

```bash
uv run python - <<'PY'
from latence_retrieval import Candidate, Capability, OrchestrationPipeline
from latence_retrieval.expand import Expander
from latence_retrieval.mcp import BearerAuth, RateLimiter, RetrievalServer, ServerCopy
from latence_retrieval.testing import FakeBackend

CORPUS  = [Candidate(id="a", score=0.9, text="alpha beta", metadata={"source": "s1", "page": 1}),
           Candidate(id="b", score=0.7, text="beta delta", metadata={"source": "s2", "page": 2})]
backend = FakeBackend(CORPUS, capabilities={Capability.BM25, Capability.FILTER})
caps    = backend.capabilities

def build(capacity=100, now=None, pipeline=None):
    import time
    return RetrievalServer(backend=backend, copy=ServerCopy.build(caps),
        auth=BearerAuth("s3cret-token"),
        rate_limiter=RateLimiter(capacity=capacity, window=60.0, now=now or time.monotonic),
        pipeline=pipeline)

srv = build()

# The config-driven, capability-tailored tool surface (the escalating ladder)
print("tools:", [(t.name, t.level) for t in srv.list_tools()])

# A valid, authenticated search
ok = srv.call("search", {"query": "beta"}, token="s3cret-token")
print("2 valid search:", ok.ok, "| count:", ok.meta["count"], "| modes:", ok.meta["fetched_modes"])

# --- the audited security checks ---
print("1 auth required   :", srv.call("search", {"query": "beta"}, token="wrong").error.code.value)
print("3 non-ascii token :", srv.call("search", {"query": "beta"}, token="sécret").error.code.value)
print("4 copy-injection  :", srv.call("search", {"query": "beta", "description": "PWN"}, token="s3cret-token").error.code.value)

# 5 rate-limit engages: capacity 1, frozen clock so no refill
clock = {"t": 0.0}; rl = build(capacity=1, now=lambda: clock["t"])
print("5a first call     :", rl.call("search", {"query": "beta"}, token="s3cret-token").ok)
print("5b throttled      :", rl.call("search", {"query": "beta"}, token="s3cret-token").error.code.value)

# 6 no raw store-metadata leak: a processor fault carries the raw metadata; the server contains it
bad = FakeBackend([Candidate(id="x", score=0.9, text="t",
                             metadata={"kg_node_ids": {"SECRET-INTERNAL": 1}})],
                  capabilities={Capability.BM25})
srv6 = RetrievalServer(backend=bad, copy=ServerCopy.build(bad.capabilities),
    auth=BearerAuth("s3cret-token"), rate_limiter=RateLimiter(capacity=100),
    pipeline=OrchestrationPipeline(processors=(Expander({"hop_weight": 0.5}),)))
env = srv6.call("search", {"query": "t"}, token="s3cret-token")
print("6 leak-check      :", env.error.code.value, "| leaked raw value?", "SECRET-INTERNAL" in env.error.message)
PY
```

Expected — every security property holds:

```text
tools: [('search', 'L1'), ('more_like_this', 'L2'), ('get_context', 'L3'), ('list_sources', 'discovery'), ('index_status', 'discovery')]
2 valid search: True | count: 2 | modes: ['bm25']
1 auth required   : unauthorized
3 non-ascii token : unauthorized
4 copy-injection  : invalid_arguments
5a first call     : True
5b throttled      : rate_limited
6 leak-check      : backend_error | leaked raw value? False
```

**How to know each hardening worked:**

- **auth required** — no/invalid token → `unauthorized`, and the tool **never ran** (the fetch is
  gated behind auth).
- **non-ASCII token handled** — `hmac.compare_digest` raises `TypeError` on a non-ASCII `str`; the auth
  compares **UTF-8 bytes** and fails *closed* → a clean `unauthorized`, never an uncaught crash (the
  audit's R2 fix, `2d7f728`).
- **rate-limit engages** — the second call in a frozen-clock window of capacity 1 → `rate_limited`,
  again without reaching the backend.
- **copy not agent-mutable** — an argument named like a copy field (`description`) is rejected as
  `invalid_arguments`; agent input can *never* reach the operator-trusted `ServerCopy`, so tool
  descriptions can't become a prompt-injection surface.
- **no raw store-metadata leak** — a candidate-processor fault whose native message embeds the raw
  candidate metadata is contained to a typed `backend_error` whose message is the **exception type
  only** (`candidate processing failed: TypeError.`) — the raw value never crosses the boundary (the
  audit's R2/R3 fixes, `307e4ab`/`5070fab`).

> **Copy is config-driven and operator-tailored.** `ServerCopy.build(caps)` starts from strong defaults
> auto-tailored to the backend's capabilities (a sparse+dense store reads "hybrid", a dense-only store
> "semantic") and merges validated operator overrides; `ServerCopy.from_file(path, caps,
> base_dir=...)` loads them from a sandboxed JSON file. An unknown tool id / field / parameter fails
> **closed** at load. **Binding — (local only):** `LoopbackTransport` accepts only a loopback host or a
> Unix socket; a public/wildcard host is rejected at construction. Remote transport (OAuth 2.1 / mTLS)
> is deferred. The thin `mcp`-SDK wire face is `pip install 'latence-retrieval[mcp]'`.

---

## 7. Verify the stateless invariant + run the whole suite yourself

**The stateless invariant (ADR-0048)** is structural, not documented — assert it directly:

```bash
uv run python - <<'PY'
import inspect
from latence_retrieval import CandidateProcessor, IdentityProcessor, RetrievalBackend, Candidate
# A processor's ONLY method is process(query, candidates) — no fetch/index/store method exists.
public = {n for n, _ in inspect.getmembers(CandidateProcessor, inspect.isfunction) if not n.startswith("_")}
print("processor public methods:", public)                       # {'process'}
# The one fetch in the library needs a caller-supplied query — the tooling never originates a search.
print("backend.search params:", list(inspect.signature(RetrievalBackend.search).parameters))  # ['self', 'query']
# No state accrues across calls: a second (empty) call reflects only that input.
p = IdentityProcessor(); p.process("q", [Candidate(id="a", score=1.0)])
print("second call retains nothing:", p.process("q", []) == [])   # True
PY
```

Expected:

```text
processor public methods: {'process'}
backend.search params: ['self', 'query']
second call retains nothing: True
```

The full behavioural pins live in `packages/latence-retrieval/tests/test_stateless_seam.py`.

**Run the whole feature test suite** — the exact gate the hostile audit certified on this branch
(1821 passed, 8 skipped; the skips are optional-dependency / heavy-model guards):

```bash
PYTHONHASHSEED=0 LATENCE_CUDA=0 uv run pytest \
  packages/latence-retrieval packages/latence-splade packages/latence-muvera packages/latence-core
```

The three query-time / signal packages alone run in under a second:

```bash
PYTHONHASHSEED=0 LATENCE_CUDA=0 uv run pytest \
  packages/latence-retrieval packages/latence-splade packages/latence-muvera
# → 552 passed, 9 skipped in ~0.8s
```

And the CI/publish gate for the index-time stacks — each asserts contracts, KG, RAG, **no-PII-leak**,
determinism, resume, and device-honesty on the bundled corpus:

```bash
uv run latence stack validate stacks/retrieval-signals.yaml
uv run latence stack validate stacks/graph-features.yaml
# both → **PASS**  (a fresh temp storage dir is used when --storage-root is omitted)
```

---

## Appendix — the whole surface at a glance

| Piece | Entry point / API | Real vs reference | Offline driver |
|---|---|---|---|
| sparse composition | `sparse.splade` / `sparse.hashing` | real (deferred) / ref | fake `SpladeModel` (§3.6) |
| multi-vector | `multivector.hashing` | ref | `FakeMultiVectorEmbedder` |
| MUVERA FDE | `fde.muvera` / `fde.reference` | both offline (no weights) | direct (§2.2) |
| graph features | `graph_features.degree` / `.pagerank` | ref | stack validate (§2.3) |
| bm25 artifact | Export `bm25:` + `tokenizer.regex` | ref | stack validate (§2.4) |
| fuse | `fuser.rrf` / `fuser.weighted` | — | direct (§3.1) |
| rerank | `processor.rerank` | real (deferred) | `FakeCrossEncoder` (§3.2) |
| pack | `processor.pack` | heuristic (offline) / `tiktoken` | direct (§3.3) |
| expand (1-hop) | `processor.expand` | — | direct (§3.4) |
| bm25 rescore | `processor.bm25` | — | direct (§3.5) |
| query sparse | `query_sparse.splade` | real (deferred) | fake model (§3.6) |
| maxsim | `processor.maxsim` | real (offline math) | fake encoder (§3.7) |
| multi-hop | `processor.multihop_expand` | DuckDB/networkx (offline) / Neo4j (deferred) | `FakeGraphSource` / DuckDB (§4) |
| orchestrator | `retrieve()` | — | `FakeBackend` (§5) |
| backend | `QdrantBackend` | real (deferred) | `FakeBackend` |
| MCP server | `RetrievalServer` | — | tool-call boundary (§6) |

**Design scope + decisions:** [`docs/retrieval-tooling.md`](retrieval-tooling.md) and ADRs
[0048](adr/0048-retrieval-tooling-is-strictly-stateless.md)–0053. **Audit trail:**
`issues/audit/SUMMARY.md`.
