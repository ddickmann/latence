# Pitch deck — slide content

*Copy-paste ready. One block = one slide. Headlines are the message, not a topic label.
`[visual]` = what to draw. `→ say:` = the one line to speak; everything else is on-screen text.*

---

## ACT 1 — THE PROBLEM (slides 1–4)

---

### 1 · Title

**Your documents already contain the answer.**
**Your search can't reach it.**

*Turning messy enterprise data into AI-ready knowledge — and retrieval that reasons across documents.*

---

### 2 · Enterprise data is not a dataset

- Scans · PDFs · Office files · text dumps
- Different systems, no shared schema
- Entities named 5 ways
- PII scattered through the body
- **No explicit link between documents**

→ say: *Every RAG project underestimates the last one.*

---

### 3 · The question and the answer live in different documents

`[visual] Doc A —— cites ——▶ Doc B`
`Question mentions A. Answer sits in B.`

- Keyword search → finds A
- Vector search → finds A
- **Nobody follows the arrow**

---

### 4 · This isn't a tuning problem. It's structural.

| | |
|---|---|
| Better embeddings | still ranks by *nearness to the question* |
| Bigger context | still has to *retrieve the right chunk first* |
| Hybrid search | two ways of measuring *the same nearness* |

→ say: *You cannot tune your way to a document you never retrieved.*

---

## ACT 2 — WHAT WE BUILD (slides 5–8)

---

### 5 · One command in. AI-ready data out.

```
latence setup   →  guided wizard, inspects your corpus
latence process →  17-stage pipeline
```

**Out:** RAG corpus + knowledge graph + quality report — **as files**

---

### 6 · Not chunks-with-vectors. Chunks *plus structure*.

**Conventional RAG corpus**
- chunks
- embeddings

**Ours — additionally**
- resolved entities (one node per real thing)
- relations + **cross-document references**
- every edge carries its **evidence sentence**
- every chunk knows **which entities it contains**

---

### 7 · The pipeline

```
PARSE ─ CHUNK ─ SCREEN ─ INDUCE ─ EXTRACT ─ REFERENCES ─ REDACT
                                                            │
                                  ◀── chunk level ──────────┘
                                  ─── corpus level ─────────▶
                                                            │
              RESOLVE ─ GRAPH+CENTRALITY ─ ENRICH ─ EXPORT ──┘
```

**One rule:** everything before redaction is **per chunk**. Everything after is **corpus-wide**.

---

### 8 · Models per stage — all swappable, all permissive

| Stage | Model |
|---|---|
| OCR | LightOn-OCR 2 (VLM, served) |
| Schema induction | any OpenAI-compatible LLM |
| Entities + relations | GLiNER2 (zero-shot) |
| PII + guardrails | GLiNER2-Guardrails-PII |
| Entity resolution | GLiNER-Linker + reranker |
| Embeddings | Granite Embedding r2 |

→ say: *The framework names capabilities, never models. Swap any box.*

---

## ACT 3 — THE KEY DESIGN CHOICE (slides 9–12)

---

### 9 · Extract narrow. Resolve wide.

| | **Per chunk** | **Corpus-wide** |
|---|---|---|
| Stages | parse → redact | resolve → export |
| Question | *what is in this chunk?* | *are these the same thing?* |
| Labels | this chunk's own few | one canonical vocabulary |
| Scales with | **number of workers** | corpus size, once |

---

### 10 · Why most stacks get this backwards

**They do:** fix one global schema → extract everything against it

**Cost:**
- schema must be capped → **rare-but-critical types vanish**
- every chunk scored against ~40 labels → **precision drops**
- label dimension explodes the tensor → **out-of-memory**

`[visual] batch × tokens × LABELS ← the term that hurts`

---

### 11 · Local questions get local answers

- *"What's in this chunk?"* → **local**. Answer with the chunk. Nothing else.
- *"Are these two the same?"* → **global**. Needs the whole corpus.

**Guessing the global answer first is what forces the cap.**
**Answering it last, with all evidence, makes both answers good.**

---

### 12 · Safety is the deliberate exception

- **Relevance labels** → scoped per chunk
- **Safety floors** → apply to **every** chunk, always

Financial PII (IBAN · card · SSN · tax id) + harmful-content categories
= enforced beneath the model, **non-bypassable by model choice**

→ say: *A leaked card number in an "irrelevant" chunk is still a leaked card number.*

---

## ACT 4 — RETRIEVAL (slides 13–17)

---

### 13 · We don't replace your search engine

**Stateless by design** — holds no index, runs no first-stage search

```
your engine  ──search()──▶  [ fuse · expand · rerank · pack ]  ──▶ context
     ▲
Qdrant · Weaviate · Elastic · Azure · Databricks · Neo4j
```

→ say: *Adopt the quality layer without a migration.*

---

### 14 · Four signals, fused

| Signal | Finds |
|---|---|
| **Dense** | meaning |
| **BM25 / sparse** | exact terms, codes, clauses |
| **Knowledge graph** | **what the document points to** |
| *Multi-vector* | *late-interaction (experimental)* |

Fused by **reciprocal rank** — scale-free, no score calibration needed.

---

### 15 · The multi-hop expander

`[visual]  found chunk (Doc A) → its entities → graph → Doc B → surface Doc B's chunk`

1. Take the top chunks the first stage **did** find
2. Read their entity nodes
3. **Traverse the graph** 2 hops
4. Boost candidates that are graph-close (decayed by distance)

**Zero ops** — runs directly over the exported graph file.

---

### 16 · The knapsack packer

**Problem:** top-10 wastes the context window on near-duplicates.

**Solve:** maximize `relevance − redundancy` under a token budget

- picks best marginal gain per token
- penalizes overlap with what's already chosen
- boosts by graph importance (centrality)

→ say: *A full context window of the same paragraph five times is a wasted window.*

---

### 17 · Multi-vector — honest status

- **MaxSim** late-interaction scoring — implemented
- **MUVERA FDE** — training-free multi-vector → single dense vector, fits any ANN index
- **Experimental, off by default. No weights shipped.**

→ say: *An on-ramp, not a checkbox.*

---

## ACT 5 — THE PROOF (slides 18–21)

---

### 18 · We measured it. Live index, no mocks.

- **80 IETF RFCs** through the wizard, on one GPU
- Questions generated from real cross-reference chains
- Kept only if fused beat **both** dense-only and BM25-only
- Retrieval run against a **live vector database**

---

### 19 · The result

| recall@10 | all 90 questions | the 7 multi-hop |
|---|---|---|
| Dense only | 0.644 | **0.000** |
| BM25 only | 0.656 | **0.000** |
| **Fused + graph** | **0.678** | **1.000** |

`[visual] make the 0.000 / 0.000 / 1.000 column huge`

---

### 20 · Read it honestly

- Across all 90: **+3 points**. Modest — most questions *are* single-signal answerable.
- On the 7 that need chaining: **0% → 100%**.

**The claim is not "5× better everywhere."**
**It is: a class of questions single retrievers structurally cannot answer.**

→ say: *That's the class your experts actually ask.*

---

### 21 · What made it work

| | Reference coverage |
|---|---|
| Headers only | 14% |
| **+ inline citations** | **~112%** |

No edges → no multi-hop.
**The retrieval result is only as good as the graph. The graph is only as good as the extraction.**

---

## ACT 6 — ENTERPRISE (slides 22–24)

---

### 22 · Built to be adopted, not migrated to

- **Model-agnostic** — capabilities, not vendors. Swap any model.
- **Cloud-agnostic** — output is files. Run it anywhere.
- **Engine-agnostic** — keep your vector database.
- **Licenses attested** — weights and code verified separately; every default permissive.
- **PII floor non-bypassable** — enforced beneath every model.
- **Incremental** — deltas, not full reprocessing.

---

### 23 · What we don't claim

- 7 clean multi-hop wins — a real signal, **a small sample**
- One standards corpus — broader domains are next
- Multi-vector path — **experimental**
- Aggregate lift — **modest by design**, and we say so

→ say: *Trust the numbers because nothing is dressed up.*

---

### 24 · Close

**Single-hop retrieval matches text.**
**Multi-hop retrieval follows the references.**

*Following the references is what answering a real enterprise question requires.*

`[visual] the two arrows from slide 3 — second one now traversed`

---

## APPENDIX — backup slides

---

### A1 · Full stage list

`source · intake-screen · parse · chunk · content-screen · induce · extract · header-refs ·
inline-refs · redact · profiling · resolve · graph(+pagerank/community) · complete · enrich ·
export-kg · export-corpus`

---

### A2 · What ships (the deliverable)

- `records.parquet` — chunks + embeddings + graph columns
- `graph-nodes/edges.parquet` + `graph.ttl` — the KG with evidence
- `bm25-stats.json` — lexical signal
- `quality-report.md` — substantiates "AI-ready"
- `manifest.json` — provenance

---

### A3 · Example: the hard questions

| Question framed in | Answer lives in |
|---|---|
| RFC 7230 (HTTP) | RFC 5246 (TLS cipher suites) |
| RFC 5861 (cache-control) | RFC 2119 (MUST / SHOULD) |
| RFC 6838 (media types) | RFC 2045 (7bit/8bit data) |
| RFC 8126 (IANA) | RFC 5226 (assignment policies) |

Dense **0** · BM25 **0** · Fused **1.0** — on every one.

---

### A4 · Why per-chunk also fixed memory

`batch × tokens × labels` → the tensor

- Global schema: ~40 labels → **multi-GB allocation → OOM**
- Per chunk: a handful → **small tensor**
- Bucket chunks by label set → **full batch throughput anyway**

→ say: *The quality fix and the memory fix are the same fix.*
