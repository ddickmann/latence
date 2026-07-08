# Embedders — the opt-in RAG-corpus vector augmentation

The **Embedder** is an *opt-in* Export augmentation (ADR-0017): a corpus Export names an Embedder
Provider in its `config['embedder']`, and every exported row gets a fixed-width `embedding` vector a
vector DB can ingest. Embeddings are **never required** — a corpus Export with no `embedder` still
exports its cleaned/chunked/PII-handled text with full Provenance/Classification. The framework names
only the `Embedder` **Capability** (a `dimension` property + `embed(texts) -> Iterator[list[float]]`);
Providers fulfil it.

## The three reference Providers (one Capability, one config line to swap)

| Provider | Package | What it is | License (code / weights) | Runs offline in CI |
|---|---|---|---|---|
| `embedding.hashing` | in-core | Deterministic, dependency-free token-hash bag-of-words (the CPU reference — **not** semantic) | Apache-2.0 / — | yes (real) |
| `embedding.sentence_transformers` | `latence-embedder-st` | Learned in-process semantic embedder over `sentence-transformers`, default `intfloat/multilingual-e5-small` (384-dim) | Apache-2.0 / MIT | no (heavy dep; verified against a stub, real weights on a rig) |
| `embedding.endpoint` | `latence-embedder-endpoint` | Any OpenAI-compatible / vLLM / Triton `/embeddings` endpoint (the served performance path) | Apache-2.0 / — (calls a user endpoint) | no (endpoint; verified against a stub) |

Swapping is a **one-line** config change on the corpus Export (ADR-0007):

```yaml
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]           # or [redaction] for a PII-handled corpus
    config:
      basename: corpus
      embedder:
        provider: embedding.sentence_transformers   # was: embedding.hashing
        config:
          model: intfloat/multilingual-e5-small     # MIT, 384-dim
          dimension: 384                            # validated against the loaded model
          prefix: "passage: "                       # e5 document-side instruction prefix
          device: auto                              # auto|cpu|cuda through the device seam
```

## e5 model-card convention (`latence-embedder-st`)

The default checkpoint `intfloat/multilingual-e5-small` (and the e5 family) is trained with an
instruction **prefix** — `query: ` for a search query, `passage: ` for a document — and scores cosine
similarity over **L2-normalised** vectors. The Provider embeds RAG-corpus text (the document side), so
it prepends `passage: ` by default and passes `normalize_embeddings=True` so every exported vector is
unit-length. Set `prefix: ""` for a model that needs no prefix (e.g. `sentence-transformers/all-MiniLM-L6-v2`,
Apache-2.0). See <https://huggingface.co/intfloat/multilingual-e5-small>.

## Bake-off

`matrix/embed.yaml` (`stage: embedding`, `stage_name: export_corpus`) compares the embedders by
swapping only the corpus Export's embedder, holding every other Stage constant. On a CPU-only host the
`embedding.hashing` row runs for real (measured, deterministic) and the learned/served candidates are
**skipped-with-flag** through the same `select_device` seam — never a fabricated number (#64). Run it
with `latence bake-off matrix/embed.yaml`.

## Offline / CI discipline (learned + endpoint Providers)

The heavy/endpoint Provider packages are installed `--no-deps` in the CPU-first, offline test env
(ADR-0007/0016); their tests monkeypatch the heavy dep (`SentenceTransformer` / the `openai` client)
with a deterministic fake, and each ships a **real** path gated on the dep being importable (skipped
offline, run where the model/endpoint exists). The Provider Conformance suite verifies them against a
stub and flags the row `conformance-verified against stub; live numbers need a rig`.
