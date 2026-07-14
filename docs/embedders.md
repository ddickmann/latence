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
| `embedding.sentence_transformers` | `latence-embedder-st` | Learned in-process semantic embedder over `sentence-transformers`, default **IBM Granite Embedding r2** — GPU tier `granite-embedding-311m-multilingual-r2` (768-dim, bf16+FA2), CPU tier `granite-embedding-97m-multilingual-r2` (384-dim, fp32) | Apache-2.0 / Apache-2.0 | no (heavy dep; verified against a stub, real weights on a rig) |
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
          # GPU tier (default): Granite r2 311m, Apache-2.0, 768-dim, bf16 + flash_attention_2.
          model: ibm-granite/granite-embedding-311m-multilingual-r2
          dimension: 768                            # validated against the loaded model
          device: cuda                              # bf16 auto on cuda (perf seam)
          attn_implementation: flash_attention_2    # -> sdpa if flash_attn absent / load raises
          # CPU tier: model: ibm-granite/granite-embedding-97m-multilingual-r2, dimension: 384, device: cpu
```

## Granite r2 model-card convention (`latence-embedder-st`)

The default is **IBM Granite Embedding r2** (Apache-2.0, ModernBERT-based, single-vector dense,
32,768-token context, 200+ languages). It needs **NO instruction/query-document prefix** — text is
encoded directly, unlike the e5 family's `query: `/`passage: `. So the default `prefix` is `""`. The
Provider passes `normalize_embeddings=True` so every exported vector is unit-length (cosine
convention, ADR-0017 uniform column). The GPU tier (`granite-embedding-311m-multilingual-r2`, 768-dim)
loads bf16 + `flash_attention_2` with a graceful `sdpa` fallback (W8 perf seam); the CPU tier
(`granite-embedding-97m-multilingual-r2`, 384-dim) loads fp32 + `sdpa`. `intfloat/multilingual-e5-small`
(MIT, needs `prefix: "passage: "`) and `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) remain
documented tiny fallbacks — not the default. Cards:
<https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2> and
<https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>. See ADR-0045.

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
