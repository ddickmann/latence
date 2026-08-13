# latence-embedder-st

The framework's default **learned** `Embedder` Provider (W1-embed/W19, ADR-0004/0007/0016/0017/0036
§1, ADR-0045): an in-process semantic embedder over the **`sentence-transformers`** library,
defaulting to **IBM Granite Embedding r2** (Apache-2.0, ModernBERT-based). It is the same opt-in
`Embedder` Capability as the in-core deterministic `embedding.hashing` reference and the
`embedding.endpoint` performance path — swap the toy hashing embedder for a real semantic model
with **one config line**, so the exported RAG corpus carries embeddings a vector DB ingests
directly.

Built as an `AdapterBase` subclass (P3-F2): it exercises the shared device seam (`cpu|cuda`), the
W8 perf seam (bf16 + flash-attention-2 on GPU, graceful `sdpa` fallback; fp32 + `sdpa` on CPU),
input-order-preserving streaming batches, and native-exception → typed `ProviderError` mapping.

## Two tiers (ADR-0045)

| Tier | Model | Dim | License (weights / code) | Load |
|---|---|---|---|---|
| **GPU (default)** | `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | Apache-2.0 / Apache-2.0 | bf16 + `flash_attention_2` (→ `sdpa` fallback) |
| **CPU** | `ibm-granite/granite-embedding-97m-multilingual-r2` | 384 | Apache-2.0 / Apache-2.0 | fp32 + `sdpa` (no FA2) |

Both are ModernBERT-based, single-vector dense, **32,768-token** context, **200+ languages**
(enhanced 52 + code), and need **NO instruction/query-document prefix**.
`intfloat/multilingual-e5-small` (MIT, 384-dim, needs `prefix: "passage: "`) and
`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) remain documented tiny fallbacks — not the
default.

## Usage

```yaml
# GPU tier (in a corpus Export or a disambiguation.embedding config; ADR-0017 opt-in):
config:
  embedder:
    provider: embedding.sentence_transformers
    config:
      model: ibm-granite/granite-embedding-311m-multilingual-r2   # Apache-2.0, 768-dim (default)
      dimension: 768                          # validated against the loaded model
      device: cuda                            # bf16 auto on cuda (perf seam)
      attn_implementation: flash_attention_2  # → sdpa if flash_attn absent or the load raises
```

```yaml
# CPU tier:
config:
  embedder:
    provider: embedding.sentence_transformers
    config:
      model: ibm-granite/granite-embedding-97m-multilingual-r2    # Apache-2.0, 384-dim
      dimension: 384
      device: cpu                             # fp32 + sdpa (no FA2)
```

## Model-card convention (Granite r2)

Granite r2 encodes text **directly** (`model.encode(texts)`) — no `query:`/`passage:` prefix, so
the default `prefix` is `""`. The Provider still passes `normalize_embeddings=True` so every
exported vector is unit-length (cosine convention, ADR-0017 uniform column). The `dimension` is read
from the loaded model. Cards:
<https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2> and
<https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>.

## Determinism honesty (anti-false-green)

bf16 + FA2 on GPU is **not** bitwise-deterministic, so the `ProviderProfile` declares
`deterministic=False`. The GPU stack is validated **structurally** (contract/shape/no-PII leak) by
`latence stack validate`, never by byte-identical determinism; the deterministic `embedding.hashing`
reference stays the default on the unit-test / golden path so the suite stays green.

## Offline / CI discipline

The real `sentence-transformers` stack (torch/transformers) is **not** installed in the CPU-first,
offline test env (ADR-0007/0016). The package is installed `--no-deps` and its tests monkeypatch
`SentenceTransformer` with a deterministic fake — the same honest boundary the gliner trio uses. A
real-weights test is present and **gated** on `importlib.util.find_spec("sentence_transformers")`
(skipped offline, run where the model exists). The `matrix/embed.yaml` bake-off compares
`embedding.hashing` (real) vs this candidate (`device: cuda`, skipped-with-flag offline — never a
fabricated number).

## Licensing

Framework code is **PolyForm Noncommercial 1.0.0** (the repository-wide licence); the `sentence-transformers` / `transformers` library is **Apache-2.0** (code,
verified). Both Granite r2 checkpoints are **Apache-2.0** (weights; card tag `license: apache-2.0`,
verified 2026-07-14), as is the ModernBERT backbone. Both permissive; nothing downloads unless a
Pipeline configures this Provider (ADR-0012/0045).
