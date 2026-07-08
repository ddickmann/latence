# latence-embedder-st

The first real **learned** `Embedder` Provider (W1-embed, ADR-0004/0007/0016/0017, ADR-0036 §1):
an in-process semantic embedder over the **`sentence-transformers`** library, defaulting to
**`intfloat/multilingual-e5-small`** (MIT, 384-dim). It is the same opt-in `Embedder` Capability as
the in-core deterministic `embedding.hashing` reference and the `embedding.endpoint` performance
path — swap the toy hashing embedder for a real semantic model with **one config line**, so the
exported RAG corpus carries embeddings a vector DB ingests directly.

Built as an `AdapterBase` subclass (P3-F2): it exercises the shared device seam (`cpu|cuda`),
input-order-preserving streaming batches, and native-exception → typed `ProviderError` mapping.

## Usage

```yaml
# In an Export stage's config (ADR-0017 opt-in Embedder augmentation):
config:
  embedder:
    provider: embedding.sentence_transformers   # was: embedding.hashing
    config:
      model: intfloat/multilingual-e5-small     # MIT, 384-dim (the default)
      dimension: 384                            # validated against the loaded model
      prefix: "passage: "                       # e5 document-side instruction prefix
      batch_size: 32
      device: auto                              # auto|cpu|cuda through the device seam
```

`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 384-dim, `prefix: ""`) is a documented
alternative.

## Model card convention (e5)

e5 models are trained with an instruction **prefix** — `query: ` for a search query, `passage: `
for a document — and score cosine similarity over **L2-normalised** vectors. This Provider embeds
RAG-corpus text (the document side), so it prepends `passage: ` by default and passes
`normalize_embeddings=True` so every exported vector is unit-length. See
<https://huggingface.co/intfloat/multilingual-e5-small>.

## Offline / CI discipline

The real `sentence-transformers` stack (torch/transformers) is **not** installed in the CPU-first,
offline test env (ADR-0007/0016). The package is installed `--no-deps` and its tests monkeypatch
`SentenceTransformer` with a deterministic fake — the same honest boundary the gliner trio uses. A
real-weights test is present and **gated** on `importlib.util.find_spec("sentence_transformers")`
(skipped offline, run where the model exists). The `matrix/embed.yaml` bake-off compares
`embedding.hashing` (real) vs this candidate (skip-with-flag offline — never a fabricated number).

## Licensing

Framework code and the `sentence-transformers` library are **Apache-2.0** (code, verified). The
default checkpoint `intfloat/multilingual-e5-small` is **MIT** (weights, verified from the HF
model-card, 2026-07-08). Both permissive; nothing downloads unless a Pipeline configures this
Provider (ADR-0012).
