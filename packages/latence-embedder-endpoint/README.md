# latence-embedder-endpoint

The **Embedder performance path** (S12, ADR-0007, ADR-0017): an `Embedder` Provider that produces
RAG-corpus embeddings via any **OpenAI-compatible / vLLM / Triton** `/embeddings` endpoint serving a
real model (e.g. multilingual-e5-large, BGE-M3). It is the same opt-in `Embedder` Capability as the
in-core deterministic `embedding.hashing` reference — swap the toy hashing embedder for a real
served model with **one config change**.

## Usage

```yaml
config:
  provider: embedding.endpoint    # was: embedding.hashing
  dimension: 1024                 # must match the served model's output width
  base_url: "http://vllm-host:8000/v1"
  model: "multilingual-e5-large"
  batch_size: 32
```

## Fault tolerance

Every batch goes through the ported endpoint machinery (`latence_core.endpoint`): a reference-counted
lease (co-located Stages pay one cold-start) and per-call retry with backoff + jitter on
429/5xx/transport, failing fast on terminal 4xx. Retries absorbed land on the Quality Report's
per-Stage `endpoint_retries`. Every returned vector is validated to the configured `dimension` so a
downstream vector DB gets a uniform column.

## Licensing

Framework code, PolyForm Noncommercial 1.0.0 (the repository-wide licence). Bundles **no** weights — calls a user-supplied endpoint. The `openai`
client is Apache-2.0 (code).
