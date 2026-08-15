# latence-parser-endpoint

The **Parse performance path** for the Latence framework (S12, ADR-0007): a `Parser` Provider that
transcribes documents to markdown via any **OpenAI-compatible / vLLM / Triton** endpoint serving a
document-understanding model (e.g. PaddleOCR-VL). It is the same `parse` Capability as the in-core
CPU reference `parser.document` — a Pipeline swaps CPU↔endpoint with **one config change**, no
pipeline edit.

## Usage

```yaml
- name: parse
  capability: parse
  provider: parser.endpoint     # was: parser.document
  depends_on: [source]
  config:
    base_url: "http://vllm-host:8000/v1"   # the only change vs the CPU path
    model: "paddleocr-vl"
    api_key: "not-needed"        # or $OPENAI_API_KEY; a local server takes any placeholder
    retry_attempts: 4            # ported exponential-backoff-with-jitter retry
```

Point `base_url` at OpenAI, a self-hosted vLLM, Triton, or a laptop-local server — the Stage,
contracts, and downstream pipeline are unchanged.

## Fault tolerance

Every call goes through the ported endpoint retry (`latence_core.endpoint`): a 429/5xx/transport
blip is retried with exponential backoff + jitter; a terminal 4xx fails fast; the number of retries
a run absorbed lands on the Quality Report's per-Stage `endpoint_retries`. An unrecoverable failure
yields a `PARSE_ERROR` record (Provenance + reason retained) rather than aborting the run — the same
graceful-degradation contract as the CPU reference, so output is identical modulo the model.

## Licensing

Framework code, Apache-2.0 (the repository-wide licence). Bundles **no** model weights — it calls a user-supplied endpoint. The
`openai` client is Apache-2.0 (code). Nothing is downloaded or called unless a Pipeline configures
this Provider.
