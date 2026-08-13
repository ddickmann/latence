# latence-ner-endpoint

The **NER performance path** (S12, ADR-0007): an `EntityExtractor` Provider that runs zero-shot NER
via any **OpenAI-compatible / vLLM / Triton** endpoint serving a large model (e.g. GLiNER-large). It
is the same `entity_extraction` Capability as the in-core `entity.gazetteer` and the in-process
learned `entity.gliner` — a Pipeline swaps CPU↔endpoint with **one config change**.

## Usage

```yaml
- name: entities
  capability: entity_extraction
  provider: entity.endpoint     # was: entity.gliner / entity.gazetteer
  depends_on: [chunk]
  config:
    labels: ["person", "organization", "location"]   # zero-shot, per-Pipeline
    base_url: "http://vllm-host:8000/v1"
    model: "gliner-large"
    threshold: 0.5
```

Mentions carry the same offset+page Provenance (the S4 round-trip) as the CPU Provider, so the
output is identical modulo the model.

## Fault tolerance & co-location

Calls go through the ported endpoint machinery (`latence_core.endpoint`): a reference-counted
`EndpointLease` means several Stages against the same served model pay the cold-start once, and
every call retries transient (429/5xx/transport) failures with backoff + jitter while failing fast
on terminal 4xx. Retries absorbed land on the Quality Report's per-Stage `endpoint_retries`.

## Licensing

Framework code, PolyForm Noncommercial 1.0.0 (the repository-wide licence). Bundles **no** weights — calls a user-supplied endpoint. The `openai`
client is Apache-2.0 (code).
