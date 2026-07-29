# Provider catalog

A **Provider** is a concrete implementation of a [Capability](../concepts/index.md) —
registered under `latence.providers` and resolved by its entry-point name. The framework
never names a model; it names Capabilities and lets Providers fulfil them. Every Provider
declares a [`ProviderProfile`](../reference/providers.md) with its compute, footprint, and
**verified weights-and-code license** — the columns below come straight from those declared
objects.

!!! tip "Reading the catalog"
    - **Compute** — `cpu` runs anywhere, `gpu` needs CUDA (skipped-with-flag on a CPU-only
      host via the device seam), `either` follows the requested device.
    - **License** — the SPDX of the model *weights* (or the code, for a pure-code Provider).
      Reference Providers ship **permissive weights only**; restricted models are opt-in
      ([ADR-0012](../adr/0012-permissive-only-reference-providers-gemma-opt-in.md)).
    - **Endpoint** Providers are thin HTTP clients (any OpenAI-compatible endpoint) — the
      served model lives behind the endpoint, so they carry no local weights.

## Built-in reference Providers (ship in `latence-core`)

Every Stage has a deterministic, dependency-light reference Provider that runs on a laptop
CPU, offline ([ADR-0007](../adr/0007-cpu-first-reference-providers.md)). These ship inside
core so the demo runs end-to-end with no extra install, and most Stages ship a *second*
CPU reference Provider so the seam is exercised by real alternatives
([ADR-0035](../adr/0035-second-cpu-reference-providers-for-single-provider-capabilities.md)).

| Capability | Provider | Technique | Compute |
| --- | --- | --- | --- |
| Source | `source.local_folder` | Filesystem / object-store listing + fetch | cpu |
| Chunk | `chunk.markdown`, `chunk.sentence_window` | Offset-preserving markdown / sentence-window packing | cpu |
| Screening (intake) | `screening.intake_signature` | Signature + size intake screen | cpu |
| Screening (content) | `screening.content_keyword`, `screening.content_fuzzy` | Keyword / fuzzy injection detection | cpu |
| Entity Extraction | `entity.gazetteer` | Zero-shot gazetteer / regex | cpu |
| Relation Extraction | `relation.pattern` | Trigger-phrase patterns | cpu |
| Redaction | `redaction.hybrid_rule` | Regex library + gazetteer, sensitivity-keyed | cpu |
| Profiling | `profiling.statistical`, `profiling.lightweight` | Streaming corpus features | cpu |
| Disambiguation | `disambiguation.cascade`, `disambiguation.exact_surface` | 5-strategy resolver / exact-surface blocking | cpu |
| Graph Assembly | `graph.canonical`, `graph.weighted` | Content-addressed nodes/edges + per-edge Evidence | cpu |
| Graph Completion | `graph_completion.reference` | Bounded 2-hop transitive/symmetric closure | cpu |
| Context Enrichment | `context.kg_header` | KG-header projection onto chunks | cpu |
| Embedding | `embedding.hashing` | Deterministic hashing embedder | cpu |
| Delta | `delta.affected_set` | Affected-set recompute + drift reconcile | cpu |
| Export | `export.jsonl_parquet`, `export.knowledge_graph` | JSONL/Parquet + TTL/GraphML files | cpu |

## Source

Connectors to source *systems* live outside core as their own Provider packages
([ADR-0014](../adr/0014-source-seam-thin-connectors.md)) — the Source seam is what makes a real
connector a plugin rather than a fork.

| Provider | Package | Technique | License | Compute |
| --- | --- | --- | --- | --- |
| `source.sharepoint` | `latence-source-sharepoint` | SharePoint / OneDrive via Microsoft Graph; content-hash Delta skips unchanged documents **without downloading them**; secrets env-only | PolyForm-Noncommercial-1.0.0 (code), **no third-party dependency** | cpu |

## Parse

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `parser.plaintext` | `latence-parser-plaintext` | Passthrough plain-text / markdown | PolyForm-Noncommercial-1.0.0 (code) | cpu |
| `parser.document` | `latence-parser-document` | PDF + text → markdown + page map | PolyForm-Noncommercial-1.0.0 (code) | cpu |
| `parser.pdfplumber` | `latence-parser-pdfplumber` | Page-aware PDF text layer (pdfplumber) | MIT (code) | cpu |
| `parser.render` | `latence-parser-render` | Universal any-file render → OCR, **license-clean (no PyMuPDF/AGPL)** | PolyForm-Noncommercial-1.0.0 (code) | either |
| `parser.lighton` | `latence-parser-lighton` | `lightonai/LightOnOCR-1B-1025` (1B VLM OCR) | Apache-2.0 | gpu |
| `parser.glm` | `latence-parser-glm` | `zai-org/GLM-OCR` (0.9B VLM OCR) | MIT | gpu |
| `parser.lighton_vllm` | `latence-parser-lighton-vllm` | `lightonai/LightOnOCR-2-1B` served via vLLM | Apache-2.0 | either (endpoint) |
| `parser.endpoint` | `latence-parser-endpoint` | PaddleOCR-VL / any vision endpoint | Apache-2.0 (client) | either (endpoint) |

## Entity & Relation Extraction

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `entity.gliner` | `latence-ner-gliner` | `urchade/gliner_multi-v2.1` (zero-shot NER) | Apache-2.0 | either |
| `entity.endpoint` | `latence-ner-endpoint` | GLiNER-large / any endpoint | Apache-2.0 (client) | either (endpoint) |
| `relation.gliner_relex` | `latence-relation-gliner` | `knowledgator/gliner-relex-multi-v1.0` (fused NER+RE) | Apache-2.0 | either |
| `relation.llm` | `latence-relation-llm` | Any OpenAI-compatible LLM (incl. local Ollama) | Apache-2.0 (client) | either (endpoint) |
| `fused_entity_relation.gliner2` | `latence-extract-gliner2` | `fastino/gliner2-multi-v1` (mdeberta-v3, fused) | Apache-2.0 | either |

## Redaction (PII)

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `redaction.gliner_pii` | `latence-pii-gliner` | `urchade/gliner_multi_pii-v1` (zero-shot PII) | Apache-2.0 | either |
| `redaction.gliner2` | `latence-pii-gliner2` | `fastino/GLiNER2-Guardrails-PII-Multi` | Apache-2.0 | either |
| `redaction.presidio` | `latence-pii-presidio` | Microsoft Presidio + spaCy `en_core_web_sm` | MIT | either |

## Disambiguation

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `disambiguation.embedding` | `latence-disambig-embedding` | Learned surface+context resolution (registry Embedder) | PolyForm-Noncommercial-1.0.0 (code) | either |
| `disambiguation.glinker` | `latence-disambig-glinker` | `knowledgator/gliner-linker-large-v1.0` neural linker + audited resolver | Apache-2.0 | gpu |

## Graph Completion & Schema Induction (optional Stages)

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `graph_completion.ultra` | `latence-linkpred-ultra` | Inductive ULTRA (`ultra_3g`) zero-shot scorer + PyKEEN fallback | MIT | either |
| `label_inducer.llm` | `latence-schema-inducer` | Any small OpenAI-compatible LLM induces each chunk's label schema (hosted, OpenRouter, or a LOCAL vLLM/Ollama box — `base_url` only) | Apache-2.0 (client) | either (endpoint) |
| `label_inducer.lexicon` | `latence-schema-inducer` | Deterministic zero-network tagger over an operator-declared candidate type space — the non-frontier per-chunk option (no model, no calls) | PolyForm-Noncommercial-1.0.0 | cpu |

## Embedders

| Provider | Package | Model / technique | License | Compute |
| --- | --- | --- | --- | --- |
| `embedding.sentence_transformers` | `latence-embedder-st` | `ibm-granite/granite-embedding-311m-multilingual-r2` (GPU 311m / CPU 97m) | Apache-2.0 | either |
| `embedding.endpoint` | `latence-embedder-endpoint` | `multilingual-e5-large` / any embeddings endpoint | Apache-2.0 (client) | either (endpoint) |

See [Embedders](../embedders.md) for how the opt-in RAG-corpus vectors fit into Export, and
[ADR-0045](../adr/0045-granite-r2-default-embedder-gpu-311m-fa2-bf16-cpu-97m.md) for why
Granite r2 is the default.

## Runners

Not Providers, but the other swap point: a **Runner** schedules the identical Stages on a
substrate. Alongside the built-in local Runner, `latence-runner-airflow` runs the same
Stages under Apache Airflow using Airflow's own state and scheduling
([ADR-0003](../adr/0003-stage-library-with-pluggable-runners.md),
[ADR-0030](../adr/0030-performance-path-endpoint-providers-and-second-runner-adapter.md)).

## Build your own

The catalog is meant to grow. To add a Provider — implement a Capability, register the
entry point, declare a `ProviderProfile`, and pass conformance — follow
**[Authoring a Provider](README.md)**, then compare it against the incumbents with a
[bake-off](../running-a-bakeoff.md).
