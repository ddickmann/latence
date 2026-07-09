# latence-schema-inducer

An unsupervised **schema-induction** Provider for the
[Latence framework](../../README.md): a small OpenAI-compatible LLM with structured
JSON output reads each document and induces the **entity / relation / PII types**
actually evidenced in it, then annotates that document's chunks so the extraction
Stages discover their own label schema — the "point it at a folder and it discovers
its own schema" property (ADR-0038, W4-schema-induction).

It registers under the `latence.providers` entry-point group as `label_inducer.llm`,
so installing this package makes the Provider discoverable by any Pipeline with **no
change to `latence-core`** (ADR-0004). It fulfils the OPTIONAL `schema_induction`
Stage, inserted after Content Screening and before the extraction Stages. **Off by
default**: a stack with no `schema_induction` Stage is byte-identical to today.

## Any OpenAI-compatible endpoint (OpenAI ⇄ OpenRouter ⇄ on-prem vLLM)

The Provider talks the OpenAI chat-completions API through the official `openai` SDK,
so the same code drives OpenAI, OpenRouter, or an **on-prem / air-gapped vLLM** with
only a `base_url` change:

- `base_url=None` ⇒ OpenAI (default `model` = **`gpt-4.1-mini`**);
- `base_url="https://openrouter.ai/api/v1"` ⇒ OpenRouter;
- `base_url="http://<vllm-host>:8000/v1"` ⇒ self-hosted vLLM.

`api_key` comes from config or `$OPENAI_API_KEY`; nothing is hard-coded. The `openai`
client is Apache-2.0 (code); this package bundles **no model weights** (the model is a
service). Fault tolerance is the ported `latence_core.endpoint` retry, identical to the
other endpoint Providers; the transient-retry count surfaces on `endpoint_retries`.

## The carrier (additive — no extractor rewrite)

The Stage annotates each `ChunkRecord` / `DocumentRecord` with an additive
`InducedLabels` (`entity_types` / `relation_types` / `pii_types` + `model_id`). The four
extraction Providers (`entity.gliner`, `relation.gliner_relex`, `relation.llm`,
`redaction.gliner_pii`) then compute their **effective** label set as
`config_labels ∪ induced` (deduped, order-stable) — config labels are the mandatory
floor, induced labels extend them. A chunk with no induced labels yields exactly its
config labels back (byte-identical backward compatibility).

## Safety nets — the LLM is untrusted output

Every response is treated as untrusted; the run **never crashes** on bad output:

1. **Structured output** — `response_format={"type":"json_object"}` (portable).
2. **Generous token budget + truncation detection** — `max_tokens` default 2048; a
   `finish_reason == "length"` reply retries at double budget, then falls back.
3. **Parse-and-repair** — `json.loads` in a guard; one corrective retry; then fall back.
   Never `eval`, never execute model text.
4. **Schema validation** — the object must be `{entity_types, relation_types, pii_types}`,
   each a list of strings; else fall back.
5. **Label sanitization** — strip control chars, cap length ≤ 64, dedupe, cap count ≤ 40.
6. **Fail-open, never crash** — any failure leaves `induced_labels` unset (extractors use
   config labels) + a typed no-content log + a Quality-Report note (`induction_fallbacks`).

## Configuration

```yaml
- name: induce_labels
  capability: schema_induction
  provider: label_inducer.llm
  depends_on: [content_screen]
  config:
    model: gpt-4.1-mini          # base_url=None ⇒ OpenAI; set base_url for OpenRouter/vLLM
    granularity: doc             # doc (one call/document, default) | chunk
    max_tokens: 2048
    max_labels: 40
    temperature: 0.0
    # system_prompt / user_prompt_template: override the defaults ({text}/{max_labels})
```

Turn the whole feature on with `auto_generate_labels: true` on the stack (see
`stacks/schema-induction.yaml`). Real `gpt-4.1-mini` validation is the maintainer's keyed
pod run — see `scripts/validate-schema-induction-on-gpu.md`; the offline mock proves the
machinery and every safety net.

## Licensing

Apache-2.0 (framework code). No model weights bundled — the model is a user-supplied
service. The `openai` client is Apache-2.0. Nothing is called unless a Pipeline
configures this Provider (ADR-0016 thin-core / heavy-dep isolation).
