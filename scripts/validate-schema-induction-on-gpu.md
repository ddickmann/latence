# Validate `label_inducer.llm` with a real gpt-4.1-mini — turnkey keyed run (W4-schema-induction)

The offline slice proves the whole machinery + **every safety net** against a **faithful `openai`
mock** (`choices[0].message.content` = JSON string + `finish_reason`): valid JSON; malformed →
repair; malformed twice → fall back; truncated (`finish_reason=length`) → doubled-budget retry;
schema-invalid → fall back; injection/oversized/control-char labels → sanitized; empty → fall back;
network/auth error → fail-open. But only a **real keyed run** proves the actual `gpt-4.1-mini` (or an
OpenRouter/vLLM model) induces a **sensible** `{entity_types, relation_types, pii_types}` schema over
a real document and that the extractors then run with those induced labels.

> **No fabricated numbers.** The induced labels + the downstream extraction counts come from THIS
> run only. If the model returns an unusable schema, that is a prompt/model finding to record — never
> hand-write a schema and call it validated.

This is a **keyed** run, not a GPU run: the LLM is a service. Any host with network to an
OpenAI-compatible endpoint works (no CUDA required). For an **on-prem / air-gapped** validation,
point `base_url` at a self-hosted vLLM instead — the same steps, no key.

## 0. Prereqs

```bash
python --version                       # 3.11+
export OPENAI_API_KEY=sk-...           # OpenAI; OR an OpenRouter key with base_url below
# on-prem/air-gapped instead: no key needed, set base_url to your vLLM, e.g.
#   export LATENCE_INDUCER_BASE_URL=http://<vllm-host>:8000/v1
```

## 1. Install the framework + this provider WITH its heavy dep

```bash
# from a clone of the repo:
cd latence-framework
pip install -e packages/latence-core
pip install -e packages/latence-schema-inducer   # pulls the openai client
python -c "import openai; print('openai', openai.__version__)"
```

## 2. Real induction — a sensible schema over a real document

Runs the reference Provider (`LLMLabelInducer`) against the real `gpt-4.1-mini` over a real document
and checks: (a) the reply is valid JSON of the exact shape, (b) the induced `entity_types` /
`relation_types` / `pii_types` are **sensible** for the text (a human eyeballs them), (c) the labels
are sanitized (bounded count/length, no control chars), (d) determinism (temperature 0 — two runs
agree).

```bash
python - <<'PY'
import os
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_schema_inducer.provider import LLMLabelInducer

text = (
    "Dr. Alice Chen, a cardiologist at Mercy General Hospital in Boston, prescribed "
    "atorvastatin to patient John Doe (SSN 123-45-6789, john.doe@example.com) on 2024-03-15. "
    "Mercy General partners with BlueCross for billing."
)
prov = Provenance(source_uri="mem://pod/doc.txt", file_name="doc.txt", file_type="txt",
                  file_size=len(text), document_id="sha256:pod-doc", char_start=0, char_end=len(text))
cls = Classification(language="en", category="clinical_note", sensitivity="confidential")
chunk = ChunkRecord(record_id="doc#chunk-0", document_record_id="doc", index=0, content=text,
                    token_count=len(text)//4, provenance=prov, classification=cls)

config = {"model": "gpt-4.1-mini"}
base_url = os.environ.get("LATENCE_INDUCER_BASE_URL")
if base_url:
    config["base_url"] = base_url   # OpenRouter: https://openrouter.ai/api/v1 ; vLLM: http://host:8000/v1

provider = LLMLabelInducer(config)
out = list(provider.induce([chunk]))
labels = out[0].induced_labels
assert labels is not None, "induction FAILED-OPEN — no schema induced (check key/endpoint/model)"
print("model_id:      ", labels.model_id)
print("entity_types:  ", labels.entity_types)
print("relation_types:", labels.relation_types)
print("pii_types:     ", labels.pii_types)
print("fail-open count:", provider.induction_fallbacks, "endpoint retries:", provider.endpoint_retries)

# Sensibility (eyeball): expect entities like person/organization/location/medication; relations
# like works_for/prescribed/partners_with; PII like ssn/email. Sanitization bounds:
assert all(len(x) <= 64 for x in labels.entity_types + labels.relation_types + labels.pii_types)
assert len(labels.entity_types) <= 40 and len(labels.pii_types) <= 40

# Determinism: a second run at temperature 0 should agree.
out2 = list(LLMLabelInducer(config).induce([chunk]))
assert out2[0].induced_labels is not None
print("second run entity_types:", out2[0].induced_labels.entity_types)
PY
```

## 3. End-to-end — the induced labels flow into the extractors

Run the opt-in stack against a folder of your documents with the heavy extractor packages installed
so the induced `{entity,relation,pii}` types actually augment the extractors:

```bash
pip install -e packages/latence-relation-gliner packages/latence-pii-gliner   # learned extractors
# edit stacks/schema-induction.yaml: replace {STORAGE_URI}/{CORPUS_DIR}, drop `device: cuda` on the
# `induce` node (the LLM is CPU/keyed — cuda would skip it), keep it on the GLiNER stages if GPU.
latence run stacks/schema-induction.yaml --run-id si-pod
```

Confirm in the Quality Report (`_latence/runs/si-pod/report.json`): the `induce` Stage ran
(`induction_documents > 0`, `induction_fallbacks` low), and the extractor Stages produced mentions /
relations / PII spans for types NOT in their config floor — the induced labels doing work.

## What this run proves that the mock cannot

- The real `gpt-4.1-mini` returns JSON of the pinned shape under `response_format=json_object`.
- The induced schema is **sensible** for real text (a human judgement the mock cannot make).
- The end-to-end union (`config ∪ induced`) actually changes what the extractors find.

Record the induced schema + the extractor deltas in `FINDINGS.md`. Do NOT fabricate any of it.
