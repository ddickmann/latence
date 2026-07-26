# latence-pii-gliner2

A learned **PII Redaction** Provider for the
[Latence framework](../../README.md) over the fastino
[**GLiNER2**](https://github.com/fastino-ai/GLiNER2) framework: it detects PII in
parsed-document text using the `fastino/GLiNER2-Guardrails-PII-Multi` checkpoint
(an mdeberta-v3 encoder) and produces the masked/replaced text variant plus a
structured, auditable list of PII spans, running in-process on CPU or GPU via the
GLiNER2 framework's `transformers`/torch stack.

It registers under the `latence.providers` entry-point group as `redaction.gliner2`,
so installing this package makes the Provider discoverable by any Pipeline with **no
change to `latence-core`** (ADR-0004). It is a **twin of `redaction.gliner_pii`** — a
true drop-in behind the same `PIIDetector` seam as the deterministic,
dependency-free `redaction.hybrid_rule` in core and the `gliner`-library
`redaction.gliner_pii`. All three share the framework's policy + masking machinery
(`latence_core.redaction_policy`), so the only difference is detection.

## What differs from `redaction.gliner_pii`

The twin wraps the `gliner` library; this wraps the **`gliner2`** framework — a
*different* Python package with a *different* API. `gliner2`'s `extract_entities`
returns a **per-label dict of dict-rows** (not the flat list `gliner` returns):

```python
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi")
out = model.extract_entities(text, entity_types, threshold=0.5,
                             include_spans=True, include_confidence=True)
# -> {"entities": {"<label>": [{"text": str, "confidence": float,
#                               "start": int, "end": int}, ...], ...}}
```

Offsets are document-char `start`/`end` (half-open), exact — they map directly to
`PIISpan(char_start=start, char_end=end, pii_type=label, confidence=confidence)`.

## Metadata-aware policy

Redaction keys its action off the document's `Classification.sensitivity` via the
same `policies` config the in-core Provider uses: a `public` document can be masked
hard while an `internal` one is only tagged for review, from one Pipeline.

## Licensing

Both the framework code and the default weights are permissive (verified
2026-07-09):

- `gliner2` library — **Apache-2.0** (code): github.com/fastino-ai/GLiNER2 LICENSE
  + README badge + PyPI `OSI Approved :: Apache Software License` classifier.
- Default checkpoint `fastino/GLiNER2-Guardrails-PII-Multi` — **Apache-2.0**
  (weights, per the HF model card; **not gated** / freely downloadable).

## Install

```bash
pip install latence-core 'latence-pii-gliner2'
```

This pulls `gliner2[local]>=1.3,<2` (and, via that extra, its `torch`/`transformers`
stack — the base `gliner2` dist ships neither). The model weights are downloaded lazily
on first run and cached by `transformers`.

### Supported `gliner2` versions (#192)

This Provider drives `GLiNER2.from_pretrained(model_id, quantize=…, compile=…)` and
`model.batch_extract_entities(…, include_spans=True, include_confidence=True, max_len=…)`.
That surface does not exist across the whole 1.x line — verified against the released
distributions (2026-07-18): 1.0.x has **no** `batch_extract_entities` at all, 1.1.x has no
`include_spans`/`max_len`, and 1.2.0–1.2.5 have no `max_len`. The package therefore pins
`gliner2[local]>=1.3,<2` (the pod-validated line), so drift fails at dependency
**resolution**. An environment the resolver never saw (a hand-`pip install`ed pod) is
caught at model load by a typed `ProviderError` naming the supported range and the install
command — never a raw native `AttributeError`.

## Use in a Pipeline

```yaml
  - name: redact
    capability: redaction
    provider: redaction.gliner2
    depends_on: [chunk]
    config:
      pii_labels: [person, email, phone number, credit card number]
      model: fastino/GLiNER2-Guardrails-PII-Multi  # Apache-2.0 default
      threshold: 0.5                       # drop spans scoring below this
      default_action: mask                 # mask | replace | hash | tag
      device: auto                         # auto | cpu | cuda (device seam)
      policies:
        public: { action: mask }
        internal: { action: tag }          # detect-only for internal docs
      # guardrails: true                   # also screen for learned guardrail categories (T2)
      # guard_labels: [malware instructions]   # extra categories over the universal guard floor
```

## Safety floors are UNIVERSAL; only induced types are per-chunk (T2)

The sibling extractor scopes its config labels **per chunk** (relevance labels). This Redactor does
the opposite with its config labels, deliberately:

* **Induced `pii_types` are per chunk** — a chunk is screened for what its own induction found.
* **The configured `pii_labels` are NOT** — they are a *safety* floor applied to **every** chunk. A
  leaked card in a chunk whose induced types are all about shipping logistics must still be masked.
  Attempting to scope them (`label_floor: per_chunk`) is a `ConfigError`, not a silent no-op.
* The deterministic **ADR-0044 financial floor** (`ssn` / `credit_card` / `iban`) is the second,
  model-independent layer of the same guarantee, enforced for every Redaction Provider at the shared
  `finalize_redacted_chunk` seam — so a model that misses one still cannot leak it.

### Learned guardrails (`guardrails: true`, off by default)

The default checkpoint is a **guardrails**+PII model, so with `guardrails: true` each chunk is also
screened for guardrail categories under the same rule: the universal guard floor
(`prompt injection`, `harmful content`) plus any `guard_labels`, plus the categories induced for
**that** chunk (`induced_labels.guard_types`). The floor is never subsetted per chunk.

A guardrail hit is **flagged, never masked**: it produces a `RiskMarker` on the chunk (categories
`prompt_injection` / `harmful` — the same vocabulary `screening.content_keyword` emits, so a RAG
consumer filters one set) whose `reason` names the category and hit count and never echoes the
matched text. Masking would silently corrupt the corpus text; the marker survives into it.

This is orthogonal to the deterministic keyword `content_screen` Stage, which is a separate,
earlier pre-filter and is untouched. Off by default ⇒ no extra labels, no markers, byte-identical.
