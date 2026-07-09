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

This pulls `gliner2[local]` (and, via that extra, its `torch`/`transformers` stack —
the base `gliner2` dist ships neither). The model weights are downloaded lazily on
first run and cached by `transformers`.

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
```
