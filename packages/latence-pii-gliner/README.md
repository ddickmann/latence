# latence-pii-gliner

A learned, zero-shot **PII Redaction** Provider for the
[Latence framework](../../README.md): it detects PII in chunked text using a
[GLiNER](https://github.com/urchade/GLiNER)-PII checkpoint and produces the
masked/replaced text variant plus a structured, auditable list of PII spans,
running in-process on CPU via `transformers`/ONNX (no vLLM, no CUDA required).

It registers under the `latence.providers` entry-point group as
`redaction.gliner_pii`, so installing this package makes the Provider discoverable
by any Pipeline with **no change to `latence-core`** (ADR-0004). It is the learned
"model half" of the hybrid the S6 issue calls for — a drop-in behind the same
`PIIDetector` seam as the deterministic, dependency-free `redaction.hybrid_rule`
that ships in core. The two share the framework's policy + masking machinery
(`latence_core.redaction_policy`), so the only difference is detection.

## Zero-shot

The PII types are supplied per Pipeline in the Stage `config` — GLiNER-PII is a
zero-shot model, so the same checkpoint recognises `person`/`email`/`phone number`
for one corpus and other types for another with no retraining. No PII type set is
hardcoded.

## Metadata-aware policy

Redaction keys its action off the document's `Classification.sensitivity` via the
same `policies` config the in-core Provider uses: a `public` document can be masked
hard while an `internal` one is only tagged for review, from one Pipeline.

## Licensing

Both the model code and the default weights are permissive (ADR-0012, verified
2026-07-06):

- `gliner` library — **Apache-2.0** (code).
- Default checkpoint `urchade/gliner_multi_pii-v1` — **Apache-2.0** (weights); base
  model `urchade/gliner_multi-v2.1` is also Apache-2.0.

## Install

```bash
pip install latence-core latence-pii-gliner
```

This pulls `gliner` (and its `torch`/`transformers` stack). The model weights are
downloaded lazily on first run and cached by `transformers`.

## Use in a Pipeline

```yaml
  - name: redact
    capability: redaction
    provider: redaction.gliner_pii
    depends_on: [chunk]
    config:
      pii_labels: [person, email, phone number, credit card number]
      model: urchade/gliner_multi_pii-v1   # Apache-2.0 default; override to opt in to another
      threshold: 0.5                       # drop spans scoring below this
      default_action: mask                 # mask | replace | hash | tag
      policies:
        public: { action: mask }
        internal: { action: tag }          # detect-only for internal docs
```
