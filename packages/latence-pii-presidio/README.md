# latence-pii-presidio

A real **Microsoft [Presidio](https://github.com/microsoft/presidio)** PII **Redaction**
Provider for the [Latence framework](../../README.md): it detects PII in the parsed
document markdown and produces the masked/replaced text variant plus a structured,
auditable list of PII spans, running in-process on CPU via a spaCy NER model plus
Presidio's pattern + context recognizers (no CUDA required; GPU-optional).

It registers under the `latence.providers` entry-point group as `redaction.presidio`,
so installing this package makes the Provider discoverable by any Pipeline with **no
change to `latence-core`** (ADR-0004). It is a **true drop-in** behind the same
`PIIDetector` seam as the deterministic, dependency-free `redaction.hybrid_rule` and the
learned zero-shot `redaction.gliner_pii` — swap the provider with **one config line**, no
other change. All three share the framework's policy + masking machinery
(`latence_core.redaction_policy`) and the H-C1 sensitive-doc no-op floor, so the only
difference is *detection*: Presidio pairs a spaCy NER model with pattern/context
recognizers — a genuinely different detection engine, so it is a real bake-off alternative.

Built through `AdapterBase` (P3-F2/ADR-0036 §1): the shared device/batch/guard/offset seams
do the cross-cutting glue one right way. Presidio's char spans land on the parsed-document
coordinate system, and each span's source page is resolved through the ADR-0031
`PageOffsetIndex` seam — never a weaker hand-rolled mapper.

## Metadata-aware policy

Redaction keys its action off the document's `Classification.sensitivity` via the same
`policies` config the in-core Provider uses: a `public` document can be masked hard while
an `internal` one is only tagged for review, from one Pipeline. Presidio's UPPER_SNAKE
entity types (`EMAIL_ADDRESS`, `PERSON`…) are mapped to the framework's lower-case
`pii_type` vocabulary (`email`, `person`…), so a `types: ["email"]` policy covers a
Presidio `EMAIL_ADDRESS` unchanged — the drop-in needs no policy rewrite.

## No PII leak

The raw PII value is never stored on a `PIISpan` (only the offset, the type, and the
deterministic placeholder), and a native Presidio/spaCy failure is mapped to a typed
`ProviderError` carrying only the provider, the op, and the native exception's *type name*
— never its message or any input text (C2/C6). A PII provider that leaked detected PII into
an error or log would be the worst kind of failure; this one is proven not to.

## Licensing

Both the code and the default weights are permissive (ADR-0012, verified 2026-07-08):

- `presidio-analyzer` library — **MIT** (code; "Copyright (c) Presidio Contributors").
- `spacy` library — **MIT** (code).
- Default spaCy model `en_core_web_sm` — **MIT** (weights); `en_core_web_lg` is also MIT.

## Install

```bash
pip install latence-core latence-pii-presidio
python -m spacy download en_core_web_sm
```

This pulls `presidio-analyzer` (and its spaCy stack). The spaCy model is installed
out-of-band (spaCy models are not PyPI dependencies) and loaded lazily on first run.

## Use in a Pipeline

```yaml
  - name: redact
    capability: redaction
    provider: redaction.presidio
    depends_on: [parse]
    config:
      model: en_core_web_sm                  # MIT default; en_core_web_lg (also MIT) for more recall
      language: en
      entities: [EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, PERSON, LOCATION]
      threshold: 0.5                         # Presidio score_threshold; drop spans below it
      default_action: mask                   # mask | replace | hash | tag
      policies:
        public: { action: mask }
        internal: { action: tag }            # detect-only for internal docs
```

Redaction scans the parsed document markdown (not the markup-stripped chunk text), so it
depends on the Parse Stage — exactly like `redaction.hybrid_rule` and `redaction.gliner_pii`.
