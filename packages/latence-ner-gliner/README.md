# latence-ner-gliner

A learned, zero-shot **Entity Extraction** Provider for the
[Latence framework](../../README.md): it finds typed entity mentions in chunked
text using a [GLiNER](https://github.com/urchade/GLiNER) checkpoint, running
in-process on CPU via `transformers`/ONNX (no vLLM, no CUDA required).

It registers under the `latence.providers` entry-point group as `entity.gliner`,
so installing this package makes the Provider discoverable by any Pipeline with
**no change to `latence-core`** (ADR-0004). It is a drop-in alternative to the
deterministic, dependency-free `entity.gazetteer` that ships in core.

## Zero-shot

The entity types are supplied per Pipeline in the Stage `config` — GLiNER is a
zero-shot NER model, so the same checkpoint tags `person`/`organization` for one
corpus and `gene`/`drug` for another with no retraining. No label set is hardcoded.

## Licensing

Both the model code and the default weights are permissive (ADR-0012, verified
2026-07-06):

- `gliner` library — **Apache-2.0**.
- Default checkpoint `urchade/gliner_multi-v2.1` — **Apache-2.0** (v2.1 relicensed
  from the earlier CC-BY-NC-4.0; the CC-BY-NC checkpoints are never the default).

## Install

```bash
pip install latence-core latence-ner-gliner
```

This pulls `gliner` (and its `torch`/`transformers` stack). The model weights are
downloaded lazily on first run and cached by `transformers`.

## Use in a Pipeline

```yaml
  - name: entities
    capability: entity_extraction
    provider: entity.gliner
    depends_on: [chunk]
    config:
      labels: [person, organization, location, date]
      model: urchade/gliner_multi-v2.1   # Apache-2.0 default; override to opt in to another
      threshold: 0.5                      # drop mentions scoring below this
```
