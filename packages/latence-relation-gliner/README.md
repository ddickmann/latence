# latence-relation-gliner

A learned, zero-shot **fused Entity + Relation Extraction** Provider for the
[Latence framework](../../README.md): a single
[gliner-relex](https://huggingface.co/knowledgator/gliner-relex-multi-v1.0)
checkpoint does joint NER **and** Relation Extraction in one pass, running
in-process on CPU via `transformers`/ONNX (no vLLM, no CUDA required).

It registers under the `latence.providers` entry-point group as
`relation.gliner_relex`. Because it is a **Fused Provider** (ADR-0013), one Stage
fulfils both the Entity Extraction (S4) and Relation Extraction (S5) Capabilities —
the Pipeline is not forced to one Provider per Stage, and the Runner threads the
mentions and relations downstream from the one fused Stage without double-extracting.

## Zero-shot

Both the entity label set and the relation label set are supplied per Pipeline in
the Stage `config` — the same checkpoint tags `person`/`organization` and relates
them with `works_for` for one corpus, and tags `gene`/`drug` and relates them with
`targets` for another, with no retraining. No label set is hardcoded.

## Licensing

Both the model code and the default weights are permissive (ADR-0012, verified
2026-07-06):

- `gliner` library — **Apache-2.0**.
- Default checkpoint `knowledgator/gliner-relex-multi-v1.0` — **Apache-2.0**.

## Install

```bash
pip install latence-core latence-relation-gliner
```

This pulls `gliner` (and its `torch`/`transformers` stack). The weights download
lazily on first run and are cached by `transformers`.

## Use in a Pipeline (fused: one Stage, both S4 + S5)

```yaml
  - name: extract
    capability: fused_entity_relation
    provider: relation.gliner_relex
    depends_on: [chunk]
    config:
      labels: [person, organization, location]
      relation_labels: [works_for, located_in, founded_by]
      threshold: 0.5
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [extract]                # exports BOTH mentions and relations
```
