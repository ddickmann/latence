# latence-extract-gliner2

Fused zero-shot **Entity + Relation Extraction** Provider for the
[Latence framework](https://github.com/ddickmann/latence-framework), over the
[fastino GLiNER2](https://github.com/fastino-ai/GLiNER2) framework
(`fastino/gliner2-multi-v1`, mdeberta-v3 encoder).

One backbone does joint NER **and** RE in one pass, so a single Provider fulfils both the
Entity Extraction (S4) and Relation Extraction (S5) Capabilities (ADR-0013). It registers
`fused_entity_relation.gliner2` and is a **drop-in swap** for `relation.gliner_relex` (same
`FusedEntityRelationExtractor` seam, same `labels` + `relation_labels` config, same
mentions + relations + evidence + provenance + offset contract) — and the blessed **unified
default extraction engine** (ADR-0036, W9), consolidating extraction onto GLiNER2.

```yaml
- name: extract
  capability: fused_entity_relation
  provider: fused_entity_relation.gliner2
  config:
    labels: [organization, person]
    relation_labels: [works for, partner of]
    threshold: 0.3
    # quantize / compile default to true on cuda, false on cpu (W9 native gliner2 knobs)
```

## Performance (W9)

The `gliner2` framework exposes **native** `quantize` (fp16) and `compile`
(`torch.compile` + fused GPU kernels, no extra deps) load knobs — `GLiNER2.from_pretrained(id,
quantize=True, compile=True)`. They default on for cuda, off for cpu, and are **guarded**: any
quantize/compile fault degrades to a plain load, never a crash, at identical output. Speed
numbers come from the maintainer's GPU pod run (never fabricated offline).

## Licensing (ADR-0012, verified 2026-07-10 — weights AND code)

- **`gliner2` library** (PyPI + github.com/fastino-ai/GLiNER2): **Apache-2.0** (code).
- **`fastino/gliner2-multi-v1`** checkpoint: **Apache-2.0** (weights, per the HF model card;
  not gated / freely downloadable).

Both permissive; nothing downloads unless a Pipeline configures this Provider. The heavy
`gliner2[local]` stack ships with this package, not the thin `latence-core` (ADR-0016).
