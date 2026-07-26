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
    # quantize defaults true on cuda / false on cpu (W9 native gliner2 fp16 knob)
    # compile defaults OFF (opt-in) — W9-scaling recompile-thrash guard, see below
    # intra_op_threads: 8   # W9-scaling many-core oversubscription cap (<=0 disables)
    # batch_size: 16        # progress cadence, clamped to 64
    # label_floor: per_chunk  # default; `universal` restores the config ∪ induced union
```

## Labels are a PER-CHUNK floor (T2)

When an optional `schema_induction` Stage runs, each chunk carries the types induced **for that
chunk**. This Provider then scores the chunk against **its own** induced types — the `labels` /
`relation_labels` above are the fallback for a chunk induction said nothing about, not a set
stamped onto every chunk:

| `label_floor` | effective labels for a chunk that induced types | for a chunk that induced none |
| --- | --- | --- |
| `per_chunk` (default) | the chunk's own induced types | the config lists |
| `universal` | config ∪ induced | the config lists |

`per_chunk` is the default because zero-shot precision **and** the span-scoring activation both
degrade with label count: prompting a chunk with ~40 corpus-wide types means ~35 irrelevant labels
competing for its spans, and a label dimension large enough to turn a routine forward into a
multi-GiB allocation. The trade is explicit — `universal` is recall insurance (a chunk whose
induction under-called is still scored for the config types); `per_chunk` trades that insurance for
label-count purity, which is sound precisely because induction is per chunk and the type vocabulary
is reconciled downstream rather than imposed upstream.

Without induction — including "predefined-only" mode, where the `induce` Stage is skipped entirely
— every chunk is scored against exactly the config lists under either value, byte-identically.

## Performance (W9) + scaling hardening (W9-scaling)

The `gliner2` framework exposes **native** `quantize` (fp16) and `compile`
(`torch.compile` + fused GPU kernels, no extra deps) load knobs. `quantize` defaults on for cuda,
off for cpu, **guarded** (any fault degrades to a plain load, never a crash, at identical output).

**`compile` defaults OFF (opt-in)** on this provider: gliner2 compiles with `dynamic=True`, which
**recompiles per chunk-shape** and thrashes `torch._dynamo` past its recompile_limit on a real
varied-length corpus (CPU-pinned, GPU idle, no progress). Enable `compile: true` ONLY for a
homogeneous / uniform-length workload. Two further scaling guards for a many-core GPU host: torch
**intra-op threads are capped** at load (`intra_op_threads`, default 8) so gliner2's internal
batching can't oversubscribe to a CPU spin with the GPU idle; and the fp16 model is **explicitly
moved to cuda and its device logged** (if a run is CPU-bound with the GPU idle despite this, set
`quantize: false`). Extraction **streams in bounded batches with per-batch progress logging** so a
large corpus never *appears* hung. Speed / GPU-placement numbers come from the maintainer's GPU pod
run (never fabricated offline).

## Licensing (ADR-0012, verified 2026-07-10 — weights AND code)

- **`gliner2` library** (PyPI + github.com/fastino-ai/GLiNER2): **Apache-2.0** (code).
- **`fastino/gliner2-multi-v1`** checkpoint: **Apache-2.0** (weights, per the HF model card;
  not gated / freely downloadable).

Both permissive; nothing downloads unless a Pipeline configures this Provider. The heavy
`gliner2[local]` stack ships with this package, not the thin `latence-core` (ADR-0016).
