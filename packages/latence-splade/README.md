# latence-splade

The framework's **real multilingual SPLADE** `SparseEmbedder` Provider (#162, ADR-0053/0049/0050/0012/0016).
It is the same index-time `SparseEmbedder` Capability as the in-core deterministic `sparse.hashing`
reference — swap one for the other with **one config line** — but its vector is not plain
`SPLADE(text)`. It is the ADR-0053 **structured composition**:

```
sparse = model_weight · SPLADE(text)  ⊕  entity_boost · confidence-gated entity_terms
                                      ⊕  metadata_boost · selected metadata_terms
```

the model's learned term-weights **unioned** with the pipeline's own confidence-gated NER entity
surface forms and selected metadata tokens, in the model's vocabulary, summed on collision. It
exploits an edge a vanilla SPLADE shop lacks (a latence corpus already ships entities + metadata),
runs **training-free**, and emits the parallel `sparse_indices` / `sparse_values` columns a store's
sparse index ingests (via #160's Export seam).

## The composition (ADR-0053)

| Leg | Source | Gate |
|---|---|---|
| model term-weights | the SPLADE MLM (`max_t log(1+relu(logits))`) | over **redacted** corpus text (default) |
| entity terms | `EntityMention` **surface forms** → model subword ids | dropped below `confidence_threshold`; **not injected if the surface didn't survive redaction** |
| metadata terms | operator-selected `metadata_fields` values | operator-chosen (safe) fields only |

Two modes: **`inject`** (default term-injection — the model's weights are kept as-is, boosted
structured terms added on top) and **`weighted`** (the model leg scaled by a tunable `model_weight`,
so its contribution dials against the structured legs). One composition runs at **both** clocks —
index time (this Provider) and query time (`encode_composed` over a query string).

## Redaction by default (ADR-0053 PII safety)

The model leg runs over the chunk's **redacted** corpus text (Export passes `masked_content`), so the
sparse signal never re-leaks PII the corpus removed. Entity injection is redaction-aware too: with
`redact: true` (default) an entity whose surface no longer occurs in the redacted text is **not**
injected, so a masked PII surface is never re-introduced as a term. `redact: false` is an explicit
opt-in for fully-internal deployments.

## Usage

```yaml
# In a corpus Export (ADR-0017 opt-in), alongside the dense embedder:
config:
  sparse_embedder:
    provider: sparse.splade
    config:
      model: <a multilingual SPLADE checkpoint you have license-cleared>   # REQUIRED, no default
      mode: inject                 # or "weighted"
      confidence_threshold: 0.5    # entities below this are dropped (weak NER never pollutes)
      entity_boost: 1.0
      metadata_boost: 1.0
      metadata_fields: [classification]
      redact: true                 # never re-inject a redacted surface
```

## No default model — license discipline (ADR-0012/0053)

**No checkpoint is pinned.** `model` is a **required** config value: ADR-0053 forbids hardcoding a
model until its **weights AND code** licenses are verified *separately* (the research-diligence
directive / ADR-0012). The `ProviderProfile` declares `model_id=None` and an honest `UNVERIFIED`
weights license (`license_verified=False`) rather than fabricating a permissive claim for a model the
operator has not chosen. The package **code** and the `transformers` library are Apache-2.0
(`license_code`), recorded truthfully.

> Candidate multilingual sparse models (each REQUIRES separate weights+code verification before it is
> wired — do **not** treat this as a clearance): `BAAI/bge-m3` (lexical-weights output), the OpenSearch
> `opensearch-neural-sparse-encoding-*` family, and the `naver/splade-*` family. NB several SPLADE
> checkpoints (e.g. parts of the naver line) are **CC-BY-NC** (non-commercial) — a restricted license
> that must be flagged opt-in, never a permissive default. This package blesses none of them.

## Offline / CI discipline (anti-false-green)

The heavy `transformers`/`torch` stack is **not** installed in the CPU-first, offline test env
(ADR-0007/0016). The package installs `--no-deps`; the model seam
(`latence_splade.model.SpladeModel`) is **dependency-injected**, so the full composition + gating are
exercised offline against a **faithful fake** model, and the real `TransformersSpladeModel`
tokenize→pool path is exercised against a fake `transformers` module (the same honest boundary the
gliner/embedder-st packages use). A real-weights test is present and **gated** on
`importlib.util.find_spec("transformers")` (skipped offline). Real-model quality validation is done
out of band with cleared weights.
