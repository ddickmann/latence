# latence-disambig-embedding

The learned (embedding-based) **Disambiguation** Provider for the
[Latence framework](https://github.com/ddickmann/latence) — semantic entity
resolution that merges cross-surface duplicates the algorithmic `disambiguation.cascade`
misses ("IBM" ↔ "International Business Machines"), while keeping distinct entities apart.

Registered as `disambiguation.embedding` under the `latence.providers` entry-point group, it
is a **true drop-in for `disambiguation.cascade`** — same `DisambiguationRecord` contract, so a
stack swaps one for the other with one config line.

## Why surface + context

The pod validation proved a **bare-surface** embedding disambiguator *over-merges* distinct
entities, because short entity surfaces aren't discriminative with a sentence embedder
(cosine: `IBM ~ Apple` = 0.870 out-ranks the margin of genuine duplicates). Embedding each
mention's **surface + its surrounding context** separates them cleanly:

| pair | bare surface | surface + context |
|---|---|---|
| IBM ~ International Business Machines (same) | 0.883 | **0.964** |
| Apple ~ Apple Inc (same) | 0.933 | **0.988** |
| IBM ~ Apple (distinct) | 0.870 | **0.808** |

So this Provider embeds `surface + context` (reaching the context through the run's
`ChunkRecord`s, threaded to `disambiguate` by the Runner) and defaults its cosine threshold to
the context-safe `0.90`.

## Enterprise safety — no silent over-merge

A pair merges only when cosine ≥ `threshold` **and** the entity types are compatible. A
similar-but-below-threshold or type-incompatible pair is recorded as a **rejected**
`MergeDecision` (`applied=False`) with its score and reason — never silently folded, never
silently dropped. O(n²) surface pairing is blocked (exact surfaces group first) and capped by
`max_pairs`. A mention with no locatable chunk gracefully falls back to its bare surface.

## Install & configure

```bash
pip install latence-disambig-embedding
```

Offline by default (resolves the in-core, deterministic `embedding.hashing`). Opt into a learned
embedder in the Disambiguation Stage config:

```yaml
- name: disambiguate
  capability: disambiguation
  provider: disambiguation.embedding
  config:
    threshold: 0.90
    context_window: 160
    embedder:
      provider: embedding.sentence_transformers   # ships its own torch/ST stack (default Granite r2)
      config: { device: cuda }
```

> **ER-threshold caveat (ADR-0045).** The default `threshold: 0.90` was pod-calibrated on
> `multilingual-e5-small` (same ≈0.96–0.99 / distinct ≈0.76–0.81). The framework default embedder is
> now **IBM Granite Embedding r2**, whose cosine distribution differs — so **0.90 is not yet validated
> for Granite** and needs pod re-validation (pod paused). Set `threshold` explicitly per your embedder
> + corpus until then.

## License

Apache-2.0 (this package's own pure-Python code — the repository-wide licence). A learned embedder opt-in carries its own
verified weights license in its own Provider package.
