# Embedders — a resolution signal

An **Embedder** maps text to a fixed-width vector. It is never a Stage of its own: a Stage names
one in its `config['embedder']` block. Since
[ADR-0068](adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)
the framework ships no query-time retrieval tooling, and an Embedder's job is to be one signal the
**resolution phase** weighs. The framework names only the `Embedder` **Capability** (a `dimension`
property + `embed(texts) -> Iterator[list[float]]`); Providers fulfil it.

## Where an Embedder is used

| Stage | Provider | What the Embedder decides | Default when no `embedder` block is given |
|---|---|---|---|
| Disambiguation | `disambiguation.embedding` | which mentions are the same entity — the mention is embedded with its surrounding chunk context | in-core `embedding.hashing` |
| Disambiguation | `disambiguation.glinker` | the STS merge rung over entity labels — only when `use_embedding_merge: true` (off by default) | in-core `embedding.hashing` |
| Type Consolidation | `type_consolidation.cascade` | the semantic rung that folds drifted type labels (`org` ↔ `organization`) — only when `use_embedding_merge: true` (off by default) | in-core `embedding.hashing` |

Two findings bind how these rungs are configured
([ADR-0046](adr/0046-enterprise-glinker-entity-linking-and-audited-resolver.md)):
a retrieval-oriented embedder puts distinct **short names** (people, dates, organisations) too
close together, so an embedding merge rung over bare surfaces over-merges. That is why both
`use_embedding_merge` switches default to **off**, and why `disambiguation.embedding` embeds a
mention with its context rather than its surface alone.

One further consumer remains from before ADR-0068: the corpus Export (`export.jsonl_parquet`)
still accepts the same `embedder` block (ADR-0017) and, when given one, writes an `embedding`
column beside the exported text. It is opt-in, and it is what the Embedder bake-off below measures.

## The three reference Providers (one Capability, one config line to swap)

| Provider | Package | What it is | License (code / weights) | Runs offline in CI |
|---|---|---|---|---|
| `embedding.hashing` | in-core | Deterministic, dependency-free token-hash bag-of-words (the CPU reference and offline fallback — **not** semantic) | Apache-2.0 / — | yes (real) |
| `embedding.sentence_transformers` | `latence-embedder-st` | Learned in-process embedder over `sentence-transformers`, default **IBM Granite Embedding r2** — GPU tier `granite-embedding-311m-multilingual-r2` (768-dim, bf16+FA2), CPU tier `granite-embedding-97m-multilingual-r2` (384-dim, fp32) | Apache-2.0 / Apache-2.0 | no (heavy dep; verified against a stub, real weights on a rig) |
| `embedding.endpoint` | `latence-embedder-endpoint` | Any OpenAI-compatible / vLLM / Triton `/embeddings` endpoint | Apache-2.0 / — (calls a user endpoint) | no (endpoint; verified against a stub) |

Swapping is a **one-line** config change on the Stage that names it (ADR-0007):

```yaml
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.embedding
    depends_on: [entities, relations]
    config:
      embedder:
        provider: embedding.sentence_transformers   # was: embedding.hashing
        config:
          # GPU tier: Granite r2 311m, Apache-2.0, 768-dim, bf16 + flash_attention_2.
          model: ibm-granite/granite-embedding-311m-multilingual-r2
          dimension: 768                            # validated against the loaded model
          device: cuda                              # bf16 auto on cuda (perf seam)
          attn_implementation: flash_attention_2    # -> sdpa if flash_attn absent / load raises
          # CPU tier: model: ibm-granite/granite-embedding-97m-multilingual-r2, dimension: 384, device: cpu
```

## Granite r2 model-card convention (`latence-embedder-st`)

The default learned Embedder is **IBM Granite Embedding r2** (Apache-2.0, ModernBERT-based,
single-vector dense, 32,768-token context, 200+ languages; ADR-0045). It needs **no
instruction/query-document prefix** — text is encoded directly, so the default `prefix` is `""` —
and the Provider passes `normalize_embeddings=True`, so every vector is unit-length (cosine
convention). The GPU tier loads bf16 + `flash_attention_2` with a graceful `sdpa` fallback; the CPU
tier loads fp32 + `sdpa`. Cards:
<https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2> and
<https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>.

## Bake-off

`matrix/embed.yaml` (`stage: embedding`, `stage_name: export_corpus`) compares Embedders by swapping
only the corpus Export's `embedder` block, holding every other Stage constant. On a CPU-only host
the `embedding.hashing` row runs for real (measured, deterministic) and the learned candidate is
**skipped-with-flag** through the same `select_device` seam — never a fabricated number. Run it with
`latence bake-off matrix/embed.yaml`. Embedding has no gold set, so that matrix reports
latency / memory / license / determinism only; the resolvers themselves are compared in
`matrix/disambiguation.yaml`.

## Offline / CI discipline (learned + endpoint Providers)

The heavy/endpoint Provider packages are installed `--no-deps` in the CPU-first, offline test env
(ADR-0007/0016); their tests monkeypatch the heavy dep (`SentenceTransformer` / the `openai` client)
with a deterministic fake, and each ships a **real** path gated on the dep being importable (skipped
offline, run where the model/endpoint exists). The Provider Conformance suite verifies them against a
stub and flags the row `conformance-verified against stub; live numbers need a rig`.
