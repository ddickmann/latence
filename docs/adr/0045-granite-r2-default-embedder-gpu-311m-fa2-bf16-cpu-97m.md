# IBM Granite Embedding r2 is the default learned Embedder (GPU 311m FA2+bf16 / CPU 97m)

**Status: accepted** — W19-granite-embedder. Replaces `intfloat/multilingual-e5-small` as the default
checkpoint of the `embedding.sentence_transformers` Provider (`latence-embedder-st`) with **IBM Granite
Embedding r2**: the GPU tier `ibm-granite/granite-embedding-311m-multilingual-r2` (768-dim, loaded
bf16 + flash_attention_2 with a graceful sdpa fallback) as the declared default, and the CPU tier
`ibm-granite/granite-embedding-97m-multilingual-r2` (384-dim, fp32, no FA2) as the documented lighter
option. `multilingual-e5-small` (MIT) and `all-MiniLM-L6-v2` (Apache-2.0) remain documented tiny
fallbacks. No contract, schema, or record shape changes — the `Embedder` Capability (a `dimension`
property + `embed(texts) -> Iterator[list[float]]`) is unchanged; both consumers (the Export RAG-corpus
vector, ADR-0017, and the `disambiguation.embedding` entity-resolution embedder) pick the new default up
automatically. Builds on ADR-0004 (Capability protocols + Provider plugins), ADR-0007 (CPU-first),
ADR-0012 (permissive-only defaults, weights and code verified separately), ADR-0016 (thin core, heavy
deps per-provider), ADR-0017 (opt-in RAG-corpus embedder), ADR-0036 (Provider ecosystem: profile,
conformance, bake-off), and the W8 perf seam (`latence_core.providers.perf`: bf16/fp32 + FA2→sdpa
fallback via ST `model_kwargs`). Amends ADR-0012's and ADR-0036's named default.

## Context

The default learned Embedder was `intfloat/multilingual-e5-small` (MIT, 384-dim, XLM-RoBERTa backbone,
512-token window, requires the e5 `query: `/`passage: ` instruction prefix). IBM Granite Embedding r2
is a stronger, permissively-licensed, ModernBERT-based family that raises retrieval quality, extends the
context window 64× (512 → 32,768 tokens), covers 200+ languages, and — being ModernBERT — supports
flash-attention-2, which the framework's W8 perf seam already knows how to drive. Making it the default
is a one-checkpoint swap plus honest wiring of the two tiers and their determinism/threshold caveats.

## Verified specs and licenses (research-diligence; weights and code checked SEPARATELY, 2026-07-14)

Primary sources — the two HF model cards, fetched 2026-07-14:

- GPU tier: <https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2>
- CPU tier: <https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>

| Property | 311m (GPU tier, **default**) | 97m (CPU tier) |
|---|---|---|
| Weights license (card tag) | **`license: apache-2.0`** | **`license: apache-2.0`** |
| Backbone | ModernBERT (replaces XLM-RoBERTa; alternating attention, GeGLU, RoPE) — Apache-2.0 | ModernBERT (SiLU variant) — Apache-2.0 |
| Flash-attention-2 | supported ("Flash Attention 2.0 for improved efficiency") | supported (`pip install flash_attn`, optional) |
| Embedding dimension | **768** | **384** |
| Max context length | **32,768 tokens** | **32,768 tokens** |
| Prompt / prefix | **NONE** — card encodes text directly (`model.encode(sentences)`), CLS pooling | **NONE** — direct `model.encode(...)`, no query/passage/instruction prefix |
| Vector type | single-vector dense (bi-encoder, cosine) | single-vector dense (bi-encoder, cosine) |
| Multilingual retrieval (MTEB, 18 tasks) | 65.2 | 60.3 |
| Language coverage | 200+ (enhanced 52 + code) | 200+ (enhanced 52 + code) |
| Runtime library | sentence-transformers / transformers (Apache-2.0) | sentence-transformers / transformers (Apache-2.0) |

**Prompt convention (load-bearing, verified unambiguous on BOTH cards):** Granite r2 needs **NO**
instruction/query-document prefix — unlike e5's `query: `/`passage: `. The default `prefix` is therefore
the empty string `""`. (A wrong prefix silently degrades every vector, so this was verified directly from
the card usage examples, not assumed.) Vectors are still passed `normalize_embeddings=True` so the
exported column is unit-length (cosine convention, ADR-0017).

**License conclusion:** both tiers' **weights are Apache-2.0** (quoted card tag `license: apache-2.0`),
their **ModernBERT backbone is Apache-2.0**, and the **runtime library** (`sentence-transformers` /
`transformers`) is **Apache-2.0** (code). All permissive; recorded in the `ProviderProfile`
evidence-bearingly (`license="Apache-2.0"`, `license_code="Apache-2.0"`, `license_weights="Apache-2.0"`,
`license_source` citing both cards + the ST LICENSE, `license_verified_on="2026-07-14"`,
`license_verified=True`).

## Decision

1. **Default = Granite r2 311m (768-dim, no prefix).** `latence-embedder-st`'s `_DEFAULT_MODEL`,
   `_DEFAULT_DIMENSION`, `_DEFAULT_PREFIX` become the 311m checkpoint, 768, and `""`. The profile's
   `model_id`/`license`/`memory_mb` are updated accordingly.
2. **GPU tier via the W8 perf seam, not new code.** A GPU stack pins `device: cuda` +
   `attn_implementation: flash_attention_2`; the perf seam resolves **bf16** (auto on cuda) and **FA2**
   (when `flash_attn` is importable), threaded through ST `model_kwargs`, with the shared
   `load_with_fa2_fallback` retrying once on **sdpa** if the FA2 load raises — never a crash. On a CPU
   box the same seam resolves fp32 + sdpa, so the default still runs (heavier, but no CUDA), preserving
   the ADR-0007 "pip install → real embeddings, no CUDA" posture.
3. **CPU tier = Granite r2 97m (384-dim, fp32).** A one-line config (`model:` the 97m id, `dimension:
   384`, `device: cpu`) — the lighter laptop option. Wired as the `latence setup` wizard's CPU-device
   choice and documented across the embedder docs.
4. **Wiring.** GPU stacks (`gpu-sota.yaml` disambiguate embedder; `gpu-sota-pod.yaml` disambiguate +
   corpus-export embedder) and the `matrix/embed.yaml` GPU candidate use the 311m GPU tier. Both
   embedding consumers resolve the new default through the unchanged `Embedder` Capability.

## Determinism honesty (anti-false-green)

bf16 + FA2 on a GPU is **not** bitwise-reproducible, and learned float inference already is not promised
byte-identical across BLAS/hardware. So the `ProviderProfile` declares `deterministic=False` **honestly**
— it does not overclaim. This mirrors how the real-e5 embedder was already treated by the determinism
gate: in `latence stack validate`, a `device: cuda` embedder is **skipped-with-flag** on a CPU-only host
(the nested-embedder device pre-flight resolves it through the same `select_device` seam the Runner
uses), so the GPU FA2 stack is validated **structurally** (contracts/shape/no-PII-leak), never by
byte-identical determinism. The deterministic, dependency-free `embedding.hashing` reference stays the
default on the unit-test / golden / offline path, so `verify-local` stays green without any faked
determinism pass over a bf16/FA2 GPU run.

## The ER-threshold caveat (flag, don't guess)

The `disambiguation.embedding` merge threshold **0.90 was pod-calibrated on `multilingual-e5-small`**
(pod-measured: same-entity pairs with context ≈0.96–0.99, distinct ≈0.76–0.81). **Granite r2's cosine
distribution WILL differ**, so **0.90 is NOT validated for Granite**. It is retained as a sensible
documented default, but it must be **pod re-validated under Granite** (re-fit the same/distinct margins,
per `scripts/validate-disambig-on-gpu.md` with the Granite embedder) before it can be claimed calibrated.
The pod is currently paused, so this is an explicit **deferred follow-up**, flagged in the
`disambiguation.embedding` provider docstring/README, both GPU stacks, `matrix/disambiguation.yaml`, and
the validation script — never faked. This ADR does not claim the ER threshold is validated for Granite.

## Consequences

- The default RAG-corpus vector is 768-dim (GPU tier), 32k-context, no-prefix, multilingual — a quality
  upgrade at Apache-2.0 for both weights and code.
- No schema/contract change; existing pipelines pin `model`/`dimension` explicitly, so they are
  unaffected until they choose to adopt the new default.
- Open follow-up (pod, deferred): re-validate the ER threshold under Granite; measure the real
  311m-FA2-bf16 throughput/quality numbers on a GPU rig (until then the bake-off ST row is
  skipped-with-flag on CPU, never a fabricated number).
