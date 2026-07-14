# latence-disambig-glinker

**Enterprise GLinker Disambiguation Provider** for the [Latence framework](https://github.com/ddickmann/latence-framework) — registers `disambiguation.glinker` under the `latence.providers` entry-point group (ADR-0004), a drop-in for the in-core `disambiguation.cascade` / `disambiguation.embedding` behind the same [`Disambiguator`](../latence-core/src/latence_core/capability.py) seam and the identical `DisambiguationRecord` contract.

This is the **enterprise entity-linking + resolution engine** ported (PORT-AS-IS) from the private `latenceai-dataset-intelligence` repo (ADR-0046), fusing two halves into one Stage:

1. **Corpus-internal KB + neural GLinker linking** — a self-referencing KB is built from the run's mentions (dominant-label vote, aliases, context descriptions), then the full GLinker **L2 → L3 → L4 → L0** pipeline (`knowledgator/gliner-linker-large-v1.0` biencoder + `gliner-linker-rerank-v1.0` reranker, FlashDeBERTa-accelerated) links each mention to its canonical KB entry over ±window-context inputs, with a **direct-KB fallback** for what the neural pass misses and provisional entities for the unresolved.
2. **Audited 5-strategy resolver** — the linked canonical entities are fused by exact → alias → acronym → substring → **IBM Granite STS embedding-by-label** (Apache-2.0, ADR-0045), with transitive merge + cycle detection and a **`MergeDecision` audit log** so nothing is silently over-merged (§19). Type-constrained: the acronym/substring rungs never cross entity labels, and the embedding rung groups by label. FAISS-GPU accelerates label groups > 1000 (optional; brute-force fallback when `faiss` is absent).

## Enterprise honesty

- **`compute="gpu"`** — the neural gliner-linker biencoder needs a GPU; on a CPU-only host the device seam skips-with-flag (never faked). If invoked without a GPU/model, the backend load fails gracefully and the Stage degrades to the deterministic direct-KB + string-cascade resolver path — **never a crash**.
- **`deterministic=False`** — the neural biencoder + FlashDeBERTa is not bitwise-reproducible; declared honestly. The string-cascade resolver rungs *are* deterministic, and the cluster emission ORDER is sorted (canonical-text), so the record stream is stably ordered.
- **No silent over-merge** — every applied merge carries reason + confidence + provenance, surfaced as contract `MergeDecision`s in the Quality Report.

## Install & use

This is a **GPU package** — not installed into the CPU-first dev/CI env (ADR-0016). On a GPU pod:

```bash
pip install -e packages/latence-disambig-glinker            # transformers/gliner path (default)
pip install -e "packages/latence-disambig-glinker[faiss]"   # + FAISS-CPU large-group acceleration
```

Swap it into any stack by one config line — see [`stacks/gpu-sota-glinker.yaml`](../../stacks/gpu-sota-glinker.yaml).

## Deferred: the vLLM serving backend

`VLLMGLiNERBackend` is a documented, seam-compatible follow-on (it needs the vllm-factory `deberta_gliner_linker` plugin). The transformers path above is the **complete enterprise default** — the full L2→L3→L4→L0 pipeline + audited resolver ships today; only the alternate serving backend defers.

## Licenses

All permissive (verified 2026-07-14; see the repo `THIRD-PARTY-LICENSES` + ADR-0046): gliner-linker-large/rerank **weights** Apache-2.0, `gliner` + `flashDeBERTa` **code** Apache-2.0, optional `faiss` **code** MIT.
