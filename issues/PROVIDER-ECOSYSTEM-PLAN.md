# Phase-3 — Provider Ecosystem & Bake-off Plan

Make "model-agnostic" real and provable: many real providers per Stage, config-driven, CPU/GPU, with a harness that (A) validates a **default stack** end-to-end, (B) runs a **per-Stage bake-off** across many providers, and (C) enforces **enterprise readiness across the board**. Architecture: **ADR-0036**. Builds on the Capability/Provider seam (ADR-0004) — no seam rewrite.

## The three goals, mapped

- **(A) Default stack, end-to-end.** A blessed CPU-first `stacks/default.yaml` (+ `gpu-sota.yaml`), validated by a stack-validation harness (the G1 E2E generalized): contracts + KG + RAG + Quality Report + reliability on a corpus, one command.
- **(B) Stage-by-stage, many plugins.** A config-driven bake-off (`matrix/<stage>.yaml`) that holds every other Stage at the default and runs the SAME corpus + gold-set through each candidate provider → a signed `MATRIX-RESULTS-<stage>.md` comparing quality (gold-set P/R/F1 or stage metric), latency/throughput, peak memory, cost, license, determinism, error rate.
- **(C) Enterprise readiness across the board.** A reusable Provider Conformance suite parametrized over EVERY registered provider: valid typed records (Provenance/Classification/offsets), graceful failure on adversarial/oversized input, resource bounds honored, license verified+permissive-or-opt-in, determinism-or-documented, no PII/secret leak. "Passes conformance" = enterprise-ready. Runs in CI for CPU providers; GPU where available.

## Ordering — Foundation first, then adapters (same discipline as Phase-2)

Build the harness/seam plumbing (Wave 0) **before** the adapter packages (Wave 1+), so every adapter is written against a stable conformance/profile contract and immediately measurable. Adapters are independent of each other (parallelizable). Everything goes through the verify gate; merges on green; license verified per adapter (research-diligence directive: weights+code separately; UNVERIFIED is flagged, never guessed).

### Wave 0 — Foundation (serial-ish; core seam)
- **P3-F1 — `ProviderProfile` + device handling. ✅ BUILT (#86).** Each provider declares compute (cpu|gpu|either), memory_mb, model_id + verified license, deterministic, batch, cost_per_1k; registry surfaces it (`ProviderRegistry.profile()`/`.profiles()`, missing = None, import-error = skip-and-flag); `device: auto|cpu|cuda` config threaded + routed by the torch-free `latence_core.providers.device` seam (`LATENCE_CUDA`-override-driven `cuda_available()`, typed `DeviceDecision`); GPU providers skipped-with-flag on CPU-only hosts (the G1 posture, never raised); profile + device columns land in the Quality Report `StageMetrics` (v16). In-core references + `entity.gliner` declare honest profiles. (ADR-0036 §2 — status note in the ADR.)
- **P3-F2 — `AdapterBase`.** Thin optional base carrying device selection, batching, native-exception→typed-taxonomy (ADR-0034) mapping, and offset/Provenance-preservation helpers — so a new adapter is small + consistent. (ADR-0036 §1)
- **P3-F3 — Provider Conformance suite (C).** Parametrized over every registered provider; the enterprise-readiness gate; wired into CI. (ADR-0036 §5)
- **P3-F4 — Stacks + stack-validation harness (A).** `stacks/default.yaml`, `stacks/gpu-sota.yaml`, a `latence stack validate <name>` harness. (ADR-0036 §3)
- **P3-F5 — Bake-off / Provider Matrix harness (B).** `matrix/<stage>.yaml`, the comparison runner, the `MATRIX-RESULTS-<stage>.md` artifact, GPU-gating. (ADR-0036 §4)

### Wave 1 — Adapter packages (parallelizable; after Wave 0; each license-verified first)
- **P3-A-parse** — `latence-parser-docling` (docling), `latence-parser-kreuzberg` (kreuzberg), and GPU: `latence-parser-paddle` (PaddleOCR-VL, Apache-2.0 ✓), `latence-parser-lighton` (LightOnOCR, Apache-2.0 ✓). **License-verify FIRST:** docling, kreuzberg, lightparse, marker, unstructured — flag any non-permissive (marker/unstructured have historically had restrictive terms) as opt-in-or-drop.
- **P3-A-chunk** — `latence-chunk-rust`: port the Rust chunking engine (superpod `text_processing` wheel) + pure-Python fallback (ADR-0015 pattern), behind the Chunk seam.
- **P3-A-embed** — `latence-embedder-st` (sentence-transformers: e5-small MIT ✓, MiniLM Apache ✓, BGE — verify), `latence-embedder-colbert` (from superpod, GPU — verify), `latence-embedder-gemma` (EmbeddingGemma — Gemma license, opt-in + flow-down ✓).
- **P3-A-ner-re** — GLiNER model variants for NER/RE (config-selected checkpoints, Apache-2.0 base ✓, per-checkpoint verify), and `latence-re-span-predictor` (the custom span-predictor encoder from superpod, decoupled from the vLLM pooling patch).
- **P3-A-pii** — `latence-pii-presidio` (Microsoft Presidio, MIT — verify) + additional GLiNER-PII checkpoints.

### Wave 2 — Assurance (C, consolidation)
- **P3-C1 — Provider-readiness report.** Run the conformance suite + the per-stage bake-offs across all installed providers; produce a consolidated `PROVIDER-READINESS.md` (the enterprise "which providers are production-ready, at what quality/cost/license" matrix). GPU/endpoint numbers where a rig exists (#64), flagged otherwise.

## Honest boundary
CPU adapters + the harness run fully here; GPU-model adapters (Paddle/LightOn/GLiNER-large/ColBERT) and endpoint providers are conformance/seam-verified with mocks, but real quality/throughput needs a GPU/perf rig (#64) — the bake-off produces those where the environment exists, never fabricated.
