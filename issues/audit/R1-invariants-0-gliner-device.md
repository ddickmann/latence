# R1-invariants-0-gliner-device — `entity.gliner` / `relation.gliner_relex` never place their model (the #191 fix was wired to only 1 of 3 plain-gliner providers)

**Severity:** major
**Invariants violated:** ADR-0036 §2 (honour-the-resolved-device-or-fail-loudly); ADR-0034 (NO SILENT
ERRORS — a `device: cuda` config that is silently ignored is exactly the swallowed failure the ADR
forbids); the #191 lesson ("silence is the bug").
**Location:**
* `packages/latence-core/src/latence_core/providers/perf.py` — `_effective_is_cuda` (:56-67),
  `load_then_cast` (:262-278) — the seam whose bf16 cast keys off a bare availability probe when the
  caller resolves no device.
* `packages/latence-ner-gliner/src/latence_ner_gliner/provider.py:209` — `load_then_cast(..., None)`.
* `packages/latence-relation-gliner/src/latence_relation_gliner/provider.py:210` — same.

> NOTE on the filename: `issues/audit/R1-invariants-0.md` already exists on `main` (a prior campaign's
> critical ticket). It is deliberately NOT overwritten; this round's ticket carries a suffix so both
> remain traceable.

## What to build

Wire the **same** #191 fix `redaction.gliner_pii` already has into the other two plain-`gliner`
providers:

1. Resolve the `device` config value (`auto|cpu|cuda`) **once at construction** through the ADR-0036
   device seam (`select_device(requested, cls.profile())`), so an invalid string is an operator
   `ConfigError` before any weights are downloaded.
2. A **skip** decision (`device: cuda` on a CPU-only host) becomes a loud `ConfigError` at load —
   never the pre-#191 silent CPU fallback.
3. Pass the **resolved** device into `load_then_cast` so the bf16/fp32 decision keys off where the
   model will actually run, not off a bare `cuda_available()` probe.
4. Call `place_on_device(model, device, provider_name=...)` after the load, so the model is actually
   moved onto the resolved device and the placement is **verified** (a model that reports a
   different device is a `ProviderError`, not a logged shrug).

Structural teeth: the AST regression guard must require `place_on_device` in **every** plain-`gliner`
provider source, so a fourth one that lands without it is caught the moment it is written.

## Failure scenario (the confirmed defect)

On a CUDA host (the Blackwell pod), run a pipeline wiring `entity.gliner` — a documented provider
(`examples/demo-pipeline.swap.yaml:58`, `matrix/entity.yaml:20`):

* The provider resolves **no** device at all: `device: cuda` in its Stage config is read by nobody —
  no `ConfigError`, no warning, no log line.
* `load_then_cast(GLiNER.from_pretrained, id, config, None)` → `_effective_is_cuda(None)` falls back
  to `cuda_available()` → `True` → `resolve_dtype_name` → `bfloat16` → `model.to(torch.bfloat16)`.
* But `GLiNER.from_pretrained` map-locations to **CPU** (asserted by this very stream in
  `place_on_device`'s docstring, perf.py:200-206) and nothing ever calls `.to(device)`.

Result: **bf16-cast weights running eager on CPU** — slower than the fp32 CPU path the seam intends,
numerically different from it, and contradicting both `load_then_cast`'s own docstring ("The cast is
applied only when … the model is on cuda") and `describe()`'s `perf_note` ("bf16 via .to() on GPU /
fp32 on CPU"). Nothing warns; the operator sees only the ~10x-class throughput cliff #191 measured
(~11.5 s/chunk with the GPU idle) — the exact failure mode #191 says must never happen, left
unfixed in 2 of the 3 plain-gliner providers.

## Acceptance criteria

1. `GlinerEntityExtractor` and `GlinerRelexExtractor` resolve `config["device"]` through
   `select_device` at construction; an unknown string (`"gpu"`) is a `ConfigError` naming the
   provider, raised **before** any model load.
2. On `LATENCE_CUDA=1`, a provider configured `device: cuda` (or `auto`) loads a model whose
   **parameters reside on `cuda`** — asserted by reading the loaded model's `parameters()[0].device`,
   not by asserting a call happened.
3. On `LATENCE_CUDA=0`, `device: cuda` raises `ConfigError` ("no CUDA device is available") instead of
   silently running on CPU.
4. A model that reports the wrong device after placement raises `ProviderError` (the `place_on_device`
   verification), for both providers.
5. The dtype cast follows the **resolved** device: `cuda` → bf16 applied via `.to()`; `cpu` → no cast.
6. Output contract is device-invariant: the emitted mentions/relations (labels, offsets, order) are
   identical across the cpu and cuda lanes — only throughput moves.
7. The AST perf guard (`packages/latence-core/tests/test_perf_fp32_regression_guard.py`) fails if any
   plain-`gliner` provider source loads a model without calling `place_on_device`.
8. `make test` stays green at or above the 2985-passed / 28-skipped baseline; no test weakened.
