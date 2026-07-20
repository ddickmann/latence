# R1-invariants-1 — the #190 "shared JIT-neutralizing seam" guard is hollow

**Severity:** major
**Where:** `packages/latence-core/tests/test_perf_fp32_regression_guard.py`
(`_JIT_NEUTRALIZING_LOADERS`, `test_mdeberta_providers_load_through_a_jit_neutralizing_seam`)
against `packages/latence-core/src/latence_core/providers/perf.py`
(`load_with_fa2_fallback` — no `neutralize_torch_jit` call).

## What is wrong

The sole regression guard for #190's acceptance criterion — *"the JIT fix lives at the SHARED seam
and every mdeberta Provider inherits it"* — reuses
`_model_from_pretrained_passed_to_perf_seam`, whose accept-set is
`_PERF_SEAM_FUNCS = ("load_with_fa2_fallback", "load_then_cast", "load_gliner2_optimized")`.

Only `load_then_cast` and `load_gliner2_optimized` call `neutralize_torch_jit`;
`load_with_fa2_fallback` does not. The test's own `_JIT_NEUTRALIZING_LOADERS` constant, which names
the correct two, appears **only inside the failure message** and is never used in an assertion.

## Failure scenario

A sixth mdeberta-backed Provider (or a refactor of one of the existing five) written as

```python
model = load_with_fa2_fallback(GLiNER.from_pretrained, self._model_id, perf)
```

is accepted by the guard: `seam_routed` is non-empty (first assertion passes) and the offender loop
finds nothing (second assertion passes). The Provider is nevertheless fully exposed to the
Blackwell / CUDA-13 `libnvrtc-builtins.so.13.0` crash on the first forward, because the TorchScript
JIT was never neutralized. The test cannot fail for the case it is named after, so #190's
shared-seam AC is unenforced.

## What to build

1. Parameterise the seam collector on the accepted loader names instead of hard-coding
   `_PERF_SEAM_FUNCS`.
2. Make the mdeberta guard scan against `_JIT_NEUTRALIZING_LOADERS` only, and flag **every**
   mdeberta model `from_pretrained` reference (including one handed to a non-neutralizing loader
   such as `load_with_fa2_fallback`, which appears as an `ast.Attribute`, not an `ast.Call`).
3. Keep `_JIT_NEUTRALIZING_LOADERS` honest: assert against the real `perf.py` source that each
   named loader actually calls `neutralize_torch_jit`, and that a perf-seam loader **not** in the
   list does not.

## Acceptance criteria

- [ ] A synthetic mdeberta model load routed through `load_with_fa2_fallback` is FLAGGED by the
      guard (a meta-test asserts this; it fails on the pre-fix accept-set).
- [ ] The five real mdeberta Providers still pass (they load via `load_then_cast` /
      `load_gliner2_optimized`).
- [ ] `_JIT_NEUTRALIZING_LOADERS` is referenced by an assertion, not only by a message, and is
      cross-checked against `perf.py`'s real bodies.
- [ ] `make test` stays green at or above 2985 passed / 28 skipped; no test weakened or deleted.
