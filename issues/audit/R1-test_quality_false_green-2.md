# R1 — The #190 JIT-neutralizing-seam guard accepts a loader that does NOT neutralize the JIT

- **Severity:** major
- **Dimension:** test-quality / anti-false-green
- **Branch:** feat/release-fleet-integration
- **Tickets:** #190 (shared JIT-neutralizing seam)
- **Status:** WONTFIX — DUPLICATE, already fixed on this branch by commit `49b2e53`
  (ticket `issues/audit/R1-invariants-1.md`). Verified non-vacuous; see Evidence.

## What was wrong (the reported defect)

`test_mdeberta_providers_load_through_a_jit_neutralizing_seam` computed its allowlist with
`_model_from_pretrained_passed_to_perf_seam`, which matched the **W8 fp32** accept-set
`_PERF_SEAM_FUNCS = ("load_with_fa2_fallback", "load_then_cast", "load_gliner2_optimized")`.

Only the latter two call `neutralize_torch_jit()` (`perf.py:270`, `perf.py:530`);
`load_with_fa2_fallback` never does. The constant that encodes the real #190 rule,
`_JIT_NEUTRALIZING_LOADERS = ("load_then_cast", "load_gliner2_optimized")`, was declared but
referenced **only inside the failure-message f-string** — never in an assertion. The test did not
test the property its name and docstring claim.

## Failure scenario

A sixth mdeberta Provider whose only model load is
`m = load_with_fa2_fallback(GLiNER2.from_pretrained, model_id, perf)`:

- `seam_routed` is non-empty, so `assert seam_routed` passes;
- the bypass scan flags nothing → **the guard is GREEN**;
- yet `load_with_fa2_fallback` never neutralizes the TorchScript JIT, so on a Blackwell/CUDA-13
  host that Provider's first forward dies with
  `RuntimeError: ... failed to open libnvrtc-builtins.so.13.0` — exactly the #190 field crash the
  guard exists to make unshippable.

## Acceptance criteria

1. The #190 guard's accept-set is `_JIT_NEUTRALIZING_LOADERS`, asserted (not merely formatted into
   a message). — **MET** (`jit_seam_offenders`, line 331; parametrized test, line 352).
2. A model load routed through `load_with_fa2_fallback` is FLAGGED by the #190 guard. — **MET**
   (`test_jit_guard_flags_a_load_routed_through_the_non_neutralizing_fa2_loader`).
3. Bare and indirected loads (`GLiNER2.from_pretrained`, `gliner2_cls.from_pretrained`,
   `self._gliner2_cls().from_pretrained`) are flagged; the two neutralizing loaders and a
   processor/tokenizer load are not. — **MET** (`test_jit_guard_flags_a_bare_and_an_indirected_mdeberta_load`,
   `test_jit_guard_accepts_the_neutralizing_loaders`).
4. The constant cannot drift into a lie: each loader named in `_JIT_NEUTRALIZING_LOADERS` really
   calls `neutralize_torch_jit` in `perf.py`, and each perf-seam loader NOT named there really does
   not. — **MET** (`test_jit_neutralizing_loaders_constant_matches_the_real_perf_seam`).

## Evidence the shipped fix is non-vacuous

Probe reconstructing the pre-fix helper (accept-set `_PERF_SEAM_FUNCS`) and the shipped
`jit_seam_offenders`, both against the audit's sixth-Provider source:

```
PRE-FIX  seam_routed non-empty (assert passes): True
PRE-FIX  offenders: [] -> guard GREEN
POST-FIX offenders: ['line 1: GLiNER2.from_pretrained'] -> guard RED
```

The pre-fix guard is blind on the exact regression it exists to catch; the shipped guard bites.
No further change is warranted — a second fix would only duplicate `49b2e53`.

## Residual (NOT this defect, logged for traceability)

`_MDEBERTA_PROVIDER_SOURCES` is a hardcoded tuple. A genuinely new sixth mdeberta Provider file is
only scanned once someone adds it to that tuple; the guard does not discover mdeberta providers by
itself. That is a separate coverage-reach question from the accept-set defect fixed here, and is
out of scope for this ticket.
