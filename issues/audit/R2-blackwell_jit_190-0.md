# R2 — ADR-0036 (W9-jit) still mandates the DEFEATED loader-internal fix and forbids the real one

**Severity:** major
**Where:** `docs/adr/0036-provider-ecosystem-and-bakeoff.md:291` (W9-jit section);
derived copies at `docs/compatibility-matrix.md:86`,
`deploy/env/compatibility-matrix.yaml:378` (FM-NVRTC-JIT `resolution`), and the operator comment
at `deploy/env/gpu-learned/stack.yaml:65-68`.

## What is wrong

The #190 fix that actually works is **ordering**: `neutralize_torch_jit()` must run **before** the
Provider evaluates `from gliner import GLiNER` / `from gliner2 import GLiNER2` / `from glinker
import ProcessorFactory`. That import transitively pulls in transformers'
`models.deberta_v2.modeling_deberta_v2`, whose relative-position helpers are decorated with
`@torch.jit.script` and are therefore compiled into real `ScriptFunction`s **at module import
time**. Neither seam lever can undo that afterwards: `PYTORCH_JIT` is read by torch only at *torch*
import, and `torch.jit._state.disable()` affects only *future* `script()` calls.

The code carries the working fix — all five mdeberta Providers and `disambiguation.glinker` call
`neutralize_torch_jit()` ahead of their backbone import, and
`test_perf_fp32_regression_guard.py::test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone`
enforces the ordering structurally (audit R1-3 / R1-4).

**The authoritative record still says the opposite.** ADR-0036 W9-jit states the neutralization is
"Called inside `load_gliner2_optimized` and `load_then_cast`, immediately before the native loader
— **not** copied into five Providers", i.e. it mandates the arrangement the field disproved and
explicitly bans the per-Provider pre-import call that is the fix. The compatibility-matrix entry
(YAML + its rendered document) repeats the loader-internal claim as the resolution of a
**verified** production failure (a killed 519-page run). `deploy/env/gpu-learned/stack.yaml` tells
operators the shared perf module handles it "before the load" and that there is "Nothing to
configure here". The **import-ordering requirement appears nowhere in `docs/`.**

## Failure scenario

A maintainer or agent reads ADR-0036 as the authoritative record, sees five Providers each calling
`neutralize_torch_jit()` at the top of their load path, and correctly identifies it as exactly the
"per-Provider folklore" the ADR bans — and deletes the calls, leaving only the seam-internal ones
the ADR mandates. Every routing-shaped assertion still passes; the JIT is neutralized *after* the
backbone import, so the mdeberta relative-position helpers stay scripted and the
Blackwell/CUDA-13 `libnvrtc-builtins.so.13.0` crash re-opens on the first forward of the next GPU
run. An operator then debugging that crash is pointed by the compatibility matrix — the record of
a real, verified 519-page production kill — at a mechanism the field already disproved.

## Acceptance criteria

1. ADR-0036's W9-jit section records the **import-ordering** property as the fix: the
   neutralization must precede the backbone import, per Provider, and the loader-internal call is
   documented as defence-in-depth for direct seam callers only — never as sufficient on its own.
2. The refuted claim ("not copied into five Providers", "applied at the shared load seam so every
   Provider inherits it") is gone from the ADR, from the compatibility-matrix YAML resolution and
   from its rendered document.
3. `deploy/env/gpu-learned/stack.yaml`'s operator comment attributes the fix to the Provider's
   pre-import call, not to the load seam.
4. `docs/compatibility-matrix.md` remains the exact render of `deploy/env/compatibility-matrix.yaml`
   (`test_compat.py::test_the_published_document_is_the_render_of_the_matrix` stays green).
5. A regression test derives the set of source files that neutralize the JIT **before** importing
   the backbone from the CODE (AST), and fails when the docs contradict that property or omit the
   ordering requirement. It fails on the pre-fix docs.
6. `make test` stays green at or above 2985 passed / 28 skipped.
