# R5 — `redaction.gliner2` never places its model on cuda (the placement seam is absent)

**Severity:** major
**Area:** wiring / dead-code (a seam that exists but is not called on one of its two required paths)
**Sites:**
- `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py` — `Gliner2PIIRedactor._load_model` (~:347) and `_reload_uncompiled` (~:379)
- twin (correct): `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:350,382`
- seam: `packages/latence-core/src/latence_core/providers/perf.py:675` — `ensure_gliner2_on_cuda`

## What is wrong

`ensure_gliner2_on_cuda` is the W9-scaling defence against the #191 fp16-eager GPU-idle hang: a
`quantize=True` (fp16) gliner2 model that silently stays on CPU is, per the seam's own docstring,
"pathologically slow and CPU-bound". Its twin Provider `fused_entity_relation.gliner2` calls it at
**both** load sites (initial load and the uncompiled reload).

`Gliner2PIIRedactor` calls it at **neither**. `load_gliner2_optimized` uses its `device` argument
only to resolve the `quantize`/`compile` knobs (`resolve_gliner2_knob`) — it performs **no
placement**. So with `device: cuda` the PII model is fp16-converted and left on CPU.

`#197` made `redaction.gliner2` the wizard's blessed default redactor in **both** profiles, so this
is the default path, not an exotic one.

## Failure scenario

Run `latence setup --profile enterprise-sota --pii` (or `lite` on a GPU box) and process a corpus.
The emitted redact stage is `redaction.gliner2` with `device: cuda`. A probe that injects a fake
`gliner2` module recording `.to()` calls, under `LATENCE_CUDA=1`, prints:

```
resolved device: cuda
from_pretrained kwargs: {'quantize': True, 'compile': False}
model.to(...) calls: []
```

Zero placement calls. The identical probe against the twin `fused_entity_relation.gliner2` prints
`['cuda']`. Result: an fp16 model running eager on CPU — a silent multi-hour redact Stage with the
GPU at 0%, correct output, and no warning.

No test in `packages/latence-pii-gliner2/tests/` asserted placement, unlike the twin's
`test_gliner2_fused_provider.py:793`.

## What to build

1. Import `cuda_available` + `ensure_gliner2_on_cuda` in the PII Provider and call the seam after
   **both** `load_gliner2_optimized` calls (`_load_model` and `_reload_uncompiled`), with
   `provider_name="redaction.gliner2"` — mirroring the twin exactly.
2. The seam is already best-effort/guarded and never raises, so no error-handling change is needed
   (ADR-0034 preserved: the CPU-fallback condition is a WARNING, never a swallowed exception).

## Acceptance criteria

- [x] `Gliner2PIIRedactor._load_model` calls `ensure_gliner2_on_cuda(self._model, cuda=cuda_available(), provider_name="redaction.gliner2")`.
- [x] `Gliner2PIIRedactor._reload_uncompiled` does the same after its uncompiled reload.
- [x] Regression test: with a GPU present (`LATENCE_CUDA=1`) a redact run records exactly one
      `.to("cuda")` on the model and the model reports `device == "cuda"`. FAILS on the old code
      (`to_calls == []`).
- [x] Regression test: on a CPU-only host (`LATENCE_CUDA=0`) the model is **not** moved
      (`to_calls == []`) — placement is a cuda-only concern, off-by-default is byte-identical.
- [x] Regression test: the compile-fault uncompiled **reload** is also placed on cuda.
- [x] Redaction output is byte-identical: no change to spans, masking, or ordering. Fast suite stays
      green at or above 2985 passed / 28 skipped.
