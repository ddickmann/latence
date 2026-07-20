# R1 / #190-4 — the JIT regression fakes mislocate *when* scripting happens, and the blind spot hides a live broken Provider

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199)
**Relates to:** #190 (Blackwell/CUDA-13 TorchScript nvrtc crash), R1-blackwell_jit_190-1..3

## What is wrong

The #190 regression suites model the mdeberta encoder with a fake that **scripts its
relative-position helper inside `from_pretrained`** — i.e. *after* the perf seam has run:

* `packages/latence-extract-gliner2/tests/test_gliner2_jit_neutralization.py:145-152`
  (`_MdebertaGLiNER2.from_pretrained`), whose docstring asserts this is "the moment transformers
  evaluates the `@torch.jit.script` decorator on the DeBERTa-v2 modeling module".
* `packages/latence-core/tests/test_perf_seam.py:620-631` (`_mdeberta_loader`), same shape.

That claim is **false**. Real transformers evaluates those `@torch.jit.script` decorators when
`transformers.models.deberta_v2.modeling_deberta_v2` is **imported**, which every real Provider
triggers via `from gliner import GLiNER` / `from gliner2 import GLiNER2` — one statement *before*
the seam is entered. By deferring the scripting until after the seam, the fake accepts an ordering
the real dependency rejects: the tests are green on code that leaves the encoder fully scripted.

The AST guard at `packages/latence-core/tests/test_perf_fp32_regression_guard.py:245-290`
(`test_mdeberta_providers_load_through_a_jit_neutralizing_seam`) has the **same blind spot**: it
only asserts that `X.from_pretrained` is *passed to* `load_then_cast` / `load_gliner2_optimized`.
It says nothing about whether the JIT was neutralized before the `gliner` import, so a Provider
that imports gliner while the JIT is live still passes.

## The live bug the blind spot is hiding

`packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py` (`redaction.gliner_pii2`) —
the only mdeberta Provider that **never calls `neutralize_torch_jit` at all**. Its `_gliner2_cls()`
runs `import gliner2` / `from gliner2 import GLiNER2` with the JIT live, so the deberta_v2 helpers
are compiled into real `ScriptFunction`s at that instant. `load_gliner2_optimized` then neutralizes
the JIT *afterwards*, which cannot un-script them (`PYTORCH_JIT` is read by torch only at *torch*
import; `jit._state.disable()` affects only *future* `script()` calls). On a Blackwell/CUDA-13 host
this Provider still dies with `failed to open libnvrtc-builtins.so.13.0` on the first forward —
the exact crash #190 exists to fix — while the whole #190 suite reports green.

(The sibling Providers were corrected in R1-blackwell_jit_190-3 / commit a22ce7c; `gliner_pii2`
was missed precisely because no test could see the ordering.)

## Failure scenario

1. Deploy `redaction.gliner_pii2` on a Blackwell / CUDA-13 pod.
2. `_gliner2_cls()` imports `gliner2` → transformers deberta_v2 → six `@torch.jit.script`
   decorators evaluated with the JIT live → real ScriptFunctions.
3. `load_gliner2_optimized` sets `PYTORCH_JIT=0` and calls `jit._state.disable()` — too late.
4. First `extract_entities` forward: `RuntimeError: ... nvrtc: failed to open
   libnvrtc-builtins.so.13.0`. PII redaction crashes mid-run.
5. Every #190 test stays green throughout.

## Acceptance criteria

* **AC1** — The AST guard asserts the *ordering* property, not merely the routing one: in every
  mdeberta Provider source, the function that imports `gliner`/`gliner2` must call
  `neutralize_torch_jit(...)` at an **earlier line** in that same function. Meta-tests prove the
  guard flags a "seam-routed but imports-first" source (no false negative) and accepts a correctly
  ordered one (no false positive).
* **AC2** — The new guard **fails on the pre-fix tree**, naming
  `latence-pii-gliner2/.../provider.py`.
* **AC3** — `redaction.gliner_pii2` neutralizes the TorchScript JIT before it imports `gliner2`,
  matching its five sibling Providers.
* **AC4** — The fakes stop asserting the false claim: the docstrings at
  `test_gliner2_jit_neutralization.py` and `test_perf_seam.py::_mdeberta_loader` no longer state
  that `from_pretrained`-time scripting is the transformers shape, and point at
  `latence-core/tests/test_perf_jit_import_ordering.py` as the suite that owns the real
  import-time shape.
* **AC5** — No behaviour change when the JIT is already off: output stays byte-identical
  (throughput-only degradation, ADR-0036). `make test` stays at or above 2985 passed / 28 skipped.
