# R4-wiring_dead_code-1 — `redaction.gliner2` is the only mdeberta Provider with zero *behavioural* #190 JIT-ordering coverage

**Severity:** major
**Stream:** feat/release-fleet-integration (tickets #178–#199)
**Touches:** `packages/latence-core/tests/test_perf_jit_import_ordering.py`
**Related:** #190 (audit R1), #190-5 (audit R2 — the identical omission for `disambiguation.glinker`),
#197 (`redaction.gliner2` promoted to the wizard's blessed redactor), ADR-0036 §W9-jit.

## What is wrong

ADR-0036 §W9-jit (`docs/adr/0036-provider-ecosystem-and-bakeoff.md:291`) states that the
"neutralize-before-the-backbone-import" ordering is *proven behaviourally* by
`test_perf_jit_import_ordering.py`, whose fake `gliner`/`gliner2` modules script the mdeberta
relative-position helper **at module import** — "the only shape that can tell 'neutralized before
the import' from 'after'" — over "the five mdeberta Providers and the `disambiguation.glinker`
backend".

`_PROVIDERS` (`:234-244`) holds exactly five params:

| id | load path |
|---|---|
| `entity.gliner` | `GlinerEntityExtractor._load_model` |
| `redaction.gliner_pii` | `GlinerPIIRedactor._load_model` |
| `relation.gliner_relex` | `GlinerRelexExtractor._load_model` |
| `fused_entity_relation.gliner2` | `Gliner2FusedExtractor._load_model` |
| `disambiguation.glinker` | `GLinkerBackend._init_executor` |

The **fifth mdeberta Provider**, `redaction.gliner2` (`Gliner2PIIRedactor`, whose backbone import
lives in `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:295-306`
`_gliner2_cls` → `:308` `_load_model`), is absent — and appears in no behavioural JIT test
anywhere: `grep -rn 'neutralize\|jit' packages/latence-pii-gliner2/tests/*.py` returns nothing, and
`packages/latence-extract-gliner2/tests/test_gliner2_jit_neutralization.py` drives only
`Gliner2FusedExtractor`.

Its sole coverage is the **AST guard** in `test_perf_fp32_regression_guard.py`
(`jit_import_ordering_offenders`), whose check is purely *lexical*: it asserts
`min(lineno of any neutralize_torch_jit call in the function) < min(lineno of any backbone import)`.

This matters more for this Provider than for any of the five that *are* covered: #197 made
`redaction.gliner2` the wizard's blessed default redactor in **both** profiles
(`wizard.py:59 BLESSED_REDACT_PROVIDER`, used at `:607` `lite` and `:830` `enterprise-sota`).

## Failure scenario (ships fully green)

In `_gliner2_cls`, wrap the seam call:

```python
if os.environ.get("LATENCE_JIT_FIX", "1") == "1":
    neutralize_torch_jit(provider_name=self.provider_name)
from gliner2 import GLiNER2
```

`min(neutralize linenos)` is still below `min(import linenos)`, so `jit_import_ordering_offenders`
returns `[]`, `test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone` passes, the
routing guard passes (the load still goes through `load_gliner2_optimized`), and `make test` stays
at 2985 passed.

On any host where `LATENCE_JIT_FIX` is set to anything else, transformers' `deberta_v2`
relative-position helpers are compiled to `ScriptFunction`s at the `gliner2` import, and the first
`redact` forward dies with `libnvrtc-builtins.so.13.0` on a Blackwell/CUDA-13 pod — in the Stage the
wizard now emits **by default in both profiles**. The lexical guard cannot see it; only a fake that
scripts at module import can.

## What to build

Add `redaction.gliner2` to `_PROVIDERS` in `packages/latence-core/tests/test_perf_jit_import_ordering.py`,
reusing the existing `("gliner2", "GLiNER2")` module-import-scripting fake (already in-suite as
`_PROVIDERS[3]`), with a loader that drives `Gliner2PIIRedactor._load_model()` — the real lazy load
path, so the assertion is on the shipped ordering, not on a re-statement of it.

`Gliner2PIIRedactor._load_model` additionally calls `require_batched_entity_api` on the loaded
object, so the shared `_MdebertaModel` fake must expose a callable `batch_extract_entities`
(inert — this suite asserts on import-time JIT state, not on inference).

## Acceptance criteria

1. `_PROVIDERS` contains a `redaction.gliner2` param whose loader calls
   `Gliner2PIIRedactor(...)._load_model()` (the real load path — constructing is not enough).
2. `test_jit_is_neutralized_before_the_gliner_import[redaction.gliner2]` asserts
   `jit_enabled_at_import is False`, that the helper is the plain eager function, and that the
   encoder forward completes — the same three assertions the other five params make.
3. The new param **fails** if `neutralize_torch_jit` does not take effect before
   `from gliner2 import GLiNER2` in `Gliner2PIIRedactor._gliner2_cls`, including for mutations the
   lexical AST guard cannot see (demonstrated with the `LATENCE_JIT_FIX` wrapper above).
4. No production behaviour changes; `make test` stays green at ≥ 2985 passed / 28 skipped.
