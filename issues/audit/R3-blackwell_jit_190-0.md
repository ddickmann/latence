# R3 — #190 import-ordering guard is blind to MODULE-LEVEL backbone imports

**Severity:** major
**Area:** `packages/latence-core/tests/test_perf_fp32_regression_guard.py` (#190 JIT guards),
`docs/adr/0036-provider-ecosystem-and-bakeoff.md` §W9-jit
**Stream:** `feat/release-fleet-integration` (tickets #178–#199)

## What is wrong

`jit_import_ordering_offenders(tree)` iterates

```python
for func in ast.walk(tree):
    if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
        continue
```

so it only ever examines statements that live **inside a function body**. A DeBERTa-backed Provider
that imports its backbone at **module level** — `from gliner2 import GLiNER2` at the top of
`provider.py` — lives in no `FunctionDef` and is therefore never examined at all. It produces zero
offenders and passes every #190 guard clean.

That is the *worst* shape, not a benign one: the module-level import is evaluated at
**provider-module / entry-point import**, i.e. before any Provider code — and therefore before any
`neutralize_torch_jit()` call inside a loading function — can possibly run. It is the one ordering
the seam can never repair: verified against real torch 2.12.1 + transformers 5.12.1, the six
`deberta_v2` relative-position helpers are already `ScriptFunction`s by then, and neither
`os.environ["PYTORCH_JIT"] = "0"` (read by torch at *torch* import) nor `torch.jit._state.disable()`
(affects only *future* `script()` calls) can un-script them.

Meanwhile the guard's own docstring, the docstrings of
`test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone` /
`test_every_in_tree_deberta_backed_source_is_scanned_by_the_jit_ordering_guard`, and
`docs/adr/0036-provider-ecosystem-and-bakeoff.md:291` all claim such a source is "caught the moment
its source lands". For the module-level shape that claim was false.

The anti-false-green meta-test `test_deberta_backed_discovery_picks_up_a_newly_landed_source` writes
its fake provider with a **function-level** import, so it never probed this shape either.

## Failure scenario (probe, executed against the real guard module)

A newly landed provider source:

```python
from gliner2 import GLiNER2            # module level — runs at entry-point discovery

class NewProvider:
    def _load_model(self):
        neutralize_torch_jit(provider_name="x.new")
        return load_gliner2_optimized(GLiNER2.from_pretrained, mid, dev, cfg)
```

Against the pre-fix guard module:

* `_imports_a_deberta_backbone_root` → `True` (it IS discovered and IS parametrized into the guards)
* `jit_import_ordering_offenders` → `[]`
* `jit_seam_offenders` → `[]`
* a load IS routed through a JIT-neutralizing loader → `True`
* no hand-rolled `PYTORCH_JIT` string

Every #190 test is green. On a Blackwell / CUDA-13 pod that provider dies on its first forward with
`nvrtc: failed to open libnvrtc-builtins.so.13.0`.

## What to build

1. Extend `jit_import_ordering_offenders` to scan **module scope** as well as function bodies. A
   backbone import evaluated at module import is an offence unconditionally — there is no
   "neutralize first" repair for it; the import must be **deferred into the loading function** that
   calls `neutralize_torch_jit()` first (which is also the ADR-0016 lazy-heavy-dep discipline every
   in-tree Provider already follows).
2. Exempt `if TYPE_CHECKING:` bodies — they are never evaluated at runtime, and every in-tree
   Provider needs them for annotations. The `else:` branch of such an `if` IS evaluated and is
   therefore still scanned.
3. Correct the over-claiming docstrings and `docs/adr/0036-…:291` so the documented property matches
   the code.

## Acceptance criteria

* AC1 — a source with a module-level backbone import is reported by
  `jit_import_ordering_offenders`, **even when** its loading function neutralizes the JIT and routes
  the load through `load_gliner2_optimized` (i.e. exactly the probe above, which the routing guard
  still reports clean).
* AC2 — the offender message names module scope and says the import must be deferred, so the fix is
  obvious from the failure.
* AC3 — no false positive: a `if TYPE_CHECKING: from gliner import GLiNER` block is NOT flagged, and
  all six in-tree DeBERTa-backed sources stay green.
* AC4 — a regression test that FAILS on the pre-fix guard and PASSES after, plus a TYPE_CHECKING
  no-false-positive test.
* AC5 — `make test` stays green at ≥ 2985 passed / 28 skipped; no test weakened or deleted.
