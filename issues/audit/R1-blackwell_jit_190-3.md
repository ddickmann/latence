# R1 — `neutralize_torch_jit()` runs TOO LATE on every real gliner-family code path (#190)

**Severity:** critical
**Stream:** feat/release-fleet-integration (tickets #178–#199)
**Relates to:** #190 (Blackwell / CUDA-13 `libnvrtc-builtins.so.13.0` crash), ADR-0036 (perf seam)

## What is wrong

`latence_core.providers.perf.neutralize_torch_jit()` is called from *inside* the two shared
loaders — `load_then_cast` (perf.py:270) and `load_gliner2_optimized` (perf.py:530). But in every
real Provider the gliner import happens **before** the loader is entered:

| Provider | gliner import | seam call |
| --- | --- | --- |
| `latence_ner_gliner.provider` | `from gliner import GLiNER` (l.222) | `load_then_cast(...)` (l.241) |
| `latence_pii_gliner.provider` | `from gliner import GLiNER` (l.235) | `load_then_cast(...)` (l.253) |
| `latence_relation_gliner.provider` | `from gliner import GLiNER` (l.225) | `load_then_cast(...)` (l.243) |
| `latence_extract_gliner2.provider` | `self._gliner2_cls()` → `from gliner2 import GLiNER2` (l.299) | `load_gliner2_optimized(...)` (l.323, l.358) |

`from gliner import GLiNER` transitively imports `transformers.models.deberta_v2.modeling_deberta_v2`,
whose six relative-position helpers (`make_log_bucket_position`, `c2p_dynamic_expand`,
`p2c_dynamic_expand`, `pos_dynamic_expand`, `scaled_size_sqrt`, `build_rpos`) are decorated with
`@torch.jit.script` and are therefore compiled into real `ScriptFunction`s **at that import**,
while the JIT is still live.

Neither lever the seam drives can undo that afterwards:

* `os.environ["PYTORCH_JIT"] = "0"` is read by torch **only at torch import** — already past.
* `torch.jit._state.disable()` affects only **future** `script()` calls — the decorators already ran.

So the seam logs `torch JIT neutralized (PYTORCH_JIT=0, runtime state disabled)` and returns `True`
while the encoder is still scripted. The operator is told the workaround is applied when it is not,
and #190's crash is NOT fixed.

## Failure scenario (verified against real torch 2.9.0 + transformers 5.1.0 + gliner)

1. Fresh process, `PYTORCH_JIT` unset. `from gliner import GLiNER` →
   `transformers.models.deberta_v2.modeling_deberta_v2` is in `sys.modules` and all six helpers are
   `torch.jit.ScriptFunction`.
2. Then call `neutralize_torch_jit()` (exactly what `load_then_cast` does next) → returns `True`,
   `torch.jit._state._enabled.enabled is False`, `os.environ["PYTORCH_JIT"] == "0"` — yet all six
   names are STILL `ScriptFunction` and still execute through the TorchScript interpreter.
3. Control: the same import with `PYTORCH_JIT=0` set *before* the process starts yields plain
   `function` objects. The env lever is the only working lever, and it is set too late.

Those helpers are called on every forward (inside `disentangled_attention_bias`), so on a
Blackwell / CUDA-13 pod `GlinerEntityExtractor` / `GlinerPIIRedactor` / `GlinerRelexExtractor` /
`Gliner2FusedExtractor` still die on the first forward with
`nvrtc: failed to open libnvrtc-builtins.so.13.0`.

## What to build

Move the neutralization **before** the gliner import in each of the four Providers' `_load_model`
(`_gliner2_cls` for gliner2) — the earliest point in the Provider's own control that is not a
discovery-time global side effect (importing torch at Provider *enumeration* would violate
ADR-0016's torch-free discovery). The existing seam calls inside the loaders stay (idempotent), so
any other caller of `load_then_cast` / `load_gliner2_optimized` keeps its guard.

`latence_disambig_glinker.backends._init_executor` already has the correct ordering (it calls the
seam before `import torch` / `from glinker import ...`); that is the pattern to match.

## Acceptance criteria

1. In all four gliner-family Providers, `neutralize_torch_jit()` is executed **before** the
   `from gliner …` / `from gliner2 …` statement.
2. A regression test proves the ordering **behaviourally**, not structurally: with a `gliner` /
   `gliner2` module whose *import* scripts the mdeberta relative-position helper (exactly what
   transformers' deberta_v2 module does) and a fake torch whose `ScriptFunction` raises the real
   `libnvrtc-builtins.so.13.0` message when called, loading the model must leave the helper
   **un-scripted** — i.e. the JIT must already be off at the moment the module is imported.
3. The test fails on the pre-fix code (the helper is a `ScriptFunction`; calling it raises the nvrtc
   error) and passes after.
4. No behaviour change when the JIT is irrelevant: throughput-only, byte-identical output
   (ADR-0036 compile-guard discipline). Off-by-default outputs unchanged.
5. `make test` stays green at or above 2985 passed, 28 skipped.
