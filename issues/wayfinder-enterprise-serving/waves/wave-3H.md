# Wave 3-H — `TorchRuntime` owns process-global torch state; the ST family gets its composite

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 3-H). Load-path only — no
pipeline output can change, and the honest oracle is not the ticket-03 comparator (zero-torch)
but the perf/embedder-st suites plus the F7 configuration repeated. Two commits, plus the F7
root-cause. **Bonus beyond the deepening: F7 stopped being a mystery — it reproduced
deterministically on this host, bisected to its polluters, and is closed at the source.**

## Part 1 — `TorchRuntime` (commit `17822c7`)

Three perf-seam steps mutate state that outlives any Provider:

| step | global state it touches |
|---|---|
| `neutralize_torch_jit` | `PYTORCH_JIT` env var + `torch.jit._state` (in-process) |
| `cap_intra_op_threads` | the torch intra-op thread pool (`set_num_threads`) |
| `maybe_compile` | TorchDynamo compilation caches/guards |

They are now **thin delegators** to the current `TorchRuntime`
(`latence_core.providers.perf.current_torch_runtime()`), which RECORDS the first prior value of
everything it changes:

- `"jit_env"` — prior `PYTORCH_JIT` (`None` = was unset; restore re-unsets it)
- `"jit_enabled"` — prior `torch.jit._state._enabled.enabled` (restore re-enables via
  `state.enable()`, else the proxy flag)
- `"num_threads"` — prior `torch.get_num_threads()` (restore re-applies it)
- `"dynamo"` — `True` when `torch.compile` was invoked (no readable prior; restore is
  `torch._dynamo.reset()`)

`restore()` is guarded (never raises, torch-free-safe, survives a torch monkeypatched away
between mutation and restore) and idempotent (records are consumed). `scoped_torch_runtime()`
pushes a fresh runtime so mutations inside the `with` land in — and are restored by — that
runtime; scopes nest, each restoring only its own. Unscoped callers hit the root runtime, which
records but is never restored — the exact process-lifetime semantics all 7 provider packages
always had. **No call site changed** (verified: `latence-embedder-st`, `latence-extract-gliner2`,
`latence-pii-gliner2`, `latence-splade`, `latence-parser-glm`, `latence-parser-lighton`,
`latence-disambig-glinker` + `AdapterBase.maybe_compile`), and the capability-gate /
fp32-regression AST guards still match the same lexical names.

## Part 2 — `load_sentence_transformer` (same commit)

The fourth composite, alongside the gliner / gliner2 / transformers three.
`embedding.sentence_transformers` was the ONE learned Provider assembling the steps piecemeal
(`load_with_fa2_fallback` + a hand-placed `maybe_compile`, no thread cap, no skip check). Its
`_load_model` is now one call; the composite owns, in order:

1. `_require_device` — a skipped `DeviceDecision` is a loud ConfigError (#191), never a silent
   CPU fallback (unreachable for this `compute="either"` profile, but the contract is uniform);
2. `cap_intra_op_threads(resolve_intra_op_threads(config))` — the embedding Stage now caps like
   every other learned family (process-wide pool, W9-scaling);
3. the deferred backbone import (the Provider's factory; its `ImportError` carries the
   "pip install latence-embedder-st" hint and propagates unwrapped);
4. `load_perf_kwargs` → the guarded native load, `st_cls(model_id, device=…, model_kwargs=…)`,
   with the FA2→sdpa retry (`load_with_fa2_fallback`) — ST's `device=` IS its placement;
5. `inspect_model` — the caller's eager-model hook (the embedder's
   `get_sentence_embedding_dimension` width check), run BEFORE any compile wrap so "inspect
   eager, then compile" is an ordering a caller cannot get wrong;
6. `maybe_compile` (guarded, off by default).

**Deliberately NO `neutralize_torch_jit` in this composite**: the kill is a DeBERTa/nvrtc
defence and the ST defaults (Granite r2 = ModernBERT) carry no `@torch.jit.script` helpers —
pulling the process-wide `PYTORCH_JIT=0` lever from this family would add exactly the unowned
cross-provider mutation this wave removes. Pinned by
`test_st_load_does_not_touch_the_jit_levers`. The old fp32-guard exemption test ("ST stays on
the step-level seam") is REVERSED into `test_sentence_transformers_loads_through_the_st_composite`.

## F7 root-caused and closed (commit `f748ee0`; FINDINGS F7 addendum)

Building the gate reproduced F7's `prims.iota only supports integer dtypes` **deterministically**
(perf suites + embedder-st real-weights in one interpreter on this torch-carrying host) and
bisection landed on a PAIR of `test_perf_jit_capability_gate.py` tests: they
`delitem(sys.modules, "torch")` and then let the seam's guarded `import torch` **re-execute
`torch/__init__.py` against the stale submodules** still in `sys.modules` — a franken re-init
that corrupts prims/decomp registration process-wide — compounded by `PYTORCH_JIT=0` leaking out
of the file (`monkeypatch.delenv` on an unset var registers no undo). Fixes: pin
`sys.modules["torch"] = None` in the two tests; wrap the file's autouse fixture and the
embedder-st real-weights boundary in `scoped_torch_runtime()`.

Calibrated claim: the roster-host polluter was never caught in the act, so this is "same
fingerprint, class now owned", not "the roster flake is proven identical".

## Gates (all run, tails recorded)

- Perf suites (all 8 files) + embedder-st package tests, one process: **196 passed** (22.9 s),
  including the real-weights Granite 311m load through the new composite.
- New `test_perf_torch_runtime.py`: **17 passed** — record/restore proven against a
  lever-faithful fake torch (env re-unset, JIT re-enabled, threads re-applied, dynamo reset;
  idempotence; first-prior-wins; nesting; guarded restore), plus the ST composite choreography.
- **F7 configuration 3×** (embedder-st + extract-gliner2 + pii-gliner2 + splade + the 8 perf
  files, single process, `PYTHONHASHSEED=0`): **388 passed / 1 skipped / 0 failed, exit 0, all
  three runs** — including the previously-failing pair ordering.
- `ruff check packages` clean; `mypy --strict` clean on `latence-core/src` (104 files) +
  `latence-embedder-st/src`.
- Full `packages/latence-core/tests` single-process run: see FINDINGS F9/F10 for the
  summary-tail caveat; verdict read from exit code + marker counts.

## Residuals

- **F10 (new)**: the gliner2/splade suites swallow pytest's final summary line (pre-existing,
  reproduced on untouched suites; F9's family). Gate verdicts there must read exit codes.
- The root runtime never restores (by design); a future Stage-boundary integration
  (`scoped_torch_runtime` around a Stage's load+run) is available but deliberately NOT wired
  into the Runner in this wave — Wave 3-G owns `runner.py`.
