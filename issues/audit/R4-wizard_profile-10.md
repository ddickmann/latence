# R4-wizard_profile-10 — [critical] `redaction.gliner2` silently turns `torch.compile` ON in the blessed GPU config

**Stream:** feat/release-fleet-integration (#178–#199) · **Ticket of record:** #197
**Surfaces:**
- `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py` (no compile default injected)
- `packages/latence-core/src/latence_core/setup/wizard.py` (emitted redact stage config)
- sibling that gets it right: `packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py` (`_COMPILE_DEFAULT = False`)
- `docs/adr/0041-…md` §7a, `docs/TUTORIAL-fresh-pod-walkthrough.md` §1.3

## What is wrong

`Gliner2PIIRedactor` resolves its `compile` knob through the generic
`resolve_gliner2_knob("compile", device, config)`, whose default is `auto` →
**True on cuda**. The wizard emits the blessed redact stage with
`{device: cuda, pii_labels: […], max_len: 768, threshold: 0.5}` — **no `compile` key** — so on the
documented Blackwell pod the redact stage loads `GLiNER2.from_pretrained(…, compile=True)`.

That is exactly the state the twin extractor Provider deliberately forbids: gliner2 compiles with
`dynamic=True`, which **recompile-thrashes on varied-length production chunks** (pod-measured: the
worker pinned at ~290% CPU, GPU 0%, no progress). The extractor injects
`_COMPILE_DEFAULT = False` into its load config at construction for precisely this reason; the
redactor — driving `batch_extract_entities` over the *same* varied-length chunk stream — does not.

Three aggravating facts:

1. **This stream flipped the behaviour.** The pre-#197 redact Provider (`redaction.gliner_pii`)
   went through `maybe_compile`, i.e. `config.get("compile", False)` — off by default. #197 swapped
   the blessed provider to `redaction.gliner2` and silently inverted the default on GPU.
2. **The crash latch does not cover it.** `_reload_uncompiled` only fires on a
   `BackendCompilerFailed`; the thrash is not a failure, it is silent slowness, so the latch never
   fires and the run simply never finishes.
3. **The docs assert the opposite of the code.** `docs/TUTORIAL-fresh-pod-walkthrough.md` §1.3:
   "The provider's `torch.compile` knob is **off by default** (opt-in) … Leave `compile: false`
   (the default)". ADR-0041 §7a lists five reasons for the provider switch and never considers the
   compile default.

No test in the stream asserts anything about the emitted/resolved `compile` knob, so the green gate
cannot see it.

## Failure scenario

Operator runs `latence setup --profile enterprise-sota --pii` on the documented Blackwell GPU pod,
then `latence process`. Verified on the project venv:

```
LATENCE_CUDA=1 → Gliner2PIIRedactor({'pii_labels': [...], 'device': 'cuda',
                                     'max_len': 768, 'threshold': 0.5})
  device   = cuda
  compile  = True   ← blessed config, no compile key
  quantize = True
```

`load_gliner2_optimized` therefore calls `from_pretrained(..., compile=True)`; the redact Stage
burns CPU with the GPU idle and makes no progress, and `_reload_uncompiled` never fires. The same
happens on the `lite` profile whenever the operator answers `cuda`.

## What to build

Mirror the sibling extractor: inject the off-by-default `compile` into the Provider's load config at
construction, so it is the single source of truth for both the eager knob validation and the lazy
load. An operator who sets `compile` explicitly still gets it (the knob stays opt-in, not removed).

## Acceptance criteria

- **AC1** With no `compile` key in config, `Gliner2PIIRedactor` loads with `compile=False` **even on
  cuda**, while `quantize` still resolves True on cuda (the fp16 win is kept).
- **AC2** An explicit `compile: true` still enables compile on cuda (opt-in preserved).
- **AC3** The wizard's blessed `enterprise-sota --pii` redact stage config, fed to the Provider,
  resolves compile OFF — asserted against the real emitted config, not a hand-written dict.
- **AC4** CPU behaviour is byte-identical to before (plain load, no `quantize`/`compile` kwargs).
- **AC5** The inference-time `BackendCompilerFailed` latch and its tests remain green.
- **AC6** `make test` stays at or above 2985 passed / 28 skipped; no test weakened or deleted.
