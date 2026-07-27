# R1 — `sparse.splade` places the model on the resolved device but never moves the tokenizer encoding

- **Severity:** critical
- **Dimension:** the five defects / a deepening that broke a working path
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-splade/src/latence_splade/model.py`
  (`TransformersSpladeModel._load` → `load_transformers_model`/`place_on_device`, and
  `TransformersSpladeModel.term_weights`/`_forward`)
- **Invariants at stake:** ADR-0036 §2 (device placement is the adapter's job — *all* of it),
  ADR-0034 (a failure surfaces typed — it does here, but the failure itself is the regression),
  "off by default means byte-identical" (a GPU host that ran `sparse.splade` before this branch
  must still run it)
- **Status:** **FIXED** on this branch.

## What was wrong

The branch's defect-3 fix put `sparse.splade` through the shared perf seam:
`load_transformers_model(...)` → `place_on_device(model, "cuda")`, verified read-back. That half is
correct and is a genuine deepening — before it, the model was loaded bare fp32 and left wherever
`from_pretrained` dropped it (CPU).

But placement has **two** halves, and only one was done. `term_weights` builds its inputs with

```python
encoding = self._tokenizer(list(texts), ..., return_tensors="pt")
output = self._forward(encoding)          # self._model(**encoding)
```

A `transformers` tokenizer knows nothing about where the model was placed: `return_tensors="pt"`
always returns a `BatchEncoding` of **CPU** tensors. Nothing moved it. So the forward now mixes a
CUDA module with CPU inputs.

The two sibling adapters that went through the *same* seam in the *same* change do it correctly —
`latence-parser-lighton/src/latence_parser_lighton/provider.py:477-479` and
`latence-parser-glm/src/latence_parser_glm/provider.py:598-600`:

```python
device = self.device.device
if device is not None:
    inputs = inputs.to(device)
```

`sparse.splade` was the one raw-`transformers` adapter that got the model move without the input
move. (The four GLiNER/GLiNER2 Providers are unaffected: they hand the library raw *text* and it
tokenises and places internally.)

## Failure scenario

On any CUDA host:

1. `SpladeCompositionEmbedder` resolves `device: auto` → `"cuda"` (`provider.py:185`) and threads
   the `DeviceDecision` into `TransformersSpladeModel`.
2. `_load` → `load_transformers_model` → `place_on_device(model, "cuda")` — the model is resident
   on cuda, verified.
3. `term_weights` tokenises to CPU tensors and calls `self._model(**encoding)`.
4. Real torch raises `RuntimeError: Expected all tensors to be on the same device …`, which
   `term_weights`' `except Exception` maps to
   `ProviderError: sparse.splade model failure during encode: RuntimeError.`

The Export's sparse-embedding stage then fails outright — **every** batch, not a slow path. Before
this branch the model stayed on CPU and the forward succeeded (slowly). So a Provider that worked
on GPU boxes is broken by the change meant to speed it up, and it fails on exactly the hosts the
change targets.

Why the branch's own tests could not catch it:

- `test_splade_places_the_model_on_the_resolved_device` asserts the model moved, but its
  `_FakeMaskedLM.__call__` ignores the device of what it is called with entirely — it can prove
  placement happened, never that the forward is still *runnable* afterwards.
- The only test that actually calls `term_weights`
  (`test_splade_forward_runs_under_inference_mode_not_no_grad`) constructs the model with **no**
  `DeviceDecision`, so no placement happens and CPU-model/CPU-inputs agree.
- The offline env is CPU-only (`device` resolves to `"cpu"`), so even the end-to-end Provider path
  never mixes devices.

## What was built

**Fix** (`model.py`, `term_weights`) — the same two lines as the sibling transformers adapters,
keyed off the same resolved device the load placed on:

```python
device = self._device.device
if device is not None:
    encoding = encoding.to(device)
```

No new layer, no new seam: the missing half of an operation ADR-0036 §2 already assigns to the
adapter, written the way this repo already writes it. `None` (a caller that resolved no device)
moves nothing, so the offline/no-placement path is byte-identical.

**Test doubles made faithful, not weakened.** Three fakes returned a bare `dict` where the real
dependency returns a `BatchEncoding` (a mapping that *also* carries `.to(device)`) — a double less
capable than the thing it stands in for, which is the same category of fake-bug the branch already
called out for `_FakeMaskedLM`'s missing `.to`/`.device`:

- `packages/latence-core/src/latence_core/conformance/stub.py` — `_StubArray` gains `device`/`.to`,
  new `_StubBatchEncoding`;
- `packages/latence-splade/tests/test_splade_embedder.py` — `_FakeArr` gains `device`/`.to`, new
  `_FakeEncoding`.

Nothing was relaxed; the fakes became stricter about matching the real contract.

## Acceptance criteria

- [x] `term_weights` moves the encoding onto the resolved device before the forward, and only when
      a device was resolved.
- [x] With the model placed on `cuda` and a double that enforces torch's real device contract, the
      forward runs and returns the correct term-weight map (pre-fix: `ProviderError`).
- [x] With no `DeviceDecision`, nothing is moved and the forward still runs (no behaviour change on
      the CPU/offline path).
- [x] The regression test fails against the pre-fix code and passes after.
- [x] No sibling adapter has the same gap (swept: the two OCR parsers already move inputs; the four
      GLiNER/GLiNER2 Providers pass raw text and never build an encoding).
- [x] `make test` green; `scripts/verify-local.sh` green.

## Evidence it failed pre-fix

`test_splade_moves_the_encoding_onto_the_device_it_placed_the_model_on`, run against the code as it
stood before the fix:

```
>           raise ProviderError(msg) from exc
E           latence_core.errors.ProviderError: sparse.splade model failure during encode: RuntimeError.

packages/latence-splade/src/latence_splade/model.py:250: ProviderError
=========================== short test summary info ============================
FAILED packages/latence-splade/tests/test_splade_perf_seam.py::test_splade_moves_the_encoding_onto_the_device_it_placed_the_model_on
```

The `RuntimeError` behind it is raised by `_DeviceStrictMaskedLM`, a double that reproduces torch's
contract verbatim: a forward whose input tensors are not resident where the module is raises
`Expected all tensors to be on the same device`.

## Regression tests

- `packages/latence-splade/tests/test_splade_perf_seam.py::test_splade_moves_the_encoding_onto_the_device_it_placed_the_model_on`
- `packages/latence-splade/tests/test_splade_perf_seam.py::test_splade_encoding_stays_on_cpu_when_no_device_was_resolved`
