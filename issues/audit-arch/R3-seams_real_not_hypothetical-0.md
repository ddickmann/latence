# R3 / seams-real-not-hypothetical #0 — `stack_components()` must walk the DECLARED slot set, not a hard-coded 3-key tuple

**Status: FIXED on `feat/arch-deepening`.**

## Reported defect (severity: major)

- **Dimension:** seams real, not hypothetical / dead new API beside a live stale one
- **Branch:** `feat/arch-deepening`
- **Sites:**
  - `packages/latence-core/src/latence_core/subprovider.py:214` — `declared_slot_keys()`, **zero
    production consumers** (one superset assertion in a test was its only reader).
  - `packages/latence-core/src/latence_core/subprovider.py:34` / `:40-42` — the module docstring
    claims this module is "the only list of nested keys in the codebase" and that the "envspec
    walker" and "GPU placement probe" enumerate slots "from the SAME declaration the runtime
    resolver uses". Neither was true.
  - `packages/latence-core/src/latence_core/envspec.py:1252` — `_NESTED_COMPONENT_KEYS`, the
    surviving hard-coded tuple, consumed at `:1280` by `stack_components()`.
  - `packages/latence-core/src/latence_core/placement.py:320` — the GPU placement probe iterates
    `stack_components(spec.stack_path)`.
- **Invariants at stake:** ADR-0034 (no silent errors — an unexercised GPU model must not read as
  a pass), determinism (identical inputs → identical component order).

C8 introduced `subprovider.declared_slot_keys()` explicitly to replace the bespoke walker key
lists. The replacement never happened for the walker that matters. `envspec._NESTED_COMPONENT_KEYS`
still existed and still listed **3 of the 7** declared slot keys
(`embedder`, `sparse_embedder`, `multivector_embedder`), so `stack_components()` silently dropped
every nested sub-Provider wired in an `ocr`, `fde_converter`, `bm25` or `graph_features` slot.

Two walkers in the *same module* disagreed: the structural `stack_providers()` found those
Providers; the key-list-driven `stack_components()` did not.

## Failure scenario (reproduced against the committed stacks, pre-fix)

```
declared_slot_keys()                     -> {bm25, embedder, fde_converter, graph_features,
                                             multivector_embedder, ocr, sparse_embedder}
envspec._NESTED_COMPONENT_KEYS           -> (embedder, sparse_embedder, multivector_embedder)

stack_components(stacks/universal-parse.yaml)   OMITS parser.lighton     (stack_providers finds it)
stack_components(stacks/retrieval-signals.yaml) OMITS fde.reference      (stack_providers finds it)
stack_components(stacks/graph-features.yaml)    OMITS graph_features.degree
```

`parser.lighton` is the node `subprovider.py:158-160` itself calls "the GPU-gated node of that
stack".

The consequence is a **false green on the #191 gate**. `placement.py:320` iterates
`stack_components()` and `continue`s on anything it does not yield. As soon as a deploy environment
stack wires `parser.render` + a nested `ocr` model (the documented universal-parse shape),
`latence env placement` reports `ok=True` having never driven a forward pass through that GPU
model. The `_nothing_to_measure` / `E-PLACEMENT-NONE` guard does **not** fire, because other
components *were* probed — the report is a genuine green earned by measuring the wrong set.
`tests/test_env_stack_constructible.py::_unconstructible` has the same blind spot: it constructs
every component `stack_components()` yields, so a nested OCR/FDE/BM25/graph-feature Provider whose
committed config the Provider rejects sails through the constructibility gate.

The only test of the new API,
`packages/latence-core/tests/test_stack_check.py:487`, asserted
`{embedder, sparse_embedder, multivector_embedder} <= declared_slot_keys()` — a **superset**
assertion that stays green forever while the walker keeps using the stale tuple. It proved nothing
about the migration it was named for.

## What to build

The seam-making-real fix, not another layer:

1. Delete `envspec._NESTED_COMPONENT_KEYS`. `stack_components()` calls
   `subprovider.declared_slot_keys()` **at walk time** (not a module-level snapshot), so a newly
   declared slot cannot leave the walker behind.
2. Walk the config's own keys in file order and admit those that are declared slots — the
   docstring already promised "in file order", and it keeps the result deterministic without
   depending on `frozenset` iteration order.
3. Replace the vacuous superset test with one that can only pass if the walker really reads the
   declaration.

No new abstraction, no compatibility shim: one hard-coded list is deleted and its single consumer
is pointed at the module that already owns the concept.

**Deliberately out of scope:** `placement.DRIVERS` has no driver for the `ocr` / `fde_converter` /
`bm25` / `graph_features` capabilities. Post-fix, a nested Provider in one of those slots that is
also in the probe set yields `E-PLACEMENT-UNDRIVABLE` — "unexercised is not passed", which is the
designed honest outcome and is exactly the truth the stale tuple was hiding. No committed
`deploy/env/*/stack.yaml` wires one today, so no environment gate changes colour. Adding a Parser
driver (driving OCR needs a real page image, unlike the text-in/vector-out drivers) is separate
work.

## Acceptance criteria

- [x] `envspec._NESTED_COMPONENT_KEYS` no longer exists; `grep` finds exactly one list of nested
      slot keys in the codebase (`subprovider._SLOTS`).
- [x] For every committed stack (`stacks/*.yaml` and `deploy/env/*/stack.yaml`), the two walkers
      agree: `{c.provider for c in stack_components(p)} == stack_providers(p)`.
- [x] `stack_components(stacks/universal-parse.yaml)` yields `parser.lighton` with
      `capability == "ocr"` and its `device: cuda` config.
- [x] Declaring a NEW slot on a Capability makes the walker pick it up with no edit to `envspec`
      (asserted by monkeypatching `subprovider._SLOTS`) — this is what makes the seam real rather
      than coincidentally-in-sync.
- [x] The GPU placement probe no longer returns `ok=True` for a stack whose nested GPU model was
      never driven.
- [x] Order is deterministic and byte-identical run to run.
- [x] `make test` green; no test weakened or deleted (the vacuous superset assertion is replaced
      by a strictly stronger one).

## Regression tests

- `packages/latence-core/tests/test_envspec.py::test_the_two_stack_walkers_agree_on_every_committed_stack`
- `packages/latence-core/tests/test_envspec.py::test_stack_components_finds_the_nested_ocr_sub_provider`
- `packages/latence-core/tests/test_stack_check.py::test_a_newly_declared_slot_is_walked_without_editing_envspec`
- `packages/latence-core/tests/test_placement.py::test_a_nested_gpu_model_cannot_be_skipped_into_a_green_report`
