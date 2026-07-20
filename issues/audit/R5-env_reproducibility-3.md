# R5 — the GPU-placement gate's probe set is silenced by an under-declared spec

- **Severity:** major
- **Dimension:** environment reproducibility / anti-false-green
- **Branch:** feat/release-fleet-integration
- **Tickets:** #198 (GPU placement probe), #191 (the silent CPU fallback), #199, ADR-0053
- **Status:** fixed

## What is wrong

`probe_placement()` (`packages/latence-core/src/latence_core/placement.py:292`) derives its entire
probe set from one line:

```python
if component.provider not in spec.learned_providers:
    continue
```

`learned_providers:` is an **optional, hand-written** field in `environment.yaml`. So the guard
that exists precisely so a silent CPU fallback (#191) can never ship again asks the spec's author
whether to guard, and skips silently when the answer is no. Nothing cross-checks that field
against `latence_core.envspec.model_loading_providers(spec, repo_root)` — the *derived* answer to
"which of the Providers this environment installs actually load a model" — even though that
derivation already exists in the same package and is consumed by
`latence_core.compat.required_components`.

This is the **identical mechanism** to R3-env_reproducibility-1 (an optional declaration silencing
a derived guard). That fix was applied to `compat.py`'s matrix coverage and **not** to
`placement.py`, so the same hole stayed open in the stricter of the two gates.

It also directly contradicts `placement.py`'s own docstring:

> A learned component whose capability is absent here is REPORTED, never skipped: an unexercised
> Provider in a GPU environment is exactly the blind spot #191 hid in.

The probe reports the *undrivable* case and skips the *undeclared* case — the strictly more
dangerous one, because an undeclared Provider is invisible rather than merely unmapped.

## Failure scenario

Measured against a throwaway copy of the tree.

**Addition.** Add the ADR-0053 sparse leg to `deploy/env/gpu-learned` — `packages/latence-splade`
in `workspace_packages`, `sparse.splade` in `providers`, and a stack stage with
`capability: sparse`, `provider: sparse.splade`, `config: {device: cuda, ...}`. A routine change,
not spec vandalism. After re-rendering `requirements.in`, `latence env verify` returns **OK** (no
finding: `E-LEARNED-UNKNOWN` and `E-LEARNED-UNDRIVEN` both key off `learned_providers`, which the
new Provider is simply absent from), and `probe_placement` attempts only

```
['fused_entity_relation.gliner2', 'redaction.gliner_pii', 'redaction.gliner2',
 'disambiguation.glinker', 'embedding.sentence_transformers']
```

`sparse.splade` is never loaded, never driven, never measured, and emits **no finding of any
kind** in any of the four smoke gates. The compatibility matrix goes red
(`E-MATRIX-CELL-UNCOVERED`) only until a documentation cell is added — after which every gate is
green while the new CUDA Provider has never been proven to allocate a single CUDA byte.

**Removal.** Deleting *only* the `learned_providers` entry for `disambiguation.glinker` — the one
Provider whose documented graceful degradation is, per `stack.yaml`'s own comment, "precisely the
#191 shape" — leaves `latence env verify` **OK** and drops it from the probe set (probed = False).

**Degenerate.** With `learned_providers: []`, `latence env placement` returns `ok=True` with 0
placements and 0 findings: the gate reports GREEN having measured nothing at all.

## What to build

Make the probe set **structural**, exactly as R3 made matrix coverage structural:

1. `latence_core.envspec.placement_probe_providers(spec, repo_root)` — the union of the declared
   `spec.learned_providers` and the derived `model_loading_providers(spec, repo_root)`. One
   function, so the two gates cannot drift apart on what "a Provider that loads a model" means.
2. `probe_placement()` drives that union. It takes `repo_root` (defaulting to the same three-
   parents-up derivation the CLI uses for `env verify`), keeping the accepted-not-created posture
   of ADR-0016.
3. `latence env placement` grows `--repo-root`, and `deploy/env/gpu-learned/smoke.sh` gate 3
   passes it — symmetric with gate 2's `env verify --repo-root`.
4. An empty probe set inside a reachable-CUDA run is itself a finding (`E-PLACEMENT-NONE`): a
   placement gate that measured nothing must never report green. "It fails; it never skips."

On the committed `gpu-learned` environment the derived set is **exactly** the declared set today,
so this changes no shipped result — it removes the ability to lose one.

Residual, stated honestly: the derivation rests on `_MODEL_RUNTIME_DISTRIBUTIONS`, a curated set
matched against packages' declared dependencies. A Provider package that loads weights through a
loader distribution not in that set is still invisible to it — a visible one-line omission in core
rather than a silently empty probe set, and the declared field is still unioned in so a spec can
always oblige the probe to *more*.

## Acceptance criteria

- [x] `probe_placement()` probes every model-loading Provider the environment installs and its
      stack instantiates, whether or not `learned_providers` names it.
- [x] Removing an entry from `learned_providers` cannot remove a Provider from the probe set.
- [x] An empty probe set yields `E-PLACEMENT-NONE` rather than `ok=True` with zero measurements.
- [x] `latence env placement --repo-root` exists and gate 3 of `gpu-learned/smoke.sh` passes it.
- [x] Non-vacuous regression tests fail on the old code (Provider silently unprobed) and pass on
      the fix, going through the public `probe_placement` entry point.
- [x] The committed `gpu-learned` probe set is unchanged by the fix (derived == declared today).
- [x] `make test` stays green at or above the 2985 passed / 28 skipped baseline.
