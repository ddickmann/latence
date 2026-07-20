# R6-env_reproducibility-5 — `probe_placement` reports green having measured nothing when the probe set is non-empty but no stack component belongs to it

**Severity:** major
**Where:** `packages/latence-core/src/latence_core/placement.py` (the `if not probe_set` guard and
the unguarded measurement loop below it)
**Test gap:** `packages/latence-core/tests/test_placement.py::test_an_empty_probe_set_is_a_finding_not_a_green`

## What is wrong

`probe_placement` guards the "green earned by measuring nothing" invariant on the **wrong
condition**. It asks whether the *probe set* is empty (`E-PLACEMENT-NONE`), not whether the gate
actually *measured* anything. Those are two different questions, and the measurement loop below the
guard is the second branch to the same false green:

```python
if not probe_set:                     # guarded
    return ... E-PLACEMENT-NONE ...
for component in stack_components(spec.stack_path):
    if component.provider not in probe_set:
        continue                      # UNGUARDED: every component may take this branch
    ...
return PlacementReport(EnvironmentReport(spec.name, ()), ())   # ok=True, measured nothing
```

Whenever the probe set is non-empty but **no** stack component belongs to it, the loop body never
runs, no finding is appended, and the report is `ok=True` with `placements=()` — exactly the state
the module docstring forbids ("the probe set is empty, so the gate would report green having
measured nothing. Also a failure, for the same reason"). The regression test asserts the invariant
only for the empty-probe-set branch, so it passes while the identical false green ships through the
other one.

## Failure scenario (reproduced)

Against a throwaway copy of the committed `deploy/env/gpu-learned/`: set `learned_providers: []`
and remove the four learned Stages (`extract`, `redact_pii`, `redact_financial`, `disambiguate`)
plus the nested `embedder:` block from `stack.yaml`. The five model-loading Providers are still
installed and still declared under `providers:`, so the DERIVED probe set is non-empty (5 entries)
and `E-PLACEMENT-NONE` does not fire — but no stack component matches, so nothing is measured.

Minimal reproduction on the injected fakes (probe set `{sparse.splade}`, stack drives only
`source.fs`):

```
probe_set        = ['sparse.splade']
stack components = ['source.fs']
ok         = True
placements = ()
findings   = []
```

On a real Blackwell pod, gate 3 of `deploy/env/gpu-learned/smoke.sh` (`latence env placement`)
therefore exits 0 and prints only `environment 'gpu-learned': OK (no findings)` with not a single
`GPU <provider> ... peak CUDA bytes` line — a passing #191 guard that proved nothing.

## What to build

Guard the **invariant**, not one of its causes. Delete the pre-loop `if not probe_set` early return
and replace it with a single post-loop check on what the gate actually measured: if no stack
component was probed, that is `E-PLACEMENT-NONE`, whatever the reason. One guard cannot have an
unguarded twin. The finding message names the cause (empty probe set vs. a non-empty probe set no
stack component instantiates) so an operator can act on it.

## Acceptance criteria

- [ ] A non-empty probe set that matches **no** stack component yields `ok=False` with a single
      `E-PLACEMENT-NONE` finding and `placements=()`, instead of `ok=True`.
- [ ] The finding message distinguishes the two causes and names the unmatched probe-set providers.
- [ ] The empty-probe-set behaviour is unchanged: still `E-PLACEMENT-NONE`, still `ok=False`
      (`test_an_empty_probe_set_is_a_finding_not_a_green` keeps passing untouched).
- [ ] A component that IS probed and fails (`E-PLACEMENT-ON-CPU`, `E-PLACEMENT-UNDRIVABLE`,
      `E-PLACEMENT-DEVICE-NOT-CUDA`, `E-PLACEMENT-FAILED`) still reports its own finding and does
      NOT additionally trip `E-PLACEMENT-NONE` — a measured failure is not "measured nothing".
- [ ] `E-PLACEMENT-NO-CUDA` still short-circuits before any of this.
- [ ] A regression test that fails on the old code and passes on the fix.
- [ ] `make test` stays green at or above 3320 passed / 29 skipped; `deploy/env/README.md`'s finding
      table matches the code.
