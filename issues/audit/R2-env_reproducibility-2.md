# R2 · env_reproducibility · [major] The GPU-placement gate (#198) measures resident CUDA memory, not GPU work

**Where:** `packages/latence-core/src/latence_core/placement.py` (`_probe_one`,
`CUDA_ALLOCATION_FLOOR_BYTES`, module docstring) ·
`packages/latence-core/tests/test_placement.py` (`FakeCuda.reset_peak_memory_stats`)

## What is wrong

`_probe_one` measures GPU work as:

```
provider = provider_cls(dict(component.config))   # <-- weights may land on CUDA HERE
allocator.reset_peak_memory_stats()
driver(provider); allocator.synchronize()
peak = allocator.max_memory_allocated()
```

Real `torch.cuda.reset_peak_memory_stats()` does **not** zero the peak — the allocator's
`Stat::reset_peak` sets `peak = current`. So after the reset the peak already equals whatever the
Provider allocated during construction. A Provider that does `model.to("cuda")` in `__init__` and
then runs the entire forward pass on the CPU measures its full weight size and **passes the gate**.

The suite cannot see it because `FakeCuda.reset_peak_memory_stats()` sets `self._peak = 0` — a fake
strictly *stricter* than the real dependency. Every green test therefore proves a property real
torch does not deliver ("a CPU-resident model allocates exactly zero"), which is precisely what the
module docstring and the `CUDA_ALLOCATION_FLOOR_BYTES` comment ("a floor of 1 cannot be tuned into a
pass") assert.

A second instance of the same hole: a Provider that lazily loads its weights onto CUDA **during the
first driven call** and computes on the CPU also allocates gigabytes inside the measured window.

## Failure scenario (proven)

With an allocator that faithfully models torch (`reset` sets `peak = current`) and a Provider that
allocates 1.1 GB in `__init__` (`model.to("cuda")`) and runs `redact()` entirely on the CPU,
`latence env placement` reports:

```
environment 'probe-env': OK (no findings)
  GPU  redaction.gliner_pii (redaction): 1100000000 peak CUDA bytes
```

`ok = True`. On the pod, gate 3 of `deploy/env/gpu-learned/smoke.sh` reports GREEN for exactly the
"correct output, configured cuda, wrong device" class (#191) the gate exists to make unshippable.

## What to build

Measure the CUDA memory the **driven inference itself** allocates, above the Provider's resident
baseline, and make the fake faithful:

1. Add `memory_allocated()` to the `CudaAllocator` Protocol (the seam already mirrors `torch.cuda`).
2. Drive the Provider once as an **unmeasured warm-up** so construction-time *and* lazy-load weight
   placement are outside the measured window.
3. Then `reset_peak_memory_stats()`, read `baseline = memory_allocated()`, drive again, `synchronize()`,
   and report `cuda_bytes = max_memory_allocated() - baseline`.
4. Make `FakeCuda` model torch's real semantics: `reset_peak_memory_stats()` sets `peak = current`,
   with a separate persistent-allocation channel (weights) and a transient one (activations).
5. Correct the module docstring / constant comment to state the measured property exactly.

## Acceptance criteria

- [ ] `CudaAllocator` exposes `memory_allocated()`; `torch.cuda` satisfies it unchanged.
- [ ] `_probe_one` reports the allocation **delta** of a measured drive that follows a warm-up drive.
- [ ] A Provider that allocates on CUDA at construction (or on first call) and computes on the CPU
      yields `E-PLACEMENT-ON-CPU`, under an allocator whose `reset` sets `peak = current`.
- [ ] `FakeCuda.reset_peak_memory_stats()` no longer zeroes the peak; no test relies on the stricter
      fake.
- [ ] Regression test fails on the pre-fix code and passes after.
- [ ] `make test` stays green at or above 2985 passed / 28 skipped; no test weakened or deleted.
