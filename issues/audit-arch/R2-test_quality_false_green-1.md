# R2 — the C4 choreography suite asserts `threads=8` against a core-clamped value, so it goes red on the CI runner

- **Severity:** major
- **Dimension:** test quality / false green — a suite that only passes on the machine that wrote it
- **Branch:** feat/arch-deepening
- **Site:** `packages/latence-core/tests/test_perf_load_choreography.py`
  - `test_gliner_load_runs_the_whole_sequence_in_order` (the `log.index("threads=8")` step, :208)
  - `test_gliner2_caps_intra_op_threads_before_importing_the_backbone` (:415-416)
  - Behaviour under test: `latence_core.providers.perf.resolve_intra_op_threads`
    (`packages/latence-core/src/latence_core/providers/perf.py:736-756`)
- **Invariants at stake:** DETERMINISM (identical inputs → identical outputs; a test whose verdict
  depends on the host's core count is not deterministic), and the merge gate itself —
  `.github/workflows/ci.yml` runs the full `uv run pytest` on `runs-on: ubuntu-latest`.
- **Status:** **FIXED** on this branch.

## What was wrong

This is the **same defect** `R1-the_five_defects-10.md` found and commit `81a4e73` fixed in the two
sibling files (`test_gliner2_device_seam.py`, `test_gliner2_pii_thread_cap.py`, which now pin
`os.cpu_count`). This file — the new C4 choreography suite, added by the same branch — was missed.

`resolve_intra_op_threads` deliberately clamps the requested cap to the core count:

```python
cores = os.cpu_count() or raw
return min(int(raw), int(cores))
```

Two tests assert the *requested* default (8) as a bare literal against the recorded effect:

```python
log.index("threads=8"),          # inside the ordering list
assert "threads=8" in log
assert log.index("threads=8") < log.index("import:gliner2:jit_live=False")
```

Those are only true on a host with at least 8 cores. Green on the 10-core dev box the branch was
written on; red on the 2–4 vCPU `ubuntu-latest` runner CI runs the whole suite on. So a correct tree
looks green locally and hands the merge gate a red suite — and the failure reads like a Provider
choreography regression (`'threads=8' is not in list`, raised out of the *ordering* assertion)
rather than what it is.

Neither test is about the machine's core count. The first is about the *order* the seven load steps
run in; the second is about the cap landing *before* the backbone import. The clamp is noise in
both.

## Failure scenario

1. Simulate the CI runner by pinning `os.cpu_count() -> 4` (a pytest plugin that patches
   `os.cpu_count` at import).
2. `PYTHONHASHSEED=0 uv run pytest -p fewcores packages/latence-core/tests/test_perf_load_choreography.py`
3. `resolve_intra_op_threads` returns `min(8, 4) == 4`; the fake torch records `threads=4`; the
   membership assertion fails and, in the ordering test, `log.index("threads=8")` raises
   `ValueError` before the ordering is ever checked.
4. On real CI this is the whole `uv run pytest` job going red on a correct tree.

## What was built

The tests were made **host-independent**, not loosened, mirroring the R1 fix so the two halves of
the codebase now treat this the same way:

- A `cores` fixture pins `os.cpu_count`. It is pinned **high** (`_MANY_CORES = 64`) in the three
  tests that observe a `threads=N` event, so the clamp is inert and each assertion is about the cap
  value and the sequence position — a literal `8` (resp. `3`) that means what it says on every
  host. Pinning high is also the *honest* setup for these two: the many-core host is precisely the
  box the oversubscription guard exists for.
- The clamp gets its **own** test, `test_the_intra_op_cap_never_exceeds_the_host_core_count`,
  pinning the count **low** (4 — the `ubuntu-latest` shape). Asking torch for 8 intra-op threads on
  a 4-vCPU box is itself the oversubscription this guard prevents, so the clamp is behaviour worth
  pinning through the seam rather than a detail to work around. It asserts both `threads=4` present
  and `threads=8` absent.
- `test_intra_op_thread_cap_is_configurable_and_can_be_opted_out` (`threads=3`) was pinned high too:
  it happens to survive at 4 vCPU, but it is host-dependent for the same reason and would go red on
  a 2-vCPU runner.

Deliberately **not** done: restating `min(8, os.cpu_count() or 8)` back at the implementation. That
is a tautology — it would keep passing if the cap constant or the clamp direction were wrong.

Net: no production code changed, the seam under test is unchanged, no deepening was undone, and
coverage got *larger* (both halves of `resolve_intra_op_threads`' documented contract are now
pinned through the choreography entry point, where only one was before).

## Acceptance criteria

- [x] Every `threads=N` assertion in the file is independent of the host's core count (verified
      green at a simulated 4 cores and on the real 10-core box).
- [x] The clamp half of the contract has its own test at this seam, and that test is **non-vacuous**:
      it fails if the clamp is removed from `perf.py`.
- [x] No test was weakened or deleted; no production code changed; no assertion restates the
      implementation formula.
- [x] `make test` green; `scripts/verify-local.sh` green.

## Evidence it failed pre-fix

Simulated `ubuntu-latest` (`os.cpu_count() -> 4`), against the file as the branch had it:

```
        order = [
            log.index("jit-off"),
            log.index("threads=8"),
E       ValueError: 'threads=8' is not in list
packages/latence-core/tests/test_perf_load_choreography.py:208: ValueError

>       assert "threads=8" in log
E       AssertionError: assert 'threads=8' in ['jit-off', 'threads=4',
E         'import:gliner2:jit_live=False', 'jit-off', 'load:acme/gliner2:[]']
packages/latence-core/tests/test_perf_load_choreography.py:415: AssertionError

=========================== short test summary info ============================
FAILED .../test_perf_load_choreography.py::test_gliner_load_runs_the_whole_sequence_in_order
FAILED .../test_perf_load_choreography.py::test_gliner2_caps_intra_op_threads_before_importing_the_backbone
```

Mutation proof for the new clamp test — with `return min(int(raw), int(cores))` replaced by
`return int(raw)` in `perf.py`, on the real 10-core box:

```
>       assert "threads=4" in log
E       AssertionError: assert 'threads=4' in ['jit-off', 'threads=8',
E         'import:gliner2:jit_live=False', 'jit-off', 'load:acme/gliner2:[]']
FAILED .../test_perf_load_choreography.py::test_the_intra_op_cap_never_exceeds_the_host_core_count
```
