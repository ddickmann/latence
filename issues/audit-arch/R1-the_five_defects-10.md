# R1 — the two new intra-op-thread-cap tests assert a literal against a core-clamped value, so they go red on the CI runner

- **Severity:** major
- **Dimension:** the five defects / a test that only passes on the machine that wrote it
- **Branch:** feat/arch-deepening
- **Site:**
  - `packages/latence-pii-gliner2/tests/test_gliner2_pii_thread_cap.py`
    (`test_redaction_gliner2_caps_intra_op_threads_at_load`, `test_the_cap_is_configurable`)
  - `packages/latence-extract-gliner2/tests/test_gliner2_device_seam.py`
    (`test_intra_op_threads_are_capped_at_load`)
  - Behaviour under test: `latence_core.providers.perf.resolve_intra_op_threads`
    (`packages/latence-core/src/latence_core/providers/perf.py:736-755`)
- **Invariants at stake:** DETERMINISM (identical inputs → identical outputs; a test whose verdict
  depends on the host's core count is not deterministic), and the merge gate itself —
  `.github/workflows/ci.yml` runs the full `uv run pytest` on `runs-on: ubuntu-latest`.
- **Status:** **FIXED** on this branch.

## What was wrong

`resolve_intra_op_threads` deliberately clamps the requested cap to the core count:

```python
cores = os.cpu_count() or raw
return min(int(raw), int(cores))
```

Both intra-op-cap regression tests added by this branch — the defect-2 proof on the redaction side
and its extract-side twin — assert the *requested* value as a bare literal:

```python
assert recording_torch == [8]   # pii, default cap
assert calls == [5]             # extract, intra_op_threads=5
```

Those assertions are only true on a host with at least 8 (resp. 5) cores. They are green on the
10-core dev box the branch was written on and red on any smaller host — including the standard
GitHub `ubuntu-latest` runner (2–4 vCPU) that CI actually executes the whole suite on. So a tree
that is green locally hands the merge gate a red suite, and the failure looks like a Provider
regression rather than what it is.

The repo already knew this. The pre-existing test for the identical behaviour on `main`
(`packages/latence-extract-gliner2/tests/test_gliner2_fused_provider.py:1010`) asserts
`min(8, os.cpu_count() or 8)` and passed the same 4-core simulation the two new ones failed. The
new tests dropped a guard the codebase had already learned.

## Failure scenario

1. Simulate the CI runner by pinning `os.cpu_count() -> 4` (a pytest plugin that patches
   `os.cpu_count` at import).
2. Run the suite: `PYTHONHASHSEED=0 uv run pytest -p fewcores <the two files>`.
3. `resolve_intra_op_threads` returns `min(8, 4) == 4` and `min(5, 4) == 4`; the fake torch records
   `[4]` in both cases; both assertions fail.
4. On real CI this is the whole `uv run pytest` job going red on a correct tree.

## What was built

The tests were made **host-independent**, not loosened. The fix is *not* to restate `min(8,
os.cpu_count() or 8)` back at the implementation (that is the pre-existing test's shape, and it is a
tautology — it would keep passing if the cap constant or the clamp direction were wrong). Instead
each test now **pins the core count** for the behaviour it is actually about:

- `os.cpu_count` is pinned **high** (64) in the tests that assert the cap, so the clamp is inert and
  the assertion is about the cap and nothing else — a literal `8` / `5` that means what it says on
  every host. On the pii side this is a small `cores` fixture; on the extract side a one-line
  `monkeypatch.setattr`, matching how that file already works.
- The clamp gets its **own** test on each side, pinning the count **low** (4) —
  `test_the_cap_never_exceeds_the_host_core_count` (pii) and
  `test_the_intra_op_cap_never_exceeds_the_host_core_count` (extract). Asking torch for 8 intra-op
  threads on a 4-vCPU box is itself the oversubscription this guard exists to prevent, so the clamp
  is behaviour worth asserting rather than a detail to be worked around.

Net: the seam under test is unchanged, the coverage got *larger* (both halves of
`resolve_intra_op_threads`' documented contract are now pinned instead of one), and no deepening was
undone. No production code changed.

Deliberately **not** changed: `test_gliner2_fused_provider.py:1010`'s `min(8, os.cpu_count() or 8)`
assertion. It predates this branch (`main`, commit `06ca47f`), it is green on every host, and
rewriting it is out of this branch's scope.

## Acceptance criteria

- [x] Every intra-op-cap assertion is independent of the host's core count (verified green at a
      simulated 4 cores and on the real 10-core box).
- [x] The clamp half of `resolve_intra_op_threads`' contract has its own test on both sides.
- [x] Those clamp tests are non-vacuous: they fail if the clamp is removed from `perf.py`.
- [x] No test was weakened or deleted; no production code changed; no assertion restates the
      implementation formula.
- [x] `make test` green; `scripts/verify-local.sh` green.

## Evidence it failed pre-fix

Simulated `ubuntu-latest` (`os.cpu_count() -> 4`), against the tests as the branch had them:

```
>       assert recording_torch == [8], (
E       AssertionError: redaction.gliner2 must cap torch's intra-op threads at load; ...
E       assert [4] == [8]
packages/latence-pii-gliner2/tests/test_gliner2_pii_thread_cap.py:75: AssertionError

>       assert calls == [5]
E       assert [4] == [5]
packages/latence-extract-gliner2/tests/test_gliner2_device_seam.py:143: AssertionError

=========================== short test summary info ============================
FAILED packages/latence-pii-gliner2/tests/test_gliner2_pii_thread_cap.py::test_redaction_gliner2_caps_intra_op_threads_at_load
FAILED packages/latence-extract-gliner2/tests/test_gliner2_device_seam.py::test_intra_op_threads_are_capped_at_load
```

The pre-existing `test_intra_op_threads_capped_at_load` passed under the same simulation.

And the two new clamp tests, run with `return min(int(raw), int(cores))` in `perf.py` temporarily
replaced by `return int(raw)` (proving they are not vacuous):

```
E       assert [5] == [4]
FAILED packages/latence-pii-gliner2/tests/test_gliner2_pii_thread_cap.py::test_the_cap_never_exceeds_the_host_core_count
FAILED packages/latence-extract-gliner2/tests/test_gliner2_device_seam.py::test_the_intra_op_cap_never_exceeds_the_host_core_count
```

## Regression tests

- `packages/latence-pii-gliner2/tests/test_gliner2_pii_thread_cap.py::test_the_cap_never_exceeds_the_host_core_count`
- `packages/latence-extract-gliner2/tests/test_gliner2_device_seam.py::test_the_intra_op_cap_never_exceeds_the_host_core_count`
- plus the three repaired assertions in
  `test_redaction_gliner2_caps_intra_op_threads_at_load`, `test_the_cap_is_configurable`,
  `test_intra_op_threads_are_capped_at_load`.
