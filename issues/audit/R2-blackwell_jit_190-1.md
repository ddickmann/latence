# R2 / #190-5 — the JIT guards scan a HARDCODED 5-path tuple, so the sixth in-tree DeBERTa-backed Provider (`disambiguation.glinker`) has zero coverage

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199)
**Relates to:** #190 (Blackwell/CUDA-13 TorchScript nvrtc crash), R1-blackwell_jit_190-1..4, ADR-0036

## What is wrong

`packages/latence-core/tests/test_perf_fp32_regression_guard.py:295` defines
`_MDEBERTA_PROVIDER_SOURCES` as a **literal 5-entry tuple**. It is the parametrization for all three
#190 structural guards (lines 341, 467, 572):

* `test_mdeberta_providers_load_through_a_jit_neutralizing_seam` (routing),
* `test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone` (ordering — the R1-#190-4 teeth),
* `test_no_provider_hand_rolls_the_pytorch_jit_workaround`.

There is **no in-tree discovery** anywhere in the file — no glob, no completeness assertion — so the
tuple is only as true as whoever last edited it. A sixth DeBERTa-backed Provider already exists and
is **not in it**:

`packages/latence-disambig-glinker/src/latence_disambig_glinker/backends.py`
(`GLinkerBackend._init_executor`) is DeBERTa-backed by its own code comment (backends.py:328-338:
*"the mdeberta relative-position helper is otherwise scripted with `@torch.jit.script` and dies in
nvrtc on Blackwell/CUDA-13"*). It calls `neutralize_torch_jit(provider_name="disambiguation.glinker")`
before `import torch` / `from glinker import ProcessorFactory` — correctly, today — but **nothing
asserts it stays that way**. It is also absent from the behavioural `_PROVIDERS` list in
`packages/latence-core/tests/test_perf_jit_import_ordering.py`, and
`grep -rn neutralize_torch_jit packages/latence-disambig-glinker/tests/` returns nothing.

The prose claims the opposite of the code, in three places:

* `test_perf_fp32_regression_guard.py:344` — "a sixth mdeberta Provider … this scan bites the moment
  its source lands";
* the module docstring — "it scans the REAL provider sources in-tree, so a new learned provider … is
  caught the moment its source lands";
* `docs/adr/ADR-0036*:293` — same claim.

A sixth provider already exists and is not scanned. This is the identical blind-spot class that let
`redaction.gliner_pii2` ship broken through R1 (R1-blackwell_jit_190-4).

## Failure scenario (proven)

1. Someone reorders `GLinkerBackend._init_executor` so `import torch` /
   `from glinker import ProcessorFactory` precede the `neutralize_torch_jit(...)` call — a
   plausible "group the imports at the top of the function" tidy-up.
2. The guard's own `jit_import_ordering_offenders()` DOES catch that shape — fed the mutated file by
   hand it returns
   `['_init_executor(): imports the deberta backbone at line 337 but only neutralizes the JIT at line 341 — too late to un-script it']`.
3. But **no test ever hands it that file**: it is not in `_MDEBERTA_PROVIDER_SOURCES`.
4. The mutation ships with `make test` and `scripts/verify-local.sh` fully green.
5. On an sm_120 / CUDA-13 pod, `disambiguation.glinker` dies with
   `nvrtc: failed to open libnvrtc-builtins.so.13.0` on the first `link_mentions` forward.

## What to build

Replace the hardcoded tuple with **in-tree discovery** plus a completeness assertion, so the file's
own docstring becomes true, and give `disambiguation.glinker` real behavioural coverage.

## Acceptance criteria

* **AC1** — The #190 structural guards parametrize over **discovered** in-tree sources, not a
  literal list: every `packages/*/src/**/*.py` that imports a DeBERTa backbone root
  (`gliner` / `gliner2` / `glinker`) is scanned by the ordering guard and by the
  no-hand-rolled-`PYTORCH_JIT` guard.
* **AC2** — A completeness regression test fails on the pre-fix tree, naming
  `latence-disambig-glinker/src/latence_disambig_glinker/backends.py` as an in-tree DeBERTa-backed
  source that no #190 guard scans.
* **AC3** — The routing guard (which legitimately applies only to the gliner-family Providers that
  load through `load_then_cast` / `load_gliner2_optimized`) is **derived** from the discovered set
  rather than hardcoded, so it too cannot drift. `disambiguation.glinker` calls the shared seam
  directly and carries no `*.from_pretrained` model load, so it is correctly outside the routing
  subset and inside the ordering subset.
* **AC4** — A meta-test proves discovery is non-vacuous: a freshly-landed provider source in a
  temporary package tree is picked up without editing any list.
* **AC5** — `disambiguation.glinker` gains BEHAVIOURAL import-ordering coverage in
  `test_perf_jit_import_ordering.py`: with a `glinker` module that scripts the mdeberta helper at
  **import** time (the real transformers shape) and a live-JIT fake torch, `_init_executor()` leaves
  the helper unscripted, and the anti-false-green companion shows the fake DOES reproduce the nvrtc
  crash when the JIT is live at import.
* **AC6** — The false prose is corrected or made true: the guard docstrings and ADR-0036 no longer
  claim coverage the code does not have.
* **AC7** — No production behaviour change (test/guard-only, plus doc text). `make test` stays at or
  above 2985 passed / 28 skipped; no test is weakened or deleted.
