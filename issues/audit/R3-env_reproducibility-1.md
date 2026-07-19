# R3 — the compatibility matrix's "derived coverage" is silenced by an under-declared spec

- **Severity:** major
- **Dimension:** environment reproducibility / anti-false-green
- **Branch:** feat/release-fleet-integration
- **Tickets:** #199 (compatibility matrix), #195 (cpu-text enterprise-sota), ADR-0012, ADR-0053
- **Status:** fixed

## What is wrong

`deploy/env/compatibility-matrix.yaml`'s own header asserts a safety property:

> Coverage is DERIVED from the committed `deploy/env/*/environment.yaml` specs, so a new enforced
> boundary or learned Provider turns this file red until it is mapped.

That is false for the learned-Provider half. `required_components()`
(`packages/latence-core/src/latence_core/compat.py:430-449`) derived learned-Provider coverage
**exclusively** from the hand-written `spec.learned_providers` field, and nothing cross-checked
that field against the Providers the environment actually installs. `learned_providers` is
optional, so an environment that simply omits it obliges the matrix to nothing.

`deploy/env/cpu-text/environment.yaml` (issue #195) omits it entirely — `grep -c learned_providers`
= 0 — while declaring under `providers:` five Providers whose packages pull a model runtime:

| Provider | package | model-runtime dependency |
|---|---|---|
| `fused_entity_relation.gliner2` | `latence-extract-gliner2` | `gliner2[local]`, `peft` |
| `redaction.gliner_pii` | `latence-pii-gliner` | `gliner` |
| `disambiguation.glinker` | `latence-disambig-glinker` | `glinker`, `flashdeberta`, `transformers`, `torch` |
| `embedding.sentence_transformers` | `latence-embedder-st` | `sentence-transformers` |
| `sparse.splade` | `latence-splade` | `transformers` |

Measured pre-fix: `required_components(cpu-text)` returned only
`{transformers, huggingface-hub, tokenizers, torch}` — **zero** learned-Provider cells — and
`verify_matrix(matrix, repo_root).findings` returned `()`. The matrix therefore has no cell for
any Provider of the CPU/text environment, including the whole ADR-0053 sparse leg that #195 names
as part of the enterprise-sota stack, and nothing turns red about it.

Note on the reported finding: the sparse leg is **not** `BAAI/bge-m3` — the committed
`deploy/env/cpu-text/stack.yaml` wires
`opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` and
`THIRD-PARTY-LICENSES.md` records bge-m3 as cleared-but-deliberately-not-wired (no MLM head).
`disambiguation.embedding` is likewise **not** a model-loading Provider: `latence-disambig-embedding`
declares only `latence-core` and takes its vectors from an injected embedder Provider. The
under-coverage itself is real and is what this ticket fixes.

## Failure scenario

An operator reads `docs/compatibility-matrix.md` to decide what is known-good on Blackwell /
CUDA-13. `sparse.splade` — a `transformers`-loading Provider that this release ships and drives in
`deploy/env/cpu-text/stack.yaml` — appears nowhere on the page: not measured, and not marked
`UNVERIFIED` either. It is simply absent, so the page reads as if the CPU/text environment had no
learned Providers at all. Worse, the gate cannot recover: adding a sixth model-loading Provider to
`cpu-text` tomorrow produces no finding, because the derivation consults only a field that file
does not have. #199 AC1 ("the matrix covers every learned Provider … with each cell verified or
explicitly UNVERIFIED") is not met.

## What to build

Make the derivation **structural** instead of trusting a hand-maintained list:

1. `latence_core.envspec.model_loading_providers(spec, repo_root)` — derive, from the workspace
   packages' own `pyproject.toml` metadata (already parsed by `_read_package`), which of the
   Providers the environment declares are registered by a package that depends on a model-runtime
   distribution. No import, no network: pure metadata, so it stays inside the offline gate.
2. `required_components()` unions that derived set with `spec.learned_providers`, so matrix
   coverage no longer depends on a spec's willingness to declare. An environment that omits the
   field is covered anyway.
3. Add the resulting cpu-text cells to `deploy/env/compatibility-matrix.yaml` (resolution verified
   from the existing offline anchor, runtime `UNVERIFIED` with a runnable `reproduce` and a
   `blocked_by`), and regenerate the published `docs/compatibility-matrix.md`.

`learned_providers` keeps its distinct, narrower meaning — the `latence env placement` GPU probe
set — which is why `cpu-text` still does not declare one (a CPU environment proves no GPU
placement, and `E-LEARNED-UNDRIVEN` would fire for the `gpu_only` `disambiguation.glinker`).

Residual, stated honestly: `_MODEL_RUNTIME_DISTRIBUTIONS` is a curated list of runtime
distributions, matched against packages' **declared** dependencies. A future Provider package that
loads a model through some other loader distribution must add it to that set. That is a visible
one-line omission in core rather than a silently empty derivation, and the set is asserted against
the committed workspace by a test.

## Acceptance criteria

- [x] `required_components()` derives learned-Provider coverage from package metadata, so an
      environment that declares **no** `learned_providers` is still obliged to have a matrix cell
      for every model-loading Provider it installs.
- [x] The committed matrix carries a cell for each of cpu-text's five model-loading Providers, each
      one measured or explicitly `UNVERIFIED` with a runnable `reproduce` + `blocked_by`; the
      published `docs/compatibility-matrix.md` is the byte-identical render.
- [x] `disambiguation.embedding` (no model runtime of its own) is **not** demanded a cell, so the
      derivation is discriminating rather than "everything in `providers`".
- [x] A non-vacuous regression test fails on the old code (coverage silent for cpu-text) and passes
      on the fix, asserting through the public `verify_matrix` path that dropping any one of the
      new cells yields `E-MATRIX-CELL-UNCOVERED`.
- [x] `make test` stays green at or above the 2985 passed / 28 skipped baseline; the full gate
      `scripts/verify-local.sh` passes (config + docs render touched).
