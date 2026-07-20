# R6 — `test_env_stack_constructible` reports PASS for `cpu-text` while constructing nothing of its Export

**Severity:** major
**Area:** test quality (false green) + a real production config-validation defect
**Files:**

- `packages/latence-core/tests/test_env_stack_constructible.py:60` — the blanket `except ImportError: continue`
- `packages/latence-core/src/latence_core/stages/export.py:584` — `isinstance(provider, protocol)` (root cause)
- `packages/latence-splade/src/latence_splade/provider.py:189` — `vocab_size` property
- `packages/latence-splade/src/latence_splade/model.py:135-137` — eager HF load

## What is wrong

The R5 guard ("every committed environment-stack component is constructible") states its own premise
in the module docstring:

> Construction is cheap and offline: every in-tree Provider defers its heavy backend import to first
> use (ADR-0016), so `__init__` only validates config.

That premise is **false** for `export.jsonl_parquet`. `Export._resolve_signal` verifies the resolved
Provider against a `runtime_checkable` Protocol with `isinstance(provider, protocol)`. CPython
implements that check with `hasattr` over the Protocol's members — and `SparseEmbedder` declares a
**property**, `vocab_size`. So the `isinstance` call *invokes* `SpladeSparseEmbedder.vocab_size`,
which calls `TransformersSpladeModel._load()`, which imports `transformers` and calls
`AutoTokenizer.from_pretrained(...)` / `AutoModelForMaskedLM.from_pretrained(...)` — **at Stage
construction time**.

Two consequences:

1. **False green.** In the dev/CI env (`transformers` absent) that eager load raises `ImportError`,
   which the guard's blanket `except ImportError: continue` swallows as "optional extra missing —
   an environment fact, not a stack defect". So `deploy/env/cpu-text/stack.yaml` — the single most
   elaborate committed config in the tree (nested `embedder` **and** `sparse_embedder` specs) —
   receives ZERO construct-validation from the gate built to construct-validate it, while the test
   reports green (`assert constructed` only demands `>= 1`).
2. **Real defect on the production path.** Where `transformers` IS installed (GPU pod, CI with the
   splade extra), the same "cheap, offline" check downloads and instantiates a Hugging Face
   checkpoint. No `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` is set anywhere in `conftest.py`,
   `pyproject.toml`, `scripts/` or `.github/`, so the merge gate reaches the network. The same
   eager load fires on `latence stack validate stacks/...` — config validation performs a model
   download.

## Failure scenario (proven)

Running the guard's own `_unconstructible` over the four committed stacks in the repo venv:

```
cpu-text          constructed=14  importerror-skipped=['export.jsonl_parquet']
full-stack        constructed=14  importerror-skipped=[]
gpu-learned       constructed=14  importerror-skipped=[]
served-ocr-client constructed=6   importerror-skipped=[]
```

with the skip's traceback originating in
`typing.__instancecheck__` → `hasattr` → `provider.vocab_size` → `model._load` → `ImportError`.

## What to build

1. **Root fix (core):** a capability check that is `isinstance`-equivalent *structurally* but never
   **invokes** a member — resolve Protocol members against the provider's **class** first (a
   property object is present on the class without being called), falling back to the instance only
   for plain instance attributes. Use it at every construction-time capability seam that resolves a
   nested `{provider, config}` spec.
2. **Test fix:** drop the blanket `except ImportError: continue`. Given (1), construction really is
   offline and dependency-free, so an `ImportError` at construction is exactly the defect class this
   guard exists for and must be REPORTED, not swallowed.

## Acceptance criteria

- `Export`/`JsonlParquetExport` construction with a `sparse_embedder` (or `embedder`,
  `multivector_embedder`, `fde_converter`) spec **never** reads a property of the resolved Provider;
  a Provider whose capability property raises on access is still accepted/rejected correctly.
- Constructing every component of `deploy/env/cpu-text/stack.yaml` succeeds offline with
  `transformers` absent — `constructed == 15`, zero skips.
- A mis-wired Provider (one that does not satisfy the Protocol) still raises `ContractError` at
  wiring time — the loud-failure contract of ADR-0034 is unchanged.
- The guard reports an `ImportError` at construction as a defect rather than skipping it.
- A regression test fails on the old code and passes on the fix.
- `make test` stays at or above 3320 passed / 29 skipped.
