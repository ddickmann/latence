# Reproducible environments (`deploy/env/`)

An **environment** is a pinned, one-command-installable Python environment for one slice of the
framework's Provider ecosystem, plus a smoke test that fails when it stops being reproducible.

This directory is the answer to a specific, expensive problem: standing up the enterprise-sota
Provider set used to be per-session trial-and-error. `torch` version churn ABI-breaks prebuilt
`flash-attention` / `torchcodec` wheels; the GLiNER family and the OCR model disagree on the
`transformers` major; externally-managed-Python (PEP 668) blocks naive installs; the Torch
JIT/nvrtc path breaks on CUDA 13; Blackwell (`sm_120`) trips stale architecture checks. Every one
of those was discovered at runtime, on a pod, hours in.

The cure is not a wiki page. **The pins are enforced constraints the build checks**, and the smoke
test is the regression guard.

| Environment | Slice | Ticket |
| --- | --- | --- |
| [`cpu-text/`](cpu-text/) | CPU/text enterprise-sota: GLiNER2 fused extraction, learned GLiNER PII redaction, GLinker disambiguation, Granite r2 dense embeddings, SPLADE sparse | #195 |
| _served-ocr/_ | the served-OCR environment (vLLM-hosted LightOn-OCR + the light HTTP Provider set) | #196 |
| [`gpu-learned/`](gpu-learned/) | GPU learned Providers on CUDA 13 / Blackwell: GLiNER2 fused extraction, GLiNER2 financial-PII + learned GLiNER PII redaction, GLinker disambiguation, Granite r2 GPU-tier embeddings — with the `transformers` split resolved and **real GPU placement asserted** | #198 |

## Using one

```bash
bash deploy/env/cpu-text/install.sh    # the ONE documented install path (creates a venv; PEP-668 safe)
bash deploy/env/cpu-text/smoke.sh      # the ONE command that proves it stands up
bash deploy/env/cpu-text/smoke.sh --spec-only   # just the offline gate — no install, no network
```

`scripts/verify-local.sh` runs `--spec-only` for **every** environment in this directory, so
dependency drift fails the merge gate on any machine.

## The `transformers` split, and the one supported resolution

The GLiNER family (`gliner`, `gliner2`, `glinker`, `flashdeberta`) is written against the
`transformers` **4.x** API. The OCR VLMs need **5.x**. One interpreter cannot hold both.

There is exactly **one** supported resolution, and it is the ADR-0030 endpoint boundary:

| side | what lives there | environment |
| --- | --- | --- |
| in-process, `transformers` 4.x | the GLiNER/mdeberta learned Provider set | [`gpu-learned/`](gpu-learned/) |
| served, `transformers` 5.x | the OCR VLMs, behind an HTTP endpoint | `served-ocr/` (#196) |

`gpu-learned` reaches OCR through `parser.endpoint` over HTTP. A **two-phase install into one
interpreter was rejected**: one `site-packages` holds one `transformers`, so whichever phase ran
second wins and the other half breaks at model load — the failure is deferred, not resolved.

That resolution is **enforced, not documented**. `gpu-learned/environment.yaml` names the
in-process OCR packages under `excluded_packages`, and gate 1 refuses them whether they are listed
directly or dragged in as another package's workspace dependency. Since gate 1 runs *before*
`install.sh`, the unsupported combination fails at dependency resolution — never at runtime.

## The shape (every environment has exactly these files)

| File | Hand-written? | What it is |
| --- | --- | --- |
| `environment.yaml` | **yes — the single source of truth** | which workspace packages the environment installs, the ENFORCED conflict boundaries *with the reason each exists*, the Provider set it must expose, the smoke stack, the target platform and indexes |
| `requirements.in` | no — **generated** | the direct third-party requirements, derived from the Provider packages' real `pyproject.toml` |
| `constraints.txt` | no — **generated** | the enforced conflict boundaries, passed to the installer as `-c` |
| `requirements.lock` | no — **solved** | the exact resolution (`uv pip compile`; regenerate with `install.sh --relock`) |
| `stack.yaml` | yes | the tiny end-to-end slice the smoke test drives |
| `install.sh` | yes | the one documented install path — no undocumented steps, no `--break-system-packages` |
| `smoke.sh` | yes | the one-command guard (three gates, below) |

`requirements.in` and `constraints.txt` are regenerated with
`latence env render deploy/env/<name>/environment.yaml`. Never edit them by hand — the verifier
byte-compares them against the render and fails if they drift.

## The three gates

`smoke.sh` runs them cheapest-first:

1. **Spec** — `latence env verify` (offline, import-free, no network). Cross-checks the generated
   files, the lock, the Provider packages' declared dependencies and entry-points, and the smoke
   stack. This is the gate that runs on every commit.
2. **Stand-up** — `latence env verify --probe` loads every declared Provider through the registry,
   which imports the heavy stack. A `transformers`/`torch` major mismatch raises at module import,
   in seconds, before a single weight is fetched.
3. **End-to-end slice** — `latence stack validate stack.yaml` runs the tiny corpus through the
   real Providers and asserts the contract/KG/RAG/report/determinism/resume checks.

A **GPU** environment adds a gate between 2 and 3:

* **GPU placement** — `latence env placement` drives one real inference through every Provider in
  `learned_providers` and reads torch's CUDA allocator around the call. A model that silently
  stayed on the CPU allocates **zero CUDA bytes** however correct its output is. That is the #191
  signature (`device: cuda` configured, ~11.5 s/chunk, the GPU idle, no warning), and this gate is
  what makes it impossible to reintroduce. No CUDA device is a **failure**, not a skip.

Gates 2 and 3 need the installed environment. If it is absent, `smoke.sh` **fails** rather than
reporting success on gate 1 alone — a smoke test that quietly narrows its own scope is a false
green. Pass `--spec-only` to ask for gate 1 deliberately.

## What gate 1 actually catches

Every one of these is a real regression test in
`packages/latence-core/tests/test_envspec.py`, injected into a copy of a real environment:

| Drift | Finding |
| --- | --- |
| a Provider package gains a dependency nothing pins (e.g. `torchcodec`) | `E-REQUIREMENTS-STALE` |
| a Provider package raises a floor / drops a cap the lock cannot satisfy | `E-LOCK-UNSATISFIED` |
| the lock drifts past an enforced conflict boundary | `E-CONSTRAINT-VIOLATED` |
| a range sneaks into the lockfile instead of a solved version | `E-LOCK-NOT-PINNED` |
| a declared dependency has no pin at all | `E-LOCK-INCOMPLETE` |
| a Provider is renamed out from under the recipe | `E-PROVIDER-UNREGISTERED` |
| the smoke stack uses a Provider the environment does not install | `E-STACK-PROVIDER-UNDECLARED` |
| `constraints.txt` is hand-edited (a boundary added *or relaxed*) | `E-CONSTRAINTS-STALE` |
| a named workspace package is gone | `E-PACKAGE-MISSING` |
| a package's declared version window is disjoint from an enforced boundary | `E-CONSTRAINT-CONFLICT` |
| an excluded package enters the set, directly or as somebody's dependency | `E-PACKAGE-EXCLUDED` |
| a `learned_providers` entry the smoke stack never instantiates | `E-LEARNED-UNDRIVEN` |
| a `learned_providers` entry the environment does not install | `E-LEARNED-UNKNOWN` |

And, for a GPU environment, what `latence env placement` catches:

| Drift | Finding |
| --- | --- |
| a learned Provider ran and allocated zero CUDA bytes (**#191**) | `E-PLACEMENT-ON-CPU` |
| no CUDA device (a GPU gate must fail, not narrow itself) | `E-PLACEMENT-NO-CUDA` |
| a learned Stage pinned to `device: cpu` in a GPU stack | `E-PLACEMENT-DEVICE-NOT-CUDA` |
| a learned Provider with no driver — unexercised is not passed | `E-PLACEMENT-UNDRIVABLE` |
| the Provider could not be constructed or raised while driven | `E-PLACEMENT-FAILED` |

## Adding an environment (#196, #198)

1. `mkdir deploy/env/<name>` and write `environment.yaml` — the package set, the boundaries **with
   reasons**, the Provider set, the platform, the indexes.
2. Write `stack.yaml` (the smallest slice that touches every heavy leg) and copy
   `cpu-text/install.sh` + `cpu-text/smoke.sh`, adjusting the interpreter/platform/index constants
   at the top.
3. `latence env render deploy/env/<name>/environment.yaml`, then solve the lock with
   `bash deploy/env/<name>/install.sh --relock`.
4. `latence env verify deploy/env/<name>/environment.yaml` until green.

Nothing else is needed: `test_envspec.py` globs `deploy/env/*/environment.yaml` and
`scripts/verify-local.sh` globs `deploy/env/*/smoke.sh`, so a new environment is covered by the
merge gate the moment it lands.

## Model licensing

Per ADR-0012 every named model gets its **weights** licence and its **code** licence verified
*separately* before it is wired into a `stack.yaml`. Verifications for the models named here are
recorded inline in the stack file that names them.
