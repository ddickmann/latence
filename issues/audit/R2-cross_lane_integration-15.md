# R2 / cross-lane integration #15 — the served-OCR GPU environment ships 25 proprietary NVIDIA wheels that the third-party licence register never mentions

**Severity:** major
**Lane collision:** #196 (served-OCR vLLM environment) × #198 (CUDA-wheel licence disclosure) × #189 (go-live: "third-party license version snapshot refreshed")
**Touches:** `THIRD-PARTY-LICENSES.md`, `deploy/env/served-ocr-server/requirements.lock`,
`packages/latence-core/src/latence_core/compat.py` (`_check_licenses`)

## What is wrong

`THIRD-PARTY-LICENSES.md` opens with an unqualified claim:

> Every runtime dependency is a permissive license (Apache-2.0 / MIT / BSD / ISC / PSF / MPL-2.0).

The #198 lane later added a carve-out section, "CUDA runtime wheels — NOT permissive", but scoped it
**explicitly and only** to `deploy/env/gpu-learned/requirements.lock` (15 `nvidia-*` wheels via
`torch==2.13.0+cu130`).

In parallel, the #196 lane committed a second pinned GPU environment,
`deploy/env/served-ocr-server/`, whose lock resolves **25** `nvidia-*` distributions via
`torch==2.11.0+cu130` / `vllm==0.24.0` / `flashinfer-python==0.6.12`. Nothing about that environment
appears in the register: `grep -c "served-ocr" THIRD-PARTY-LICENSES.md` → `0`, and `vllm` /
`flashinfer-python` appear nowhere in the register at all.

Two lanes shipped a pinned GPU environment. Only one disclosed. ADR-0012 says licences are verified
before a dependency is named; #189's acceptance criterion ("third-party license version snapshot
refreshed") silently fails on the very page the go-live flip publishes.

## Why no gate caught it

`compat._check_licenses` skips every cell whose `kind` is not `served_model`:

```python
for cell in matrix.cells:
    if cell.kind != _KIND_SERVED_MODEL:
        continue
```

so the matrix's `vllm` and `flashinfer-python` **distribution** cells are accepted carrying no
licence field at all, and no gate in the tree relates a committed lock to the licence register.
`uv run latence env matrix deploy/env/compatibility-matrix.yaml` and `bash scripts/verify-local.sh`
both stay green.

## Failure scenario

```
$ grep -c '^nvidia-' deploy/env/served-ocr-server/requirements.lock
25
$ grep -c 'served-ocr' THIRD-PARTY-LICENSES.md
0
$ grep -rn 'vllm\|flashinfer' THIRD-PARTY-LICENSES.md
(nothing)
```

An operator or downstream redistributor reads the register before standing up
`deploy/env/served-ocr-server/`, is told every runtime dependency is permissive, and installs 25
distributions carrying NVIDIA's proprietary CUDA / cuDNN / NCCL / cuSPARSELt / CUTLASS EULA — which
is exactly the surprise the #198 section says must "be a decision, not a surprise".

## What to build

1. **Disclose.** Rewrite the carve-out section so it is scoped to *every* committed environment that
   pins proprietary wheels, not one lock. Name each `nvidia-*` distribution of BOTH locks with the
   licence actually read back from the PyPI JSON API (several are permissive — `nvidia-ml-py` BSD,
   `nvidia-cudnn-frontend` MIT, `nvidia-nvtx` Apache-2.0 — and saying so is part of the disclosure).
   Add `vllm`, `flashinfer-python`, `flashinfer-cubin`, `triton==3.6.0` and the remaining served-OCR
   serving dependencies to the register with verified licences. Qualify the header claim so it no
   longer asserts something the tree contradicts.
2. **Gate it.** Extend `compat` so the omission cannot recur:
   * every distribution named in a committed `requirements.lock` that is proprietary (or belongs to
     the `nvidia-*` family, whose metadata is unreliable) must be named in `THIRD-PARTY-LICENSES.md`,
     and the lock's own path must be cited there;
   * every `kind: distribution` cell of the compatibility matrix must be named in the register too —
     closing the `vllm` / `flashinfer-python` hole that `_check_licenses` waved through.

## Acceptance criteria

* `THIRD-PARTY-LICENSES.md` names `deploy/env/served-ocr-server/requirements.lock`, all 25 of its
  `nvidia-*` distributions, and `vllm` / `flashinfer-python`, each with a verified licence and a
  verification date; the header no longer claims blanket permissiveness without the carve-out.
* A new finding code fires when a lock pins a proprietary/`nvidia-*` distribution that the register
  does not name, and when a matrix distribution cell's component is absent from the register.
* A regression test injects both omissions and asserts the finding; it FAILS on the pre-fix code.
* `make test` stays green at or above 2985 passed / 28 skipped; `scripts/verify-local.sh` green.
* No protected invariant regressed (ADR-0016 thin core: the gate stays offline, import-free,
  filesystem-and-YAML only).
