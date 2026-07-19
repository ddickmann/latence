# R5 / wiring_dead_code-2 — `redact_financial` is unconstructible in the committed GPU stacks

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199)
**Sites:**
`deploy/env/gpu-learned/stack.yaml` (`redact_financial`, ~line 95),
`deploy/env/full-stack/stack.yaml` (`redact_financial`, ~line 88),
vs `packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:231-239`

## What is wrong

Both committed environment stacks declare the blessed learned redactor

```yaml
  - name: redact_financial
    capability: redaction
    provider: redaction.gliner2
    config:
      device: cuda
```

with **no `pii_labels`**. `Gliner2PIIRedactor.__init__` hard-raises on a missing/empty
`pii_labels` (by design — the PII entity type set is supplied per Pipeline, ADR-0038 unions the
induced `pii_types` on top of it). The Stage therefore cannot be constructed at all.

The wizard's generated stacks are correct — `latence_core.setup.wizard` emits
`pii_labels: list(FINANCIAL_PII_LABELS)` for exactly this Provider. Only the two hand-written
environment stacks omit it, so the committed stacks disagree with the generator on the one
Provider ADR-0044's financial-PII floor rides on.

## Failure scenario

Probe over the committed files (`stack_components(...)` → `ProviderRegistry().load(name)(config)`
for every component):

```
== deploy/env/gpu-learned/stack.yaml
  redaction.gliner2 CONSTRUCT-FAIL ValueError: Gliner2PIIRedactor requires a non-empty
  'pii_labels' list in its config (the PII entity type set is supplied per Pipeline).
== deploy/env/full-stack/stack.yaml
  redaction.gliner2 CONSTRUCT-FAIL ValueError: ... (same)
```

Every other component in both stacks constructs.

On a real CUDA-13 pod `bash deploy/env/gpu-learned/smoke.sh` therefore reports
`E-PLACEMENT-FAILED` at **gate 3** (`latence env placement`) and aborts **gate 4**
(`latence stack validate`) — making #198's GPU-placement proof structurally unreachable for one of
the five declared `learned_providers`.

## Why the green gate cannot see it

* **Gate 1** (`latence env verify`) — the only gate `scripts/verify-local.sh` runs — checks provider
  *names* against the stack. It never constructs a Provider, so it reports
  `environment 'gpu-learned': OK (no findings)`.
* `LATENCE_CUDA=0 latence stack validate deploy/env/gpu-learned/stack.yaml` returns PASS because the
  device seam skips-with-flag every `device: cuda` Stage
  (`redact_financial [cuda requested but no CUDA device is available (skipped)]`) **before** the
  Provider is constructed.

## What to build

1. Declare the ADR-0044 financial-PII floor explicitly on the `redact_financial` Stage in **both**
   `deploy/env/gpu-learned/stack.yaml` and `deploy/env/full-stack/stack.yaml`, using the same label
   set the generator emits (`latence_core.setup.wizard.FINANCIAL_PII_LABELS`).
   The Provider's hard-raise is correct and must NOT be weakened into a silent default.
2. Add a structural guard that catches this whole class: every component of every committed
   `deploy/env/*/stack.yaml` must be **constructible** from its committed config.

## Acceptance criteria

* `stack_components(p)` → `ProviderRegistry().load(c.provider)(dict(c.config))` succeeds for every
  component of every committed `deploy/env/*/stack.yaml`, for every Provider whose heavy backend is
  importable in the running environment (a missing optional dependency is a skip, a `ValueError` /
  `ConfigError` is a failure).
* The guard is non-vacuous: it is proven to BITE by stripping `pii_labels` from a copy of the
  committed `gpu-learned` stack and asserting the checker reports `redaction.gliner2`.
* `redact_financial` in both stacks declares a `pii_labels` list that is a superset of
  `FINANCIAL_PII_LABELS` — the floor claimed by `docs/compatibility-matrix.md` and ADR-0041 §7a is
  the floor the file actually carries.
* `make test` stays green at ≥ 2985 passed / 28 skipped. No test weakened or deleted.
