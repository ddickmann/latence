# R5-cross_lane_integration-11 — `redact_financial` in both GPU deploy stacks lacks `pii_labels`

**Status: WONTFIX — DUPLICATE of [R5-wiring_dead_code-2](R5-wiring_dead_code-2.md), already fixed on
this branch at commit `4069523`. No code change made; ticket kept for audit traceability.**

Severity as reported: critical.
Lane: cross-lane integration (deploy config × Provider contract × #198/#199 gates).

## The reported finding

The `redact_financial` Stage in `deploy/env/gpu-learned/stack.yaml` and
`deploy/env/full-stack/stack.yaml` wires `redaction.gliner2` with `config: {device: cuda}` and no
`pii_labels`, while `Gliner2PIIRedactor.__init__`
(`packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:232-239`) hard-requires a
non-empty `pii_labels` list. The Stage is constructed by
`packages/latence-core/src/latence_core/runner.py:1347` (`provider_cls(stage.config)`), so both the
#198 placement gate (`deploy/env/gpu-learned/smoke.sh:99,107`) and the #199 full-stack verification
(`deploy/env/full-stack/verify-full-stack.sh:95-112`) would abort at Stage construction with a raw
`ValueError` traceback, before any Provider runs.

## Failure scenario (as reported)

`Gliner2PIIRedactor({'device':'cuda'})` raises
`ValueError: Gliner2PIIRedactor requires a non-empty 'pii_labels' list in its config`. Because
`redaction.gliner2` declares `compute="either"`, the device seam never skips the Stage, so on the
Blackwell host `latence env placement` reports `E-PLACEMENT-FAILED`, `latence stack validate` fails,
and `verify-full-stack.sh` fails at the composed `latence run` — the two headline deliverables of
#198 and #199 unable to execute. No offline gate catches it, because `env verify` and the compat
matrix are import-free and never construct a Provider.

## Why this is WONTFIX

The Provider-contract half of the finding is real and still reproduces — that is the Provider
behaving as designed:

```
$ uv run python -c "from latence_pii_gliner2.provider import Gliner2PIIRedactor; Gliner2PIIRedactor({'device':'cuda'})"
ValueError: Gliner2PIIRedactor requires a non-empty 'pii_labels' list in its config
  (the PII entity type set is supplied per Pipeline).
```

But the **config half — the premise that the committed stacks omit `pii_labels` — is no longer
true.** Commit `4069523` ("fix(audit-R5): The gpu-learned environment's stack declares
`redaction.gliner2` with config `{device: cuda}` only …") landed on this branch and fixed exactly
this defect, in exactly these two files, ahead of this ticket. Both stacks now carry the ADR-0044
financial-PII floor explicitly:

```
deploy/env/gpu-learned/stack.yaml -> ['person', 'email', 'phone number', 'address', 'iban',
  'credit card number', 'tax id', 'bank account number', 'social security number']
deploy/env/full-stack/stack.yaml  -> ['person', 'email', 'phone number', 'address', 'iban',
  'credit card number', 'tax id', 'bank account number', 'social security number']
```

That commit also added the general guard this finding calls for,
`packages/latence-core/tests/test_env_stack_constructible.py`, which closes the class rather than the
instance:

* `test_every_component_of_a_committed_environment_stack_constructs` — parametrised over **every**
  `deploy/env/*/stack.yaml`, actually calls `provider_cls(component.config)`. `ImportError` is
  skipped (an absent optional backend is an environment fact, ADR-0016), any other exception is a
  reported defect. It asserts a non-zero constructed count so it cannot pass vacuously.
* `test_the_guard_bites_when_a_stack_drops_a_required_config_key` — strips `pii_labels` back off the
  gpu-learned redactor and asserts the checker reports it. This is the bite test for precisely the
  shape this ticket describes.
* `test_the_learned_redactor_carries_the_financial_pii_floor` — import-free, so it is non-vacuous on
  a CPU host with no gliner2 extra installed: it asserts the committed `pii_labels` list exists and
  covers every label in `latence_core.setup.wizard.FINANCIAL_PII_LABELS` (ADR-0044).

Ten tests, all green. Re-fixing would be a no-op edit and a second, redundant guard.

## Residual, deliberately not actioned here

The finding's secondary claim — that a Stage-construction failure surfaces as a raw Rich traceback
rather than the typed `error [category]` path ADR-0034 prescribes — is accurate: `latence run` /
`_execute_pipeline_file` (`packages/latence-core/src/latence_core/cli.py:197-262`) has no
`except (LatenceError, ValueError)` handler, unlike `delta`/`retract`/`purge`/`env *`
(cli.py:976, 1011, 1058, 1102, 1140, 1173, 1236).

This is **out of scope for this ticket and for this audit stream**: `git show
main:packages/latence-core/src/latence_core/cli.py` shows `run` and `_execute_pipeline_file` are
byte-identical on `main`. It is pre-existing behaviour that this stream's changes do not introduce
or interact with, and it is a whole-CLI concern (every Provider config error on every stack), not a
deploy-config defect. It should be raised as its own ticket against the CLI error-surface lane.

## Acceptance criteria (all already satisfied by `4069523`)

1. `redaction.gliner2` in `deploy/env/gpu-learned/stack.yaml` and `deploy/env/full-stack/stack.yaml`
   declares a non-empty `pii_labels` list. **MET.**
2. That list is at or above the ADR-0044 financial-PII floor
   (`latence_core.setup.wizard.FINANCIAL_PII_LABELS`). **MET.**
3. An offline, CPU-host gate fails if any committed `deploy/env/*/stack.yaml` names a Provider with a
   config that Provider rejects — i.e. the gap is closed as a class, not patched as an instance.
   **MET** by `test_env_stack_constructible.py`.
4. That gate is demonstrably non-vacuous: it can be made to fail by reintroducing the defect.
   **MET** by `test_the_guard_bites_when_a_stack_drops_a_required_config_key`.
5. `make test` stays green at or above 2985 passed / 28 skipped. **MET** (no change made).
