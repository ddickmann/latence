# R2-docs_truth-8 — every deployment doc sends a Blackwell operator at `--enforce-eager`, the fix the matrix says is NOT the fix

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199)
**Status:** FIXED

## The contradiction

`deploy/env/compatibility-matrix.yaml` § `FM-FLASHINFER-SM120` is this stream's own machine-checked,
`verified`-observation record of a real Blackwell (`sm_120`) startup abort. Its recorded resolution is:

> Set `VLLM_USE_FLASHINFER_SAMPLER=0` for the serve. This disables only the SAMPLER, so cudagraphs
> stay ON — it is deliberately **NOT** `--enforce-eager`, which would cost the throughput the served
> path exists for.

`deploy/env/served-ocr-server/serve.sh:24-27` is the only file in the tree that carries it.

Every operator-facing deployment surface told the operator the opposite:

| Surface | What it said |
| --- | --- |
| `docs/deployment.md:62` (§ GPU requirements + the Blackwell note) | "Add `--enforce-eager` there (a ~perf floor) **or** upgrade flashinfer." |
| `docs/deployment.md:122` (`vllmOcr.extraArgs` row) | "Add `--enforce-eager` on Blackwell." |
| `docs/serving-lighton-vllm.md:50` | option 1 of 2 = "add `--enforce-eager`" |
| `docs/TUTORIAL-fresh-pod-walkthrough.md:222, :1051` | "Either add `--enforce-eager` …"; troubleshooting-table Fix column |
| `deploy/docker-compose.yml:45` | "add `--enforce-eager` here (a ~perf floor)" |
| `deploy/helm/latence/values.yaml:101` | "Add `--enforce-eager` on Blackwell (SM 12.x)" |
| `docs/adr/0036-…:304` | "→ add `--enforce-eager` there (a ~perf floor) OR upgrade flashinfer" |

`VLLM_USE_FLASHINFER_SAMPLER=0` appeared in **no** deployment doc, and was **not settable at all**
through the Helm chart — `deploy/helm/latence/templates/vllm-ocr.yaml` rendered a fixed `env:` block
(`HF_HOME` + optional `HF_TOKEN`) with no operator-supplied env passthrough. The Helm/VPC path is the
documented Option B production path.

## Failure scenario

An operator stands up Option B (Helm) on a Blackwell node. vLLM aborts with
`FlashInfer requires sm75+`. They open `docs/deployment.md` § "GPU requirements + the Blackwell note",
which routes them to the `vllmOcr.extraArgs` row, which says to add `--enforce-eager`. They do. vLLM
starts and serves correct output — silently, permanently, **without cudagraphs**: the exact throughput
floor the verified field report says to avoid, on the served path whose entire reason to exist is the
~2.7× throughput over in-process transformers. The one-line fix that keeps cudagraphs on is reachable
from none of the pages they were routed to, and could not be expressed in the chart even if they knew it.

## What was built

1. **All operator-facing surfaces lead with the matrix's recorded resolution**
   (`VLLM_USE_FLASHINFER_SAMPLER=0`) and state explicitly that `--enforce-eager` is not the fix:
   `docs/deployment.md`, `docs/serving-lighton-vllm.md`, `docs/TUTORIAL-fresh-pod-walkthrough.md`
   (both the §2 callout and the §9 troubleshooting row), `deploy/docker-compose.yml`,
   `deploy/helm/latence/values.yaml`. ADR-0036 keeps its historical text and gains a
   **superseded-by** note pointing at `FM-FLASHINFER-SM120`.
2. **The Helm chart can express the fix.** New `vllmOcr.extraEnv` (list of `{name, value}`,
   default `[]` — off-by-default is byte-identical: an empty list renders no extra env entries),
   rendered by `deploy/helm/latence/templates/vllm-ocr.yaml` after the built-in env. Documented in
   `values.yaml` with the Blackwell setting as the worked example, and as a row in
   `docs/deployment.md`'s values table.
3. **`FM-FLASHINFER-SM120.enforced_by`** now lists the deployment surfaces that carry the resolution,
   so the existing `latence env matrix` verifier fails if one is deleted or renamed.
   `docs/compatibility-matrix.md` regenerated from the YAML.

## Acceptance criteria

- [x] No operator-facing deployment surface recommends `--enforce-eager` as the remedy for the
      `sm75` failure signature.
- [x] Every surface naming that signature also names `VLLM_USE_FLASHINFER_SAMPLER=0`.
- [x] The resolution is settable through the Helm chart without editing templates.
- [x] The rule is enforced by a regression test whose expected values are **derived from
      `deploy/env/compatibility-matrix.yaml`**, not hard-coded — changing the matrix's recorded
      resolution changes what the docs are required to say.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.

## Regression test

`packages/latence-core/tests/test_blackwell_flashinfer_docs_truth.py`
