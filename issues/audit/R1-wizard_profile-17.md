# R1-wizard_profile-17 — the go-live tutorial never installs the package that registers the blessed redactor

**Severity:** major
**Where:** `docs/TUTORIAL-fresh-pod-walkthrough.md` §1.2 (editable-install block; the `gliner` bullet), §1.4 (verify probe)
**Introduced by:** #197 (`BLESSED_REDACT_PROVIDER = "redaction.gliner2"` in BOTH wizard profiles)

## What is wrong

`packages/latence-core/src/latence_core/setup/wizard.py` blesses `redaction.gliner2` for every
profile. The **only** distribution that registers that entry point in the `latence.providers`
group is `packages/latence-pii-gliner2`:

```toml
# packages/latence-pii-gliner2/pyproject.toml
[project.entry-points."latence.providers"]
"redaction.gliner2" = "latence_pii_gliner2.provider:Gliner2PIIRedactor"
```

The tutorial's §1.2 editable-install list installs `packages/latence-pii-gliner` (which registers
`redaction.gliner_pii`) and never `packages/latence-pii-gliner2`. Worse, the §1.2 prose explicitly
tells the reader the blessed redactor "needs no package beyond the `gliner2` above" — conflating
the third-party PyPI wheel `gliner2` with the Provider distribution `latence-pii-gliner2`.

On `main` the doc was self-consistent (the `enterprise-sota` wizard emitted `redaction.gliner_pii`,
matching the installed `latence-pii-gliner`), so this stream introduced the break.

Nothing covers it: no test or gate references `TUTORIAL`, so a fully green suite cannot see it.

## Failure scenario

Fresh GPU pod. Operator follows §1.2 verbatim, then §3's documented fast path
`latence setup --profile enterprise-sota` → `latence process`. The emitted config wires
`redact.provider: redaction.gliner2`, which is not registered. The run aborts at the `redact`
Stage with a raw
`KeyError: "No Provider registered as 'redaction.gliner2' in group 'latence.providers'. Available: ..."`.
§1.4's own probe would print `MISSING redaction.gliner2`, while its explanatory note says that
means "its package wasn't installed in §1.2" — but §1.2 lists no package that provides it.

## What to build

1. Add `-e packages/latence-pii-gliner2` to the §1.2 editable-install block.
2. Rewrite the redactor bullets so the Provider **distribution** and the PyPI **wheel** are named
   distinctly, and the blessed-vs-supported-twin relationship is stated truthfully.
3. Add a machine gate so the tutorial can never again promise a stack it does not install.

## Acceptance criteria

* **AC1** — Every Provider the wizard emits for `--profile enterprise-sota` (the profile §3
  documents) is registered by a distribution the tutorial's §1.2 editable-install block installs,
  or by `latence-core` itself.
* **AC2** — Every Provider name the §1.4 verify probe asserts discoverable resolves to a
  distribution §1.2 installs (or to `latence-core`), so `MISSING` always means operator error and
  never a doc lie.
* **AC3** — The guard is derived from the repository (parsed install block × parsed
  `packages/*/pyproject.toml` entry points × the real wizard emission), never from a restated
  literal, so it keeps failing if the blessed Provider changes again.
* **AC4** — `make test` stays green at or above 2985 passed / 28 skipped.
