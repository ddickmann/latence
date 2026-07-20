# R4-acceptance_criteria-11 — the SBOM coverage guard does not protect the artifact that gets signed

**Severity:** major
**Stream:** #178 (release integrity / CRA component inventory), #179 (anti-false-green statement)
**Status:** PARTIALLY PRE-FIXED — the originally-reported scope bug was closed by `ec4a654`;
this ticket closes the structural remainder that commit left open.

## What the audit reported

> The per-release CycloneDX SBOM describes only the 12 workspace packages resolved into the
> CPU-first dev environment. 20 of the 32 distributions the same release signs and publishes,
> and every heavy third-party dependency they pull, are absent from it.

## What is actually true on this branch

The **scope half of the finding is already fixed**. Commit `ec4a654` added `--all-packages`
to `scripts/generate-sbom.sh` plus a fail-closed coverage check, and updated `docs/SBOM.md`
and `docs/RELEASING.md`. Measured on this tree:

| probe | result |
| --- | --- |
| `uv export --frozen --no-editable --no-emit-project` (old code) | 12 `./packages/...` entries |
| `uv export ... --all-packages` (current code) | **32** `./packages/...` entries |
| `bash scripts/generate-sbom.sh` | **179** components |
| torch / transformers / gliner / gliner2 / peft / sentence-transformers / openai / presidio-analyzer / python-docx / openpyxl / python-pptx / pypdfium2 / glinker / flashdeberta | **all present** |

So the released SBOM is no longer a false negative. The audit's `--sbom`-override sentence
and its "54 pinned components" doc quote both describe pre-`ec4a654` state.

## The remainder that is still a real defect

The finding's structural claim survives verbatim:

> nothing anywhere compares the SBOM's components against the distributions actually being
> released.

The **only** coverage guard lives inside `scripts/generate-sbom.sh`, and it validates the
**intermediate requirements file**, not the document that is stamped, checksummed and signed.
`scripts/release-integrity.sh` is the choke point every release SBOM passes through, and its
validation (lines ~112-135) is only:

- `bomFormat == "CycloneDX"`
- `specVersion` non-empty
- `components` is a non-empty list

Two ways a coverage-blind SBOM still reaches a signed release:

1. **`--sbom PATH` bypasses the generator entirely.** The caller-supplied document is `cp`'d
   in and receives only the three checks above. A document listing one component passes, is
   stamped `latence-framework <version>`, digest-recorded in `SHA256SUMS`, and Sigstore-signed.
2. **The guard is on the wrong file.** Even on the generate path, the check reads the temp
   requirements export. Anything that drops components between that file and the emitted
   CycloneDX JSON is invisible to every gate.

`test_refuses_to_attach_a_useless_sbom` only covers wrong-format / no-specVersion / empty /
non-JSON. The test fixture `_CYCLONEDX_DOC` is a 2-component document with no workspace
packages at all — and it currently passes the gate.

## Failure scenario

A release is cut with `--sbom` pointing at a stale or hand-edited document (a cached artifact
from a previous run, a partial regeneration, a truncated download). It contains 12 workspace
packages and none of the Provider dependency surface. `release-integrity.sh` accepts it,
stamps `metadata.component = latence-framework <version>`, records its digest in the
Sigstore-signed `SHA256SUMS`, and `VERIFY.md` tells the adopter it is "a CycloneDX SBOM for
this exact release". An adopter who installs `latence-pii-gliner2` feeds it to grype or
Dependency-Track to answer "am I affected by CVE-X?" for a CVE in gliner2/peft/torch. The
scan returns **CLEAN** — a cryptographically authenticated false negative on the security
evidence artifact. No gate can see it.

## Acceptance criteria

- [x] The coverage check runs on the **final CycloneDX document** inside
      `release-integrity.sh`, before stamping/checksumming/signing — not only on the
      generator's intermediate requirements file.
- [x] The check applies to the `--sbom` override path, so a supplied document cannot bypass it.
- [x] The failure is **fail-closed and typed**: a named error naming the missing packages, no
      `SHA256SUMS` written, non-zero exit (ADR-0034 — no silent errors).
- [x] The test fixture `_CYCLONEDX_DOC` reflects a real release SBOM (carries the workspace
      path components), so the guard is enforced unconditionally rather than special-cased.
- [x] A non-vacuous regression test fails on the old code and passes on the fix.
- [x] `make test` stays green at or above 2985 passed / 28 skipped; no test weakened or deleted.
