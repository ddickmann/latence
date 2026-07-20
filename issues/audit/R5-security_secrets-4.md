# R5-security_secrets-4 — the shipped verification procedure cannot detect an ADDED release asset

**Severity:** major
**Where:** `scripts/release-integrity.sh` (the generated `VERIFY.md` + the `SHA256SUMS`
generation), `docs/RELEASING.md`, `docs/CONTINUITY.md`
**Guarding test that was green on a property the scheme did not have:**
`packages/latence-core/tests/test_release_integrity.py::test_checksum_verification_rejects_a_tampered_asset`

## What is wrong

`VERIFY.md` asserts *"Every asset of `${TAG}` is covered by `SHA256SUMS`"* and then ships a
two-step procedure: verify the Sigstore bundle over `SHA256SUMS`, then `sha256sum -c
SHA256SUMS`. `sha256sum -c` only checks the files the manifest *lists*; it never enumerates
the directory. So the claim "every asset is covered" is not a property the procedure can
check — it is an assumption about the upload, and the upload is exactly what an attacker with
release-write controls.

That defeats the one threat the keyless signature exists to stop: an attacker who can write
to the published Release but cannot mint a Sigstore identity.

## Failure scenario

An attacker obtains release-write (stolen maintainer PAT, or any workflow granted
`contents: write`) but cannot mint a Sigstore identity. They upload one extra file — e.g.
`latence_core-0.1.1-py3-none-any.whl` — to the published `v0.1.0` release. `SHA256SUMS` and
its Sigstore bundle are untouched.

The adopter follows `VERIFY.md` exactly:

* step 1 (`sigstore verify identity ... SHA256SUMS`) passes — `SHA256SUMS` is byte-identical
  to what the workflow signed;
* step 2 (`sha256sum -c SHA256SUMS`) passes — it verifies only the 66 filenames the manifest
  lists.

The adopter then does the obvious `pip install ./*.whl` and installs the attacker's wheel,
having just been told by the *signed* instructions that verification succeeded.

The guarding test parametrises only over files already listed in `SHA256SUMS` (the wheel,
`sbom.cdx.json`, `VERIFY.md`) and overwrites them. No test adds an unlisted file, so the gate
is green on a property the scheme does not have.

## Acceptance criteria

* [ ] The procedure shipped in `VERIFY.md` detects a file present in the asset directory that
      `SHA256SUMS` does not list, and fails naming that file.
* [ ] The coverage step is portable (GNU coreutils and macOS/BSD `shasum` alike) and needs no
      tool beyond what step 2 already requires.
* [ ] `VERIFY.md`'s prose no longer claims coverage the reader cannot check; the coverage
      claim is tied to the step that establishes it.
* [ ] `docs/RELEASING.md` and `docs/CONTINUITY.md` describe the same procedure the script
      actually emits (no "two commands" drift).
* [ ] A regression test runs the coverage step **extracted from the generated `VERIFY.md`**
      (not a re-implementation), asserting it passes on a clean asset set and fails on a set
      with one added, unlisted distribution. It must fail on the pre-fix script.
* [ ] `make test` stays green at or above 2985 passed / 28 skipped.
