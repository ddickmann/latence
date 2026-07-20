# R1 — docs/CONTINUITY.md contradicts itself and the release workflow on release signing

- **Severity:** major
- **Dimension:** docs-truth / anti-false-green (buyer-facing claims)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), issue #179 (the continuity statement)
- **Status:** fixed

## What is wrong

`docs/CONTINUITY.md` is the **buyer-facing** continuity & security statement: procurement
reads it and takes it at face value. Commit `7b972ec` rewrote §5.3/§5.4 to state the new,
true posture — releases **are** signed with keyless Sigstore — but left the two places that
assert the *opposite* as fact:

- **§6 (CRA readiness), line 255:** lists under *"What it does not give you, and you must
  cover yourself"* → **"a signed release chain (§5.4)"** — pointing at the very section that
  says the release chain *is* signed.
- **§7 (Verification appendix), line 279:** the row
  `| No release signing | absence of any cosign/sigstore/attestation step in `.github/workflows/release.yml` |`

Both are factually false. `grep -n -i sigstore .github/workflows/release.yml` returns 12+
hits, including the job `release-integrity` (`name: SBOM + sign (Sigstore keyless)`,
`id-token: write`) and a `github-release` step that hard-fails with
`"no Sigstore bundle — refusing to publish an unsigned release"`. `README.md` §*Security &
release integrity* and `docs/SBOM.md` both state releases are Sigstore-signed, so
`CONTINUITY.md` is the sole outlier.

Worse, the document **advertises** that this cannot happen. §7's preamble promises: *"If a
row's artifact is missing, the regression test … fails and the merge gate goes red — so this
page cannot quietly become aspirational."* The guard
(`packages/latence-core/tests/test_continuity_statement.py::test_signing_claim_matches_the_release_workflow`)
is **keyword-hollow**: with `workflow_signs=True` it only asserts
`not doc_claims_unsigned`, where `doc_claims_unsigned = "unsigned" in doc.lower()`.
`grep -i unsigned docs/CONTINUITY.md` returns nothing, so both false claims sail through and
the suite stays green.

## Failure scenario

A procurement evaluator reads the continuity statement to assemble a CRA evidence pack.
§5.4 tells them the release chain is signed with keyless Sigstore and `SHA256SUMS` carries a
`SHA256SUMS.sigstore.json` bundle. §6 then tells them a signed release chain is **not**
provided and they must cover it themselves; §7, the appendix explicitly framed as
machine-verified proof, states as verified fact that there is an *"absence of any
cosign/sigstore/attestation step"* in the release workflow. The evaluator either (a)
concludes the document is untrustworthy in a section whose whole premise is trustworthiness,
or (b) believes §6/§7 and needlessly builds compensating controls for signing that already
ships. Verified pre-fix:
`PYTHONHASHSEED=0 python -m pytest packages/latence-core/tests/test_continuity_statement.py
packages/latence-core/tests/test_front_door_truthfulness.py -q` → **13 passed**, green, with
both false claims in place.

## What to build

1. **Correct §6.** Remove "a signed release chain (§5.4)" from the *what it does not give
   you* list, and add the signed-release-chain evidence as a row in the CRA-needs table
   (integrity/authenticity of the delivered artifact is CRA Annex I evidence, so it belongs
   on the *supplied* side).
2. **Correct §7.** Replace the `| No release signing | absence of any … |` row with the true
   claim and its real artifacts (`scripts/release-integrity.sh`, the `release-integrity` job
   in `.github/workflows/release.yml`).
3. **Make the guard non-vacuous.** `test_signing_claim_matches_the_release_workflow` must
   detect *negative signing claims as claims*, not as a single keyword: a phrase-level check
   over the whole document (covering "unsigned", "no release signing", "not signed",
   "absence of any cosign/sigstore…", and "a signed release chain" appearing inside a
   *"does not give you"* sentence), plus a positive requirement that when the workflow signs,
   the document names the concrete signing artifact. The check must be direction-symmetric:
   if the workflow ever loses its signing step, the document must go back to saying so.

## Acceptance criteria

- [x] `docs/CONTINUITY.md` contains **no** claim that releases are unsigned / that a signed
      release chain is not provided — §5.4, §6 and §7 agree with each other and with
      `.github/workflows/release.yml`.
- [x] §7's verification-appendix row for signing cites artifacts that exist
      (`scripts/release-integrity.sh`, `.github/workflows/release.yml`).
- [x] The regression test **fails on the pre-fix document** (both false claims present,
      workflow signing) and **passes on the fixed document**, and is phrase-based rather
      than keyed on the single token `unsigned`.
- [x] The guard still fails in the opposite direction: if `release.yml` loses every signing
      token, the document is required to state the unsigned posture.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped.
