# Continuity & security process

*The procurement answer to two questions: **what happens if the maintainer stops**,
and **how are vulnerabilities handled**.*

This document is deliberately unflattering where reality is unflattering. Every
claim below points at an artifact in this repository that proves it; the
[verification appendix](#7-verification-appendix) maps claim → artifact, and a
regression test (`packages/latence-core/tests/test_continuity_statement.py`) fails
the build if a claim here drifts from what the repository actually contains.
Nothing on this page is a roadmap item stated in the present tense.

---

## 1. Maintainer model — the honest bus factor

**The Latence Framework is maintained by one person.** Dennis Dickmann
(`ddickmann81@gmail.com`) is the sole maintainer, sole copyright holder, and sole
reviewer. `.github/CODEOWNERS` lists exactly one owner (`@ddickmann`) for every
path in the repository, including the security-sensitive ones. There is no team,
no company behind the project, no commercial support contract, and no second
person with commit rights.

**Bus factor: 1.** If the maintainer stops, nobody currently has the access to cut
a release, publish a security advisory, or merge a fix in this repository.

This is stated up front because the mitigations that follow are structural — they
work *without* the maintainer — rather than promises that depend on him.

The project is personally owned by design, not by accident: see
[ADR-0001](adr/0001-open-source-core-owned-personally.md) (open-source core,
personally owned, published as pre-existing IP). Ownership is deliberately kept
outside any employer.

## 2. What survives the maintainer: license & fork posture

- **Apache License 2.0** (SPDX `Apache-2.0`), repository root `LICENSE` — the canonical
  text byte for byte — with `NOTICE` and per-package `license = "Apache-2.0"` metadata on
  every publishable distribution. **Any** fork, commercial included, needs nobody's
  permission and does not depend on the maintainer's cooperation or continued existence.
  This is the single biggest change to this page's risk profile: the project carried
  PolyForm Noncommercial from 2026-07-29 to 2026-08-15 (ADR-0055), under which the
  commercial lane ran through one reachable person; [ADR-0063](adr/0063-relicence-to-apache-2-0.md)
  relicensed back to Apache-2.0 and removed that dependency entirely. Read §1 and §5 with
  that in mind — the bus factor still bites for *maintenance*, no longer for *rights*.
- **No carve-outs, no exceptions.** Every file in the repository is Apache-2.0, and
  `packages/latence-core/tests/test_publish_readiness.py` fails the build if any source
  header, package manifest or licence document disagrees. (Under PolyForm, seven query-time
  retrieval modules kept a file-level Apache-2.0 grant so those
  algorithms stayed commercially reusable; that exception was dissolved by the relicence, and
  the retrieval package itself has since been removed — ADR-0068.)
- **The patent grant is broad and contributor-wide.** Apache-2.0 §3 grants a patent licence
  from **every contributor**, for any purpose, terminating only if you initiate patent
  litigation alleging the software infringes (§3's defensive-termination clause). This is
  the clause enterprise legal reads after the identifier, and it is the reason ADR-0063
  chose Apache-2.0 over MIT, which grants no patent licence at all.
- **The grant is irrevocable.** Apache-2.0 §2 and §3 are perpetual and irrevocable, so a
  fork is durable unconditionally. There is no cure-period clause that can end your rights,
  and no field-of-use boundary anyone has to interpret.
- **A CLA is currently required, though the reason it existed has gone.**
  `CONTRIBUTING.md` → *Before you write code: the Contributor Licence Agreement*
  requires a signed CLA on every contribution, including one-line and documentation
  changes. It was written for the PolyForm model, where the owner could only grant
  commercial licences over code he held sufficient rights in; Apache-2.0 §5 makes every
  contribution arrive on the Work's own terms, so a DCO would now suffice and whether to
  switch is an open decision (ADR-0063). While it stands, it is a real concentration of
  rights in one person — it is the
  deliberate trade for the commercial lane, not an oversight, and it is the opposite of
  the inbound-equals-outbound posture this section previously described.
- **Permissive dependencies only.** `THIRD-PARTY-LICENSES.md` records every runtime
  dependency and every default model with its licence, and
  [ADR-0012](adr/0012-permissive-only-reference-providers-gemma-opt-in.md) makes
  permissive-only the *default* rule: restricted-licence models (EmbeddingGemma)
  are opt-in and never a default. Model **weights** and model **code** licences are
  verified separately. A fork therefore inherits no copyleft or field-of-use surprise
  *from the dependency tree* — the field-of-use restriction it does inherit is the
  project's own, stated in the first bullet. The one disclosed exception is the CUDA wheel set pulled in
  by the two pinned GPU environments (`deploy/env/gpu-learned/`,
  `deploy/env/served-ocr-server/`), which carry NVIDIA's proprietary EULAs; each wheel
  is named in `THIRD-PARTY-LICENSES.md` and the disclosure is machine-gated
  (`E-LICENCE-UNDISCLOSED`), so a new GPU environment cannot ship undisclosed.
- **No hosted service, no phone-home, no licence key.** The framework is a library
  plus a CLI that runs on the adopter's own machines; there is no server that can be
  switched off. Endpoint Providers call an endpoint *the operator chooses and
  configures*. Nothing about a running deployment depends on this project's
  infrastructure continuing to exist.
- **No proprietary build infrastructure.** `bash scripts/verify-local.sh` reproduces
  the entire merge gate on a laptop with no GitHub Actions, no runners, and no
  secrets. A fork can build, test, and gate itself on day one.

**Concretely: the worst case is a fork.** An adopter who needs a fix the maintainer
never ships can fork, apply it, and build a release themselves, using the same gate
this repository uses.

## 3. Docs-as-handover — the design is legible without the author

The documentation is structured as a handover to a stranger, and this is enforced
in review rather than left to goodwill (`CONTRIBUTING.md` → *ADR + CONTEXT
discipline*):

- **`CONTEXT.md`** is the canonical glossary — Stage, Runner, Capability, Provider,
  Provenance, Evidence, Quarantine — with banned synonyms listed explicitly. New
  concepts require a glossary entry.
- **[`docs/adr/`](adr/index.md)** holds the numbered, dated architecture decisions,
  each with the alternatives considered and the consequences accepted. Surprising
  trade-offs must land as an ADR, not as tribal knowledge.
- **Overclaiming is a defect.** `CONTRIBUTING.md` requires that a docstring or ADR
  which overclaims relative to what the code does be corrected as part of the change.
- **The seams are documented as seams** — [writing an adapter](writing-an-adapter.md),
  the [conformance checklist](CONFORMANCE.md), the
  [ProviderProfile/device seam](provider-profile.md) — so a new maintainer can extend
  the framework without reverse-engineering it.
- **The work itself is version-controlled**, not in a private tracker: `issues/`
  carries the issue records, the definition-of-done tiers, and the roadmaps.
- **Provenance of the code is recorded**:
  [ADR-0002](adr/0002-clean-monorepo-mined-from-private-repos.md) records that this
  monorepo was mined from the maintainer's own earlier implementations into a clean,
  single-author history, and every ported asset names its origin and the terms of its
  port in its own module docstring and in the ADR that adopted it.

This is the real continuity mitigation. A successor maintainer inherits the
decisions and the vocabulary, not just the source.

## 4. Support-lifetime statement — what you can rely on, and for how long

Read this as the *ceiling*, not the aspiration.

| Question | Answer today |
| --- | --- |
| Release line | **Pre-1.0.** The project ships from `main`; version `0.x` per `docs/RELEASING.md` (SemVer, `0.x` minor bumps may break APIs). |
| Supported versions | `main` (HEAD) and the latest release only. **No maintained release branches, no LTS, no backports** (`SECURITY.md` → *Supported versions*). |
| Security fixes for older releases | **None.** Upgrade to the latest release is the only supported remediation path. |
| Support commitment | **Best-effort, no SLA.** There is no commercial support offering and no contractual response time. |
| End-of-life policy | **None published.** No deprecation window is promised for a `0.x` line. |
| Published packages | The `latence-*` distributions build from this repository; automatic PyPI publishing is **deliberately disabled** (a leading `false &&` short-circuits the `if:` condition on the `pypi-publish` job in `.github/workflows/release.yml`). Verify what you install. |

An integrator who needs a longer or contractual lifetime should plan on either
vendoring a pinned version or maintaining a fork — both are cheap here (§2), and
both are the honest answer rather than a promise this project cannot keep.

## 5. Security process — exactly as it exists

### 5.1 Coordinated vulnerability disclosure (CVD)

`SECURITY.md` is the policy of record and it is real, not a template:

- **Private reporting** via GitHub Security Advisories ("Report a vulnerability" on
  the Security tab) or email to `ddickmann81@gmail.com` with subject
  `SECURITY: latence-framework`.
- **Stated response targets** (best-effort, explicitly labelled as such for a
  personally-maintained pre-1.0 project): acknowledgement within **3 business
  days**, triage within **10 business days**, fix severity-dependent.
- **Coordinated disclosure** with a target **90-day** embargo, a public advisory
  when the fix ships, and credit unless the reporter declines it.
- **A documented trust model** so a reporter knows what counts: `SECURITY.md`
  states what is trusted (the operator, the pipeline YAML) and what is not (every
  ingested document), and enumerates what is out of scope by design.
- **A full threat model** with per-boundary mitigations and honestly-enumerated
  residual risks: [`docs/THREAT-MODEL.md`](THREAT-MODEL.md).

These targets are one person's best effort. They are not an SLA and `SECURITY.md`
says so.

### 5.2 Patch & release policy

- Releases are cut by pushing a `v*` tag; `.github/workflows/release.yml` builds a
  wheel + sdist for **every** package under `packages/` and attaches them to a
  GitHub Release with auto-generated notes. The package list is derived by iterating
  the directory, so a new package cannot be silently omitted.
- **The tagged commit must already be green.** The release build has no test gate of
  its own by design; `docs/RELEASING.md` → *Before you tag* requires
  `bash scripts/verify-local.sh` (secret-scan · ruff · `mypy --strict` · pytest ·
  provider conformance · stack validate · bake-off) plus the heavy lanes in
  `.github/workflows/ci-heavy.yml`.
- **All publishable packages share one version** and are released together, so an
  adopter never has to reason about a matrix of inter-package versions.
- **Changes are recorded** in `CHANGELOG.md` (Keep-a-Changelog style,
  `[Unreleased]` → `[X.Y.Z]` at release time).
- **Fix delivery for a vulnerability = a new release from `main`.** There is no
  backport lane (§4).

### 5.3 SBOM posture

- **A CycloneDX 1.6 SBOM is generated on demand** by `scripts/generate-sbom.sh`,
  derived from the committed `uv.lock` via `uv export --frozen` → `cyclonedx-py`.
  Because it is derived from the lockfile, it always matches what `uv sync`
  installs; it is never hand-maintained. See [`docs/SBOM.md`](SBOM.md).
- **It is a release-time artifact, not a merge gate** — deliberately not wired into
  `.github/workflows/ci.yml` or `scripts/verify-local.sh`, and `docs/SBOM.md` says so.
- **Attached to every release automatically.** The `release-integrity` job in
  `.github/workflows/release.yml` runs `scripts/release-integrity.sh`, which emits a
  per-release CycloneDX SBOM (`sbom.cdx.json`, stamped with the release version) into
  the uploaded asset set. It is no longer a manual step.

### 5.4 Release-signing posture — **signed, keyless Sigstore**

- **Release artifacts are signed.** `scripts/release-integrity.sh` assembles the release
  asset set — the distributions, `sbom.cdx.json`, `SHA256SUMS`, and a VERIFY.md instruction sheet — and
  signs `SHA256SUMS` with **keyless Sigstore** (sigstore-python), producing
  `SHA256SUMS.sigstore.json`. Because `SHA256SUMS` covers every other asset by digest,
  that one signature transitively authenticates the whole release.
- **Keyless means there is no private key to leak.** The signing identity is the GitHub
  Actions workflow itself, certified by Fulcio from the job's OIDC token and logged in
  Rekor. There is no long-lived key to store or rotate. A signature can therefore only be
  produced inside the release workflow; a local dry run uses `--no-sign` and produces
  everything except the bundle.
- **Signing cannot be skipped by accident.** `--sign`/`--no-sign` is required and explicit,
  with no default: a missing `id-token: write` permission fails the job loudly rather than
  quietly shipping an unverifiable release.
- **Adopters verify in three commands.** A generated VERIFY.md ships inside the asset set with
  copy-pasteable instructions: verify the Sigstore bundle over `SHA256SUMS`, run
  `sha256sum -c SHA256SUMS`, then diff the asset directory against the manifest — because
  `sha256sum -c` checks only the files listed and would not notice an asset *added* to the
  published release.
- **PyPI publication is not enabled, and the decision is open.**
  [ADR-0062](adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
  closed it on the strength of the noncommercial licence — `pip install` is overwhelmingly a
  corporate action and PolyForm did not licence commercial use. [ADR-0063](adr/0063-relicence-to-apache-2-0.md)
  removed that licence, so the rationale is superseded and the decision is reopened pending the
  maintainer's call. Today the signed GitHub Release remains the only channel.
  The `pypi-publish` job is written against **PyPI Trusted Publishing (OIDC)** — no long-lived
  API token is ever stored — and stays disabled behind a `false &&` short-circuit on its `if:`
  condition plus a protected `pypi` GitHub Environment requiring a human reviewer, kept only so
  a reversal (which would amend the ADR) stays cheap. What that means for you as an adopter:
  the artifact you install comes from a Release URL or a clone, and it is signed.
- **Verification by rebuild still works.** The lockfile is committed and the build is
  reproducible from source with `uv build --package <name>`; an adopter with a high
  assurance bar can still build from a tagged commit rather than trust any published
  artifact.

### 5.5 Supply-chain controls (implemented)

| Control | Where |
| --- | --- |
| Pinned dependency resolution | committed `uv.lock`; `uv export --frozen` is hash-pinned |
| Weekly dependency updates | `.github/dependabot.yml` — `pip` + `github-actions`, minor/patch grouped, majors reviewed alone |
| GitHub Actions pinned to commit SHAs | **every** external `uses:` in `.github/workflows/*.yml` and `.github/actions/*/action.yml` (e.g. `actions/checkout@93cb6ef…`), each with a `# vX.Y.Z` comment — a moved tag cannot change what runs. Enforced, not asserted: `packages/latence-core/tests/test_continuity_statement.py` fails the build on any tag-pinned action |
| Least-privilege workflow tokens | `permissions: contents: read` at workflow top level; jobs elevate only what they need |
| SAST | `.github/workflows/codeql.yml` (CodeQL over the Python sources) |
| Supply-chain posture scoring | `.github/workflows/scorecard.yml` (OpenSSF Scorecard, default branch + weekly, never on fork PRs) |
| Secret-scanning gate | `scripts/secret_scan_gate.py` with an audited `.secrets.baseline`, run in CI **and** in `scripts/verify-local.sh`; set-based so it cannot go false-green on baseline churn |
| Secret hygiene in artifacts | run manifests scrub secret-looking config values to `***REDACTED***` before writing (`SECURITY.md` → *Built-in controls*) |
| Licence audit of dependencies | `THIRD-PARTY-LICENSES.md`, [`docs/DEPENDENCY-AUDIT.md`](DEPENDENCY-AUDIT.md) |
| Reproducible local gate | `scripts/verify-local.sh` — the full merge gate without CI |

**Not implemented (deliberately named):** dependency-CVE *gating* (hard-failing a
build on a known-vulnerable dependency) — tracked as Phase-2 hardening in
`SECURITY.md` and `docs/SBOM.md`. The SBOM is the input that makes such a gate
possible; the gate itself does not exist yet.

## 6. CRA readiness (EU Cyber Resilience Act)

*Informational, not legal advice. An adopter placing a product on the EU market is
responsible for its own CRA classification.*

**The dates that matter.** The Cyber Resilience Act — Regulation (EU) 2024/2847 —
entered into force on 10 December 2024. Its **vulnerability- and incident-reporting
obligations (Article 14) apply from 11 September 2026**; the remainder of the
Regulation applies from 11 December 2027.

**How this project sits relative to the CRA.**

- **This classification moved twice and is back where it started.** It rested on the
  project being FOSS released by an individual with "no monetisation attached to it". That
  premise ended on 2026-07-29 when the project relicensed to PolyForm Noncommercial and
  began offering commercial licences, and was restored on 2026-08-15 by
  [ADR-0063](adr/0063-relicence-to-apache-2-0.md): the licence is Apache-2.0, there is no
  commercial licence to sell, and nothing is monetised. Recorded rather than left implicit,
  because the intervening state is in the git history and a reader will find it.
- **Apache-2.0 is an OSI-approved open-source licence**, so the CRA's FOSS-specific
  carve-outs are available to this repository on the same footing as before the PolyForm
  interlude. Under PolyForm they were not, because it restricts the field of use.
- What has **not** changed, and still bears on the analysis: there is no legal person
  behind the project (§1), no hosted service, no phone-home, and no paid support
  contract or contractual response time (§4, §5.1). Charging for a commercial licence
  is monetisation of the *licence*, not the placing of a supported product on the
  market, and no commercial agreement has been executed at the time of writing.
- The CRA's lighter **"open-source software steward"** regime applies to a *legal
  person* providing sustained support for FOSS intended for commercial activities.
  This project has no such legal person behind it (§1), so it does not present itself
  as a steward either.
- **Status: open, and deliberately not asserted.** Whether the *manufacturer*
  obligations are engaged once a commercial licence is actually sold is a question this
  page does not answer, because answering it correctly requires the legal advice this
  section disclaims. An evaluator should treat the CRA position as **unresolved**, not
  as cleared. If a commercial agreement is executed, a sponsoring entity is formed, or
  a hosted offering appears, the classification must be settled with counsel **before**
  that ships, not after.

**What the integrator actually gets.** If you integrate this framework into a
product you place on the EU market, *you* are the manufacturer of that product and
you carry the CRA obligations for it, including for the third-party components you
incorporate. Against that, this repository already supplies most of the evidence
you will be asked for:

| CRA-shaped need (Annex I / Article 13–14 territory) | What this repository gives you |
| --- | --- |
| Component inventory / SBOM | `scripts/generate-sbom.sh` → CycloneDX 1.6, derived from `uv.lock` ([`docs/SBOM.md`](SBOM.md)) |
| A vulnerability-reporting channel with stated response times | `SECURITY.md` (§5.1) |
| Coordinated-disclosure policy | `SECURITY.md` — private advisories, 90-day embargo target |
| Documented security risk assessment | [`docs/THREAT-MODEL.md`](THREAT-MODEL.md) — trust boundaries, mitigations, residual risks |
| Secure-by-default configuration | scheme allowlist denying `http`/`ftp`/`sftp` by default, two Screening checkpoints, universal redaction floor ([ADR-0044](adr/0044-universal-redaction-floor-and-optional-mask-before-extract.md)) |
| Integrity & authenticity of the delivered artifact | keyless Sigstore signature over `SHA256SUMS` + the VERIFY.md instruction sheet, produced by `scripts/release-integrity.sh` (§5.4) |
| Evidence that fixes are tested | `scripts/verify-local.sh`; the profound-testing gate record in `docs/evidence/G1-RESULTS.md` |
| Ability to remediate independently of the supplier | Apache-2.0 §2's irrevocable right to prepare and distribute derivative works + reproducible local gate (§2) — for **any** use, commercial included, with no agreement and no dependence on the maintainer |

**What it does not give you**, and you must cover yourself: automated dependency-CVE
gating (§5.5), a supported-version lifetime with backports (§4), and any contractual
response time (§5.1). Plan for a vendored or forked copy if your CRA evidence pack
needs those.

## 7. Verification appendix

Every load-bearing claim above, with the artifact that proves it. If a row's
artifact is missing, the regression test
`packages/latence-core/tests/test_continuity_statement.py` fails and the merge gate
goes red — so this page cannot quietly become aspirational.

| Claim | Artifact |
| --- | --- |
| Apache-2.0, personally owned | `LICENSE`, `NOTICE`, [`COMMERCIAL.md`](https://github.com/ddickmann/latence/blob/main/COMMERCIAL.md), [ADR-0001](adr/0001-open-source-core-owned-personally.md), [ADR-0063](adr/0063-relicence-to-apache-2-0.md) |
| Single maintainer, every path | `.github/CODEOWNERS` |
| CLA required on every contribution | `CONTRIBUTING.md` → *Before you write code: the Contributor Licence Agreement* |
| Docs-as-handover discipline | `CONTEXT.md`, [`docs/adr/`](adr/index.md), `CONTRIBUTING.md` → *ADR + CONTEXT discipline* |
| Permissive-only defaults, weights ≠ code | `THIRD-PARTY-LICENSES.md`, [ADR-0012](adr/0012-permissive-only-reference-providers-gemma-opt-in.md) |
| Supported versions = `main` + latest only | `SECURITY.md` → *Supported versions* |
| CVD channel + response targets | `SECURITY.md` → *Reporting a vulnerability* |
| Threat model with residual risks | [`docs/THREAT-MODEL.md`](THREAT-MODEL.md) |
| Release mechanics | `.github/workflows/release.yml`, `docs/RELEASING.md`, `CHANGELOG.md` |
| PyPI auto-publish disabled | the `false &&` short-circuit in the `if:` condition of `pypi-publish` in `.github/workflows/release.yml` |
| Signed release chain, keyless Sigstore | `scripts/release-integrity.sh` (signs `SHA256SUMS`, emits the bundle + VERIFY.md); the `release-integrity` job in `.github/workflows/release.yml`, whose publish step refuses a release without the bundle |
| SBOM on demand, release-time | `scripts/generate-sbom.sh`, [`docs/SBOM.md`](SBOM.md) |
| Supply-chain controls | `.github/dependabot.yml`, `.github/workflows/codeql.yml`, `.github/workflows/scorecard.yml`, `scripts/secret_scan_gate.py`, `.secrets.baseline` |
| Reproducible merge gate without CI | `scripts/verify-local.sh` |
| Testing gate record | `docs/evidence/G1-RESULTS.md` |

**Sources for the CRA dates:**
[EC — CRA policy page](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act),
[EC — summary of the legislative text](https://digital-strategy.ec.europa.eu/en/policies/cra-summary).
