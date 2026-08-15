# Distribution is signed GitHub Releases only — no PyPI under PolyForm Noncommercial

!!! warning "Rationale superseded / decision REOPENED (2026-08-15) by [ADR-0063](0063-relicence-to-apache-2-0.md)"
    **The reasoning below no longer holds, and the decision is open.** This ADR argued
    against PyPI because `pip install` is a corporate action, PolyForm Noncommercial does
    not licence commercial use, and corporate SCA scanners hard-block the
    `PolyForm-Noncommercial-1.0.0` identifier (§2) — and because the retained-Apache
    carve-out was invisible in package metadata (§3). ADR-0063 relicensed the project to
    **Apache-2.0**: every `pip install` consumer is now licensed, no scanner blocks the
    identifier, and there is no carve-out. §5's own revisit condition ("reopen if the
    `pip install` audience changes") has been met.

    **This is not a reversal.** Whether to publish to PyPI is a distribution decision —
    name claims, release cadence, support expectations, an irreversible first upload —
    that ADR-0063 deliberately did not weigh. The `pypi-publish` job in
    `.github/workflows/release.yml` therefore **stays disabled** behind its `false &&`,
    exactly as §4 describes, pending the copyright holder's call. Reversing it needs a new
    ADR, not a flag flip.

**Status: rationale superseded, decision reopened (2026-08-15); accepted 2026-08-15.** The
project's published artifacts are, for now, distributed **only** as
signed GitHub Releases (wheels + sdists + CycloneDX SBOM + `SHA256SUMS` + its Sigstore bundle).
Nothing is uploaded to PyPI. Follows from [ADR-0055](0055-relicence-to-polyform-noncommercial.md)
(the relicence to PolyForm Noncommercial 1.0.0) and settles decision **D1** of
`issues/wayfinder-enterprise-serving/tickets/05-decisions.md`. No contract, schema or API change.

## Context

`.github/workflows/release.yml` already builds every distribution, generates the per-release SBOM,
checksums the whole asset set and signs `SHA256SUMS` with keyless Sigstore, then attaches the
signed set to the GitHub Release for the tag. It also carries a fully-formed `pypi-publish` job
using PyPI Trusted Publishing (OIDC, no stored token) behind a protected `pypi` environment —
scaffolded, and short-circuited by a leading `false &&` in its `if:` so a tag push can never
auto-publish.

That scaffolding was written under Apache-2.0. ADR-0055 changed the outbound licence to PolyForm
Noncommercial 1.0.0 and did not revisit the distribution channel, which left the question open in
two places at once: `docs/RELEASING.md` § *Open decisions* discussed the namespace and whether to
auto-publish but never raised the licence question, and the enable checklist in `release.yml` read
as *pending the maintainer's go-ahead*. An unanswered question in release tooling is the kind that
gets answered by accident, at the tag.

**PyPI's Terms of Use do not require an OSI-approved licence**, and PolyForm Noncommercial is
royalty-free and on the SPDX License List (PEP 639 metadata accepts
`PolyForm-Noncommercial-1.0.0`). So an upload is *permitted*. This ADR records a **choice**, not a
prohibition — that distinction matters, because a future maintainer who finds this decision needs
to know that reversing it is a policy call and not a licensing fight.

None of the five reference repositories surveyed in
`issues/wayfinder-enterprise-serving/tickets/05-part-a-comparison.md` is a clean precedent (all
five are permissively licensed), but the *shape* is established at the top of the field: qdrant
ships binaries and signed images with no PyPI; `NousResearch/hermes-agent` has no PyPI workflow at
all — a curl installer plus Docker — at 230k stars. "Signed GitHub Releases, no PyPI" is a normal
distribution posture, not a consolation prize.

## Decision

### 1. The distribution channel is the signed GitHub Release, and only that

An adopter installs from a Release URL (`pip install <URL-of-the-wheel>`) or from a clone. Every
release carries its own verification evidence — SBOM, checksums over every asset, one Sigstore
signature over the checksum file, and the copy-pasteable verification procedure in `VERIFY.md`.
That is a *stronger* provenance story than an unattested PyPI upload, so nothing about integrity
is being traded away here.

### 2. The reason is who `pip install` reaches, not what PyPI allows

`pip install` is overwhelmingly a corporate action. PolyForm Noncommercial does not licence
commercial use, and corporate SCA scanners hard-block the `PolyForm-Noncommercial-1.0.0` SPDX
identifier as a matter of policy. Publishing would therefore place the artifact in a channel whose
**majority consumer is, by the licence's own terms, unlicensed** — which buys support load and
policy-violation reports rather than adoption. The noncommercial users the licence is actually for
(researchers, students, teachers, noncommercial organisations) are the population least obstructed
by a Release URL or a clone.

### 3. The retained Apache-2.0 subset does not carve out an exception

ADR-0055 §4 keeps a file-level Apache-2.0 grant on seven query-time modules in
`latence-retrieval`, enumerated in `LICENSE`. That carve-out is real at the *source* level and
**invisible at the distribution level**: a scanner reads package metadata, and the package metadata
says `PolyForm-Noncommercial-1.0.0` for the whole distribution. Publishing `latence-retrieval` to
PyPI as a "permissive subset" would therefore be read by the exact tooling it was meant to reassure
as a PolyForm package, and would additionally split the project across two channels for no gain.
The seven modules stay reusable the way they already are: by reading `LICENSE` and taking the
files.

### 4. The scaffolding stays; only the framing closes

`release.yml`'s `pypi-publish` job is **not deleted**. It stays disabled by the same `false &&`,
with the same `pypi` environment and the same OIDC wiring, and its checklist now records this
decision (and points here) instead of reading as a pending to-do. A reversal must remain a
one-line edit plus a name claim, not an archaeology exercise. `docs/RELEASING.md` states the same
thing in prose so the two cannot disagree.

### 5. Revisit condition

Reopen this decision if — and only if — **the `pip install` audience changes**: a dual-licence
(an OSI-approved option alongside the commercial one) or a commercial SKU that gives corporate
consumers a licence they can actually satisfy. At that point the objection in §2 dissolves, and
enabling the job is cheap by construction (§4). A revisit amends this ADR; it does not happen by
flipping the flag.

## Consequences (accepted)

- **Install friction for the honest noncommercial user.** `pip install latence-core` does not work
  and will not resolve to anything of ours. `README.md` § *Install* already leads with the
  not-yet-published caveat and the from-clone path; that caveat is now permanent policy rather
  than a temporary state.
- **No PyPI-side provenance.** PEP 740 attestations, which the scaffolded job would have produced
  from the Trusted Publishing identity, are not available to adopters. The Sigstore bundle on the
  Release is the whole provenance story, and `docs/RELEASING.md` § *Release integrity* is where an
  adopter learns to check it.
- **Reduced discoverability.** The project will not appear in PyPI search, in dependency-graph
  suggestions, or in `pip index`. Discovery is the docs site, the repository, and whatever the
  benchmark results earn.
- **The `latence-*` names stay unclaimed on PyPI.** `latence-core` was verified free; `latence`
  itself already exists (0.2.0, MIT) under the maintainer's own separate `latenceainew` account
  and is unrelated to this workspace. An unclaimed name can be taken by a third party, and a
  typo-squat under a name this project's docs mention is a real (if low-probability) risk. It is
  accepted rather than mitigated, because the only mitigation — uploading a placeholder
  distribution to reserve the name — is the very act this ADR declines, and a stub upload would
  still carry the PolyForm metadata into the channel §2 is about.
- **Nothing changes about how a release is cut.** The tag flow, the gates, the SBOM and the
  signing identity are untouched; the tag simply has one fewer job it could ever reach.

## Alternatives considered

- **Publish everything to PyPI** (flip the job on). Rejected per §2: permitted, but it targets the
  population the licence excludes.
- **Publish only `latence-retrieval`**, on the strength of its retained Apache-2.0 modules.
  Rejected per §3: the retained grant is invisible in package metadata, so the move does not buy
  the reassurance it appears to buy, and it fragments the distribution story.
- **Upload stub distributions to reserve the `latence-*` names without publishing real code.**
  Rejected: it puts PolyForm-metadata packages on PyPI anyway — the same channel problem in
  miniature — and a name reserved by an empty package is a name an adopter can install by
  accident.
- **Delete the `pypi-publish` job outright** now that the answer is no. Rejected per §4: the
  decision is reversible by design, and deleting the OIDC + protected-environment wiring would
  make a future reversal expensive enough to bias it.

## Where the record lives

| Question | Artifact |
| --- | --- |
| How is a release produced, signed and verified? | [`docs/RELEASING.md`](../RELEASING.md) |
| Why is the outbound licence PolyForm Noncommercial? | [ADR-0055](0055-relicence-to-polyform-noncommercial.md) |
| Which files keep an Apache-2.0 grant, and what may I do with them? | `LICENSE` § *SCOPE: THE RETAINED APACHE-2.0 MODULES* |
| How do I get a commercial licence? | `COMMERCIAL.md` |
| What would enabling PyPI take? | `.github/workflows/release.yml`, the `pypi-publish` job header |
