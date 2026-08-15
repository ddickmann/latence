# Relicence to Apache-2.0 — the licence the intended adopter is allowed to install

**Status: accepted (2026-08-15).** The project's outbound licence is the **Apache License
2.0** (SPDX `Apache-2.0`), replacing PolyForm Noncommercial 1.0.0. **Supersedes**
[ADR-0055](0055-relicence-to-polyform-noncommercial.md) in its entirety, and restores the
publication licence originally recorded in
[ADR-0001](0001-open-source-core-owned-personally.md). It also **reopens the rationale of**
[ADR-0062](0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)
(no PyPI), whose argument rested on this licence and does not survive it — see *What this
reopens* below. No API, contract, schema version, default or artifact changes.

ADR-0055's body is not rewritten. It is a dated record of a decision that was really made,
and the reasoning it contains is the reasoning this ADR argues against; erasing it would
leave no trace that the licence ever moved, and no way to check whether the argument for
moving back is better than the argument for moving away. ADRs 0001, 0012 and 0023 likewise
stay as they are.

## Context

ADR-0055 changed the outbound licence to PolyForm Noncommercial 1.0.0 on 2026-07-29, to
avoid giving away the commercial lane for free before the project had shipped anything.
The reasoning was internally sound. What it did not weigh is **who the framework is built
for**.

The Latence Framework is enterprise data infrastructure. Its whole premise — the messy →
AI-ready pipeline, the Quality Report, the provenance and redaction guarantees, the
continuity statement, the CRA-readiness posture, the SBOM and signed releases — is aimed at
an organisation with a procurement process. Every one of those artifacts exists to survive
someone else's review. And that review begins, in practice, with an automated licence
check:

- **Corporate SCA scanners hard-block the identifier.** `PolyForm-Noncommercial-1.0.0` sits
  on the standard deny-list in Black Duck, Snyk, FOSSA, Mend and every internal equivalent,
  alongside the AGPL family — not because someone assessed this project, but because the
  identifier is categorically non-permissive and non-OSI. The block lands before a human
  reads the README.
- **"Ask us for a commercial licence" is not a step an enterprise takes for an unproven
  tool.** It requires legal review, a supplier onboarding, a single-person bus factor
  accepted in writing, and a budget line — for a framework the evaluator has not yet run.
  The realistic outcome is not a negotiation; it is that the evaluation stops.
- **The evaluation itself was unlicensed.** A for-profit engineer piloting the framework to
  see whether it is worth proposing is doing so for a commercial purpose. PolyForm made the
  single action the project most needs — a corporate try-out — a licence violation, and
  ADR-0062 then acknowledged the same problem at distribution level and responded by
  declining to publish at all.

So the noncommercial licence did not defend a commercial lane; it foreclosed the adoption
that any commercial lane would have to grow out of. ADR-0055 priced an option that only has
value if adoption happens first, and paid for it with the adoption.

**The switch is cheap right now, and expensive very soon.** The same never-published record
that made ADR-0055's relicence clean is still intact and is what makes this one clean:
the repository has been private since creation, nothing is on PyPI (ADR-0062), no release
was cut, no version tag exists, every commit is authored by the sole copyright holder, and
**no CLA has ever been signed**, so there is no contributor whose consent would be needed.
The moment a signed release goes out, that stops being true: recipients hold a grant, forks
exist, and a relicence becomes a negotiation instead of a commit.

## Decision

### 1. The outbound licence is Apache-2.0, for the whole repository

`LICENSE` is the canonical Apache-2.0 text and **nothing else** — no preamble, no
provenance note, no pointers. Scanners identify this file by its bytes; the previous model
could afford BEGIN/END markers around a licence body because it was already outside the
scanners' happy path, and this one cannot. Everything the project says *about* its licence
lives in `NOTICE`, which Apache-2.0 §4(d) already propagates to every redistributor.

### 2. Apache-2.0 rather than MIT, for the patent grant

MIT and BSD would clear the same procurement scanner. Apache-2.0 is chosen for what
enterprise legal actually reads past the identifier:

- **§3 grants an express patent licence** from every contributor, for the contributions
  they made. MIT grants none — its copyright grant is sometimes argued to imply one, which
  is exactly the ambiguity a reviewer must escalate rather than wave through.
- **§3's defensive termination** ends that patent licence for anyone who sues alleging the
  software infringes. This protects the project and, more to the point, reassures an
  adopter that the *other* users of the framework carry the same constraint.
- **§4(b) requires modified files to be marked**, and **§4(d) propagates `NOTICE`** — so
  the attribution and the provenance record travel with a fork rather than depending on
  goodwill.
- It is the licence enterprise legal has already approved a thousand times. Familiarity is
  a real feature of a licence.

### 3. The retained Apache-2.0 subset is dissolved, not widened

ADR-0055 §4 kept a file-level Apache-2.0 grant on seven query-time modules in
`latence-retrieval`, pinned by test and documented in a `LICENSE` section. The concept has
no referent once the whole repository is Apache-2.0: there is nothing to be an exception
*to*. The section, the pinned set and the three gates that enforced it are removed, and the
per-file headers lose their "deliberate, bounded exception" framing.

What replaces them is **stronger**, not weaker: every source file that declares an SPDX
identifier must declare `Apache-2.0`, checked over a full-tree walk rather than against a
list. The old gate permitted a bounded set of files to disagree with the packages'
metadata; the new one permits none, and a new file cannot escape it by not being
enumerated.

### 4. Inbound provenance survives the change untouched

`pack.py`'s record — the QKP packer derives from the maintainer's own `colsearch` project
and was relicensed Apache-2.0 for that use — is about **where code came from**, not what
may be done with it. It was true under Apache-2.0, stayed true under PolyForm, and stays
true now. The two claims are now spelled with the same identifier, which makes them easier
to conflate, so the file keeps its explicit `INBOUND` / `OUTBOUND` labels and the gate that
requires them.

### 5. The relicence provenance record is now a permanent fixture of `NOTICE`

Three licence states in three weeks is a fact a future reader, acquirer or lawyer will find
in `git log` whether or not it is written down. `NOTICE` records all of it as checkable
facts: the dates, both directions, the never-published evidence, and a pointer here.

## What this reopens (and does not decide)

### ADR-0062 (no PyPI): rationale superseded, decision reopened

**ADR-0062 rests entirely on the licence this ADR removes**, and both of its load-bearing
arguments have evaporated:

- §2, "`pip install` is overwhelmingly a corporate action and PolyForm does not licence
  commercial use, so the majority consumer would be unlicensed" — under Apache-2.0 every
  one of those consumers is licensed. The objection is not weakened; it is void.
- §2's second limb, "corporate SCA scanners hard-block the `PolyForm-Noncommercial-1.0.0`
  identifier" — they do not block `Apache-2.0`. That was the argument's whole force.
- §3, "the retained-Apache carve-out is invisible in package metadata" — there is no
  carve-out any more, and the metadata now says `Apache-2.0` for the whole distribution.
- §5's own revisit condition was "reopen if the `pip install` audience changes". It has.

ADR-0062 is therefore marked **rationale superseded / decision reopened** in its own header
and in the index. It is *not* silently reversed here: whether to publish to PyPI is a
distribution decision with its own consequences (name squatting, release cadence, support
expectations, an irreversible first upload) that this ADR has not weighed and is not
entitled to decide as a side effect of a licence change. **The `pypi-publish` job in
`.github/workflows/release.yml` stays disabled behind its `false &&`**, and
`docs/RELEASING.md` continues to describe the current state, pending the copyright holder's
call.

### `CLA.md`: no longer required for the reason it was written

`CLA.md` exists to secure the right to relicense contributions and grant them commercially
— a right the PolyForm model genuinely needed, because inbound-equals-outbound would have
handed the maintainer contributions on noncommercial terms he could not include in a
commercial licence. Apache-2.0 §5 makes every contribution arrive under the same licence as
the work, so that gap does not exist and a **DCO would now suffice**.

This ADR does not retire the CLA. It is the copyright holder's decision, it has a real
(if smaller) residual benefit — a recorded, explicit grant and the ability to relicense
later without tracing contributors — and `CONTRIBUTING.md`, `CLA.md` and the gate that
checks them all currently agree with each other. Changing that is a separate decision with
its own ADR. Recorded here so it is not lost.

## Consequences (accepted)

- **The commercial lane is given away, permanently and to everyone.** This is ADR-0055's
  central objection and it is correct on its own terms: nobody will ever need to buy a
  licence for this code. What is bought with it is the possibility of anyone using it. If a
  commercial product is ever wanted, it will have to be built on service, hosting, support
  or proprietary extensions — not on withholding the framework.
- **It is irreversible in practice.** The next relicence will not be clean: once a release
  is published, every recipient holds a perpetual, irrevocable Apache-2.0 grant to that
  version, and a future restriction can only ever apply to future versions of a codebase
  that anyone may already have forked. This ADR is the last cheap licence decision the
  project gets.
- **Anyone may fork it commercially, including a competitor.** Apache-2.0 permits closed,
  proprietary redistribution subject only to §4. The mitigations are the ordinary
  open-source ones — maintainership, release cadence, the trademark not being licensed
  (§6) — and not the licence.
- **`COMMERCIAL.md` becomes a correction page rather than a route.** It keeps its name and
  its inbound links so that readers arriving from ADR-0055, from `NOTICE`, or from a
  bookmark are told the gate is gone instead of finding a 404 or a stale demand for an
  agreement.
- **The project may again be described as open source.** Apache-2.0 is OSI-approved, so the
  procurement checklists that ADR-0055 accepted failing now pass, and the deprecated
  `License :: OSI Approved ::` trove classifier becomes *accurate* — which is precisely why
  the publish gate keeps asserting its **absence**: PEP 639 forbids pairing it with an SPDX
  `License-Expression`, and "restoring" it would break the upload it appears to fix.
- **`docs/CONTINUITY.md`'s fork posture improves.** A fork is now durable without the
  maintainer's cooperation for any purpose, commercial included — which removes the single
  largest continuity risk ADR-0055 accepted (a bus factor of 1 gating the commercial lane).
- **Still not legal advice.** Apache-2.0 needs no review; the project's statements around it
  do, and `RELEASE-GATES.md` Gate 3 stays open. `COMMERCIAL.md` § *Legal review* lists what
  remains.

## Alternatives considered

- **Stay on PolyForm Noncommercial.** Rejected: it blocks the adopter the framework is
  built for, at the exact moment (pre-adoption, pre-revenue) when access matters more than
  an unexercised option. ADR-0062 is the evidence — the licence had already cost the
  project its distribution channel before it cost it a single user.
- **MIT or BSD-3-Clause.** Rejected: they clear the scanner just as well but grant no
  patent licence and carry no defensive-termination clause. For infrastructure software an
  enterprise runs over its own corpus, the express §3 grant is the part legal reads after
  the identifier.
- **Dual-licence Apache-2.0 + commercial.** Rejected as incoherent: once the code is
  available under Apache-2.0 there is nothing a commercial licence could sell. A dual
  licence only functions with a copyleft public side (AGPL), which was already rejected in
  ADR-0055 for reaching into the adopter's own pipeline code — and which enterprises block
  as reliably as PolyForm.
- **Business Source License with a change date.** Rejected for the same reason ADR-0055
  rejected it, plus one more: BUSL is on the same SCA deny-lists as PolyForm, so it would
  not solve the problem that motivated this ADR.
- **Keep PolyForm and add a broad free-evaluation grant.** Rejected: it requires drafting a
  bespoke exception this project cannot do without counsel, produces a non-standard
  identifier no scanner recognises, and leaves procurement facing a licence they must read
  rather than one they can match.
- **Wait until adoption proves the commercial lane is worth keeping.** Rejected on timing:
  the option is only cheap to exercise while nothing has been published. Waiting means the
  choice is made by default, and made expensive.

## Where the record lives

| Question | Artifact |
|---|---|
| What are the terms? | `LICENSE` — canonical Apache-2.0, digest-pinned, nothing added |
| What may I do commercially? | `COMMERCIAL.md` — short answer: anything |
| Why did the licence change twice, and is that clean? | `NOTICE` § *PROVENANCE* |
| What was the superseded reasoning? | [ADR-0055](0055-relicence-to-polyform-noncommercial.md) |
| Is the no-PyPI decision still valid? | [ADR-0062](0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md) — rationale superseded, decision reopened |
| What binds me regardless of this licence? | `THIRD-PARTY-LICENSES.md` |
| What does it mean for a fork or for continuity? | [`docs/CONTINUITY.md`](../CONTINUITY.md) §2 |
| What stops it drifting back? | `packages/latence-core/tests/test_publish_readiness.py` |
