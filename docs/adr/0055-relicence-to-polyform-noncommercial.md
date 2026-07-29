# Relicence to PolyForm Noncommercial — free for research, commercial by agreement

**Status: accepted (2026-07-29).** The project's outbound licence is
**PolyForm Noncommercial License 1.0.0** (SPDX `PolyForm-Noncommercial-1.0.0`), replacing
Apache-2.0. **Supersedes** the publication licence recorded in
[ADR-0001](0001-open-source-core-owned-personally.md), and **amends** the two later ADRs that
restate it in passing — [ADR-0012](0012-permissive-only-reference-providers-gemma-opt-in.md)
(the in-core pure-code Providers' `license_code`) and
[ADR-0023](0023-relation-mention-record-and-fused-stage-routing.md) (`relation.llm`'s framework
code). None of those three has its body rewritten: each is a dated record of what was decided
then, and ADR-0001's decision in particular *was* made under Apache-2.0 — that is the provenance
this relicence rests on, not a mistake to erase.

## Context

ADR-0001 chose Apache-2.0 for one specific job: publish the core as pre-existing, personally owned
IP before the Deloitte start date, so it is consumed at work like any third-party OSS tool and
never absorbed into an employer's asset base. That job is unaffected by which permissive-or-not
licence the code carries — what establishes pre-existing IP is authorship and date, not the grant.

What Apache-2.0 *also* did was give away the commercial lane for free, permanently and to everyone,
before the project had shipped anything. The framework is a maintained, single-owner product with
a plausible commercial future; nothing about the pre-employment publication required financing that
future at zero.

The relicence is clean, and that is a checkable claim rather than an opinion: the repository was
private from creation, nothing was ever published to PyPI, no release was cut, and every commit is
authored by the sole copyright holder. **No one holds a prior Apache-2.0 grant, because no one ever
received a copy under one.** The verified record — what was checked, how, and when — is at the top
of `LICENSE` and is deliberately kept *outside* the licence terms.

## Decision

1. **The outbound licence is PolyForm Noncommercial 1.0.0.** Any noncommercial purpose — personal
   study, research, teaching, evaluation, noncommercial organisations — is free and needs nobody's
   permission. Commercial use requires a separate written agreement with the copyright holder.
2. **Commercial terms are by agreement, and are not invented in this repository.** No price list,
   no tiers, no standard contract text. `COMMERCIAL.md` routes the request and says explicitly
   that scope, fee, duration and warranty are settled in writing.
3. **The licence text is reproduced verbatim and pinned by digest.** `LICENSE` embeds the licence
   steward's official plain text between BEGIN/END markers; a test recomputes its SHA-256 on every
   run, so a hand-edited clause fails the build instead of shipping. Everything this project says
   *about* its licence sits outside those markers and cannot alter the terms.
4. **One bounded permissive carve-out is retained, deliberately.** Seven query-time retrieval
   modules in `latence-retrieval` keep a file-level Apache-2.0 grant so the algorithms stay
   reusable by anyone, commercially included. The set is enumerated in `LICENSE` (section
   *"SCOPE: THE RETAINED APACHE-2.0 MODULES"*), stated in each file's header, and machine-pinned
   by `packages/latence-core/tests/test_publish_readiness.py` — which fails if an eighth file
   acquires a non-PolyForm grant or one of the seven loses it. It is a decision with a boundary,
   not a leftover, and it does not extend to the rest of the package those modules import from.
5. **Contributions now need a CLA.** Inbound-equals-outbound is incoherent under this model: it
   would hand the owner contributions on *noncommercial* terms, leaving him without the right to
   license them commercially. `CONTRIBUTING.md` requires a signed CLA on every contribution,
   including one-line ones.

## What this does NOT change

- **The retained Apache-2.0 subset** (decision 4 above). Those seven files are outside the PolyForm
  grant on purpose. `LICENSE` is the authoritative statement of which files and what it means
  downstream; this ADR is the reasoning behind it.
- **Third-party licences.** An outbound licence is not a shield: `easydict` (LGPL-3.0, opt-in
  extra), the NVIDIA CUDA wheels (proprietary EULAs), the Gemma licence (opt-in, non-default) and
  the non-commercial benchmark corpora bind the *user* directly, exactly as they did before.
  A commercial agreement with the copyright holder settles rights in *this* code and nothing else.
  See `THIRD-PARTY-LICENSES.md`.
- **The permissive-only rule for default Providers** (ADR-0012). It governs *inbound* model and
  library licences and is untouched by a change to the *outbound* one. Only its incidental sentence
  about the framework's own code licence is amended.
- **The historical provenance.** ADR-0001's decision, its date, and the fact that publication was
  planned under Apache-2.0 stay on the record verbatim. The never-published history is what makes
  this relicence clean, so erasing it would destroy the evidence the relicence depends on.
- **What the framework does.** No API, contract, schema version, default or artifact changes.

## Consequences (accepted, and already recorded in `docs/CONTINUITY.md` §2)

- **PolyForm Noncommercial is not OSI-approved.** It is on the SPDX License List and is not
  deprecated, so PEP 639 metadata accepts it — but the project may no longer be described as
  open-source, and procurement checklists that require an OSI licence will fail it. There is no
  correct `License :: OSI Approved ::` classifier to use, so none is declared.
- **The patent grant narrows.** PolyForm's *Patent License* grants a patent licence from **the
  licensor only** — not, as Apache-2.0 §3 would, from every contributor — and only for the
  licence's permitted (i.e. noncommercial) purposes. A *Patent Defense* clause ends it immediately
  if you assert a patent claim against the software. Anyone needing a broad, contributor-wide,
  commercially-scoped patent grant must raise it as a term of the commercial agreement.
- **The grant is revocable.** Unlike Apache-2.0, PolyForm's *Violations* section ends all of your
  licences if a violation notified in writing is not cured within 32 days. A fork is durable only
  for as long as it stays inside the permitted purposes.
- **Continuity now depends on the maintainer for the commercial lane.** A *noncommercial* fork
  still needs nobody's cooperation; a commercial licence needs the maintainer to be reachable.
  With a bus factor of 1, that is the single largest continuity risk on the project.
- **"Noncommercial" is a boundary someone will have to interpret.** The licence defines permitted
  purposes positively and this repository adds no definition of its own.
- **It is not legal advice.** The relicence was prepared by the copyright holder with machine
  assistance and has not been reviewed by counsel; `COMMERCIAL.md` § *Legal review* lists what a
  lawyer still has to confirm.

## Alternatives considered

- **Stay Apache-2.0.** Rejected: it permanently forecloses the commercial lane in exchange for a
  benefit — pre-existing-IP provenance — that a noncommercial licence delivers just as well.
- **Dual-licence AGPL-3.0 + commercial.** Rejected: AGPL is OSI-approved and keeps the
  open-source label, but its copyleft reaches the adopter's *own* pipeline code, which for a
  data-processing framework is a heavier imposition on the honest noncommercial user than
  PolyForm's field-of-use restriction — and enterprises block AGPL outright, so the commercial
  conversation never starts.
- **Business Source License (BUSL-1.1).** Rejected: its change-date mechanism converts the code to
  an open licence on a schedule, which is a commitment about the future the owner is not ready to
  make, and its "additional use grant" would need drafting this project cannot do without counsel.
- **Relicence only the pipeline, keep `latence-retrieval` permissive.** Rejected as the *general*
  rule, kept as the bounded exception: the seven algorithm modules are worth giving away, but the
  package's contracts, backends, orchestrator and MCP server are product surface.
- **Rewrite ADR-0001, 0012 and 0023 to say PolyForm.** Rejected. An ADR is a dated record of a
  decision, not a description of the present. Rewriting them would erase the pre-employment
  Apache-2.0 publication decision — the exact fact the relicence's clean-provenance argument rests
  on — and would leave no trace that the licence ever changed.

## Where the record lives

| Question | Artifact |
|---|---|
| What are the terms? | `LICENSE` — verbatim, digest-pinned |
| Why is the relicence clean? | `LICENSE` § *PROVENANCE OF THIS LICENCE* |
| Which files stay Apache-2.0, and what may I do with them? | `LICENSE` § *SCOPE: THE RETAINED APACHE-2.0 MODULES* |
| How do I get a commercial licence? | `COMMERCIAL.md` |
| What binds me regardless of this licence? | `THIRD-PARTY-LICENSES.md` |
| What does it mean for a fork or for continuity? | [`docs/CONTINUITY.md`](../CONTINUITY.md) §2 |
| What stops it drifting back? | `packages/latence-core/tests/test_publish_readiness.py` |
