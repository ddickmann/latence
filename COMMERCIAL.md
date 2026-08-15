# Commercial use

**Commercial use of the Latence Framework is granted by its licence. No separate
agreement is required, and there is nothing to buy from the copyright holder.**

The framework is licensed under the **Apache License 2.0** (SPDX: `Apache-2.0`),
reproduced verbatim in [`LICENSE`](./LICENSE). That licence grants you — any
person or company, for any purpose including a commercial one — the right to
use, modify, distribute, sublicense and sell the software, together with a
patent licence from every contributor, subject only to the attribution
conditions in its §4.

This document is a plain-language pointer. **`LICENSE` governs.** Where this
document and `LICENSE` differ, `LICENSE` wins.

> **Why this page still exists.** Between 2026-07-29 and 2026-08-15 the project
> carried the PolyForm Noncommercial 1.0.0 licence, and this page was the route to
> a commercial licence. That model is gone
> ([ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md), which supersedes
> [ADR-0055](./docs/adr/0055-relicence-to-polyform-noncommercial.md)). The page is
> kept, at its old name, so that every inbound link — from the README, from
> `NOTICE`, from the superseded ADRs, from anything anyone bookmarked — lands on
> a correction instead of a 404 or, worse, on a stale page telling a reader to
> ask permission they do not need.

---

## What you may do

Everything Apache-2.0 permits, without asking and without telling anyone:

- Use the framework inside a for-profit company, on customer data, in production.
- Build a product or a paid service on it, and sell that.
- Offer it, or anything built on it, as a hosted or managed service.
- Deliver consulting, integration or systems-integration work with it.
- Embed it in a commercial distribution, an appliance, or an SDK.
- Fork it, modify it, and redistribute your fork — including under a different
  licence, provided you meet §4.

Three obligations, and they are the whole of it (Apache-2.0 §4):

1. **Ship the licence.** Give every recipient a copy of `LICENSE`.
2. **Ship the notice.** The distribution includes a [`NOTICE`](./NOTICE) file;
   redistributors must reproduce its attribution notices. `NOTICE` also carries
   the provenance record for the two relicences.
3. **Mark your changes.** Modified files must carry a prominent notice saying you
   changed them.

Two things worth knowing rather than discovering:

- **The patent grant terminates defensively.** Apache-2.0 §3 grants you a patent
  licence from every contributor, and ends it if you initiate patent litigation
  alleging the software infringes.
- **No trademark rights.** §6 grants none; "Latence" is not licensed to you as a
  name.

## Support, consulting and sponsorship

The licence grants rights, not a relationship. There is no commercial support
contract, no SLA, and no paid tier — see
[`SUPPORT.md`](./SUPPORT.md) for what support actually exists (best-effort,
via GitHub issues) and [`docs/CONTINUITY.md`](./docs/CONTINUITY.md) for the
honest bus-factor and support-lifetime statement an enterprise reader should
read before depending on this.

If you want to talk to the maintainer anyway — about a deployment, about
prioritising something you need, about consulting or sponsorship, or simply to
say you are using it:

**Dennis Dickmann — ddickmann81@gmail.com**

That is an invitation, not a condition. Nothing on this page gates your use of
the software, and no arrangement made by email changes the licence in
`LICENSE`.

## What the licence does *not* reach

Apache-2.0 settles rights in **the Latence Framework's own code** — the
copyright Dennis Dickmann holds. It cannot and does not grant rights in
third-party software, model weights, or datasets, several of which carry their
own commercial-use restrictions. An outbound licence is never a shield against
inbound terms. The ones that matter, all documented in
[`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md):

| Component | Terms | Why it matters commercially |
|---|---|---|
| `easydict` (opt-in `latence-linkpred-ultra[ultra]`) | **LGPL-3.0** | Weak copyleft with real conditions if you ship binaries. See the LGPL section of `THIRD-PARTY-LICENSES.md`. |
| NVIDIA CUDA wheels (`deploy/env/gpu-learned/`, `deploy/env/served-ocr-server/`) | **Proprietary NVIDIA EULAs** | Not open-source. Installing either environment means accepting NVIDIA's terms. |
| `google/embeddinggemma-300m` | **Gemma license** — opt-in, non-default, no Gemma-backed Provider ships here | Permits commercial use but carries a Prohibited Use Policy and a flow-down obligation to every downstream recipient. |
| `opendatalab/OmniDocBench` | **Research-only, non-commercial** | Gated behind `--accept-non-commercial` in the benchmark harness. Never usable in a commercial benchmark. |
| `naver/splade-v3` weights | **CC-BY-NC-SA-4.0** | Gated behind `--accept-noncommercial-license`. The SPLADE *code path* in this repository is Apache-2.0; those *weights* are not. |
| `Babelscape/REDFM` | CC BY-SA 4.0 | Commercial use permitted with attribution; ShareAlike binds adaptations of the corpus. |
| Every other runtime dependency and default model checkpoint | Permissive (Apache-2.0 / MIT / BSD / ISC / PSF / MPL-2.0) | No restriction on commercial use. |

You are responsible for your own compliance with those third-party terms.
`latence env matrix` machine-checks the CUDA disclosure, and the benchmark
harness's licence gate refuses a non-commercial corpus unless you opt in
explicitly — but neither is a substitute for your own review.

## Contributions

To date the project has had **no outside contributions**: every commit in the
repository's history was authored by the copyright holder.

[`CLA.md`](./CLA.md) at the repository root is a Contributor Licence Agreement,
and `CONTRIBUTING.md` makes a recorded signature a precondition of every merge.
It was written for the PolyForm model, where it was load-bearing: under
noncommercial-public/commercial-by-agreement terms, a contribution arriving on
noncommercial terms could not lawfully be included in a commercial licence, and
Section 2 of the CLA was the grant that closed that gap.

Apache-2.0 is inbound-equals-outbound — §5 places every contribution under the
same licence as the work — so that gap no longer exists, and a DCO would now
suffice. **Whether to retire the CLA in favour of a DCO is an open decision for
the copyright holder**, recorded as such in
[ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md); until he takes it,
`CONTRIBUTING.md` is what governs.

`CLA.md` has not been reviewed by a lawyer, and it says so at the top. That
review is tracked as `RELEASE-GATES.md` **Gate 3 — Licence coherence**, and it
is open.

## Legal review

The licensing in this repository was prepared by the copyright holder with
machine assistance. **It is not legal advice, and it has not been reviewed by
counsel.** Apache-2.0 itself needs no review — it is one of the most widely
deployed licences in existence and is reproduced here verbatim and digest-pinned
— but the project's own statements around it do. Before relying on this
commercially, have a lawyer confirm at minimum:

1. the LGPL-3.0 `easydict` interaction, specifically LGPL-3.0 §4 and §6 if
   anything is ever shipped as a binary or a container image;
2. that the two relicences are clean given the repository's never-published
   history (the facts are recorded in [`NOTICE`](./NOTICE) and in
   [ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md));
3. the contributor position above — whether to keep `CLA.md` or move to a DCO,
   and if the CLA is kept, the open points its own
   [Notes for the reviewer](./CLA.md#notes-for-the-reviewer) list.
