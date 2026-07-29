# Commercial licensing

The Latence Framework is licensed under the **PolyForm Noncommercial License
1.0.0** (SPDX: `PolyForm-Noncommercial-1.0.0`). Noncommercial use is free.
**Commercial use requires a separate written agreement with the copyright
holder.**

> **Provenance.** This repository carried an Apache-2.0 `LICENSE` internally from
> its first commit until 2026-07-29 and was **never published publicly** — the
> GitHub repository has been private since it was created, no distribution was
> ever published to PyPI, no release was ever cut, and every commit in its
> history was authored by the sole copyright holder, Dennis Dickmann. The
> relicence is therefore clean: no one holds a prior Apache-2.0 grant, because no
> one ever received a copy under one. The verified record — what was checked, how,
> and when — is at the top of [`LICENSE`](./LICENSE).

This document is a plain-language pointer. **`LICENSE` governs.** Where this
document and `LICENSE` differ, `LICENSE` wins.

---

## What is free

Everything the PolyForm Noncommercial License calls a *permitted purpose*. In the
licence's own words, that is:

- **Any noncommercial purpose.**
- **Personal use** — "research, experiment, and testing for the benefit of public
  knowledge, personal study, private entertainment, hobby projects, amateur
  pursuits, or religious observance, without any anticipated commercial
  application".
- **Use by a noncommercial organization** — "any charitable organization,
  educational institution, public research organization, public safety or health
  organization, environmental protection organization, or government institution
  … regardless of the source of funding or obligations resulting from the
  funding".

Within a permitted purpose the licence grants you the right to use, distribute,
and make changes and new works, plus a patent licence, subject to its `Notices`,
`Patent Defense`, and `Violations` sections. Read those sections; they are short.

Two obligations that catch people out:

1. **Carry the notice.** If you give anyone a copy of any part of this software,
   you must also give them these terms (or the URL for them) **and** the
   `Required Notice:` line at the top of `LICENSE`.
2. **Fair use is untouched.** The licence explicitly does not limit rights you
   already have under law.

## What you do not need a licence for at all

Seven modules in `latence-retrieval` — the query-time retrieval primitives — are
**not** under the PolyForm licence. They carry a file-level
`SPDX-License-Identifier: Apache-2.0` grant, deliberately, so you may use, modify
and redistribute those files **commercially** with no agreement here.

That exception is bounded, enumerated and machine-pinned in
[`LICENSE`](./LICENSE), section *"SCOPE: THE RETAINED APACHE-2.0 MODULES"* — that
section is authoritative and this is only a pointer to it. Read it before you rely
on the exception: it names the exact seven files, and it is explicit that the rest
of `latence-retrieval` (which those seven import from) is **not** covered.

## What counts as commercial use

Commercial use is **anything that is not a permitted purpose** under the licence.
The licence defines the permitted purposes positively (above); it does not define
"commercial" separately, and this document does not add a definition.

Cases that are, in practice, outside the permitted purposes and need an
agreement — this list is illustrative, not a definition:

- Using the framework in or for a product or service you sell, license, or
  monetise, including internally within a for-profit company's operations.
- Offering it, or anything built on it, as a hosted or managed service.
- Consulting, integration, or systems-integration work delivered to a paying
  client.
- Embedding it in a commercial distribution, appliance, or SDK.
- Using it to process a customer's data as part of a commercial engagement.

If you are unsure which side of the line you are on, ask before you build.
Asking costs an email.

## How to get a commercial licence

Contact the copyright holder:

**Dennis Dickmann — ddickmann81@gmail.com**

Please include:

- who the licensee is (legal entity name and jurisdiction);
- what you want to do with the framework, concretely;
- which packages/extras you intend to use (they have different third-party
  obligations — see below);
- whether you intend to redistribute, and in what form (source, wheels, binaries,
  container images, an appliance).

**Terms are by agreement.** There is no published price list, no tier structure,
and no standard contract text in this repository. Anything you have read
elsewhere about pricing or terms for this project is not from the copyright
holder. Scope, fee, duration, support, and any warranty are settled in the
written agreement itself.

## What a commercial licence from the copyright holder does *not* cover

A commercial licence covers **the Latence Framework's own code** — the copyright
that Dennis Dickmann holds. It cannot and does not grant rights in third-party
software, model weights, or datasets. Those carry their own terms, and several of
them have their own commercial-use restrictions that survive any agreement made
here. The ones that matter, all documented in
[`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md):

| Component | Terms | Why it matters commercially |
|---|---|---|
| `easydict` (opt-in `latence-linkpred-ultra[ultra]`) | **LGPL-3.0** | Weak copyleft with real conditions if you ship binaries. See the LGPL section of `THIRD-PARTY-LICENSES.md`. |
| NVIDIA CUDA wheels (`deploy/env/gpu-learned/`, `deploy/env/served-ocr-server/`) | **Proprietary NVIDIA EULAs** | Not open-source. Installing either environment means accepting NVIDIA's terms. |
| `google/embeddinggemma-300m` | **Gemma license** — opt-in, non-default, no Gemma-backed Provider ships here | Permits commercial use but carries a Prohibited Use Policy and a flow-down obligation to every downstream recipient. |
| `opendatalab/OmniDocBench` | **Research-only, non-commercial** | Gated behind `--accept-non-commercial` in the benchmark harness. Never usable in a commercial benchmark. |
| `Babelscape/REDFM` | CC BY-SA 4.0 | Commercial use permitted with attribution; ShareAlike binds adaptations of the corpus. |
| Every other runtime dependency and default model checkpoint | Permissive (Apache-2.0 / MIT / BSD / ISC / PSF / MPL-2.0) | No commercial restriction. |

You are responsible for your own compliance with those third-party terms.
`latence env matrix` machine-checks the CUDA disclosure, and the benchmark
harness's licence gate refuses a non-commercial corpus unless you opt in
explicitly — but neither is a substitute for your own review.

## Contributions

The project has **no signed CLA and no DCO process**. To date it has had no
outside contributions: every commit in the repository's history was authored by
the copyright holder.

`CONTRIBUTING.md` currently carries only an inbound-equals-outbound sentence
("by contributing you agree your contributions are licensed under the project's
licence"). **That is not sufficient for this licensing model, and it is not yet
updated for the relicence.** Under a dual arrangement — noncommercial terms to
the public, commercial terms by agreement — inbound-equals-outbound means the
copyright holder would receive a contribution under *noncommercial* terms and
would then have no right to license that contribution commercially. Every
commercial licence granted after such a contribution would be partly
unauthorised.

Until a CLA (or a copyright assignment) is in place, outside contributions
should be declined or held. This is flagged, not solved, here.

## Legal review

The relicence in this repository was prepared by the copyright holder with
machine assistance. **It is not legal advice, and it has not been reviewed by
counsel.** Before relying on it commercially — as licensor or as licensee — have
a lawyer confirm at minimum:

1. that PolyForm Noncommercial 1.0.0 is the right instrument for the intended
   business model, and that its "noncommercial purpose" boundary is enforceable
   in the relevant jurisdictions;
2. the LGPL-3.0 `easydict` interaction, specifically LGPL-3.0 §4 and §6 if
   anything is ever shipped as a binary or a container image;
3. that the Apache-2.0 → PolyForm transition is clean given the repository's
   never-published history (the facts are recorded at the top of `LICENSE`);
4. the contributor position above — specifically, what instrument (CLA or
   assignment) is needed so that contributions can be licensed commercially, and
   what `CONTRIBUTING.md` must say instead of inbound-equals-outbound — **before
   any outside contribution is accepted**.
