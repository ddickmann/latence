# Licence terms and permission to use

**The Latence Framework is proprietary. No licence is granted without the written
permission of the copyright holders.** There is no public grant, no price list
and no tier on this page; permission is given in writing or not at all.

The terms are in [`LICENSE`](./LICENSE) (SPDX: `LicenseRef-Proprietary`):
"Copyright (c) 2026 The Metis authors. All rights reserved." This document is a
plain-language pointer. **`LICENSE` governs.** Where this document and `LICENSE`
differ, `LICENSE` wins.

> **The public snapshot stays Apache-2.0.** The repository's public history
> began at a snapshot committed on 2026-11-11 under the Apache License 2.0. That
> grant was made and cannot be withdrawn: a copy of that snapshot may still be
> used under Apache-2.0 by anyone who holds it. Nothing added or changed after it
> is covered by that grant
> ([ADR-0076](./docs/adr/0076-relicence-the-private-line-to-proprietary.md), which
> supersedes [ADR-0063](./docs/adr/0063-relicence-to-apache-2-0.md) for the
> private line). The page keeps its name so that inbound links from the README,
> `NOTICE` and the earlier ADRs land on the current terms.

---

## Asking for permission

To ask for permission to use, evaluate or deploy the framework, write to:

**ddickmann81@gmail.com**

A permission is whatever the written agreement says; nothing on this page or in
an informal exchange grants one.

## What the terms do *not* reach

The project's terms settle rights in **the Latence Framework's own code** and
nothing else. They cannot and do not grant or withhold rights in
third-party software, model weights, or datasets, several of which carry their
own commercial-use restrictions. The project's terms are never a shield against
inbound terms, and those terms bind you directly. The ones that matter, all documented in
[`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md):

| Component | Terms | Why it matters commercially |
|---|---|---|
| `easydict` (opt-in `latence-linkpred-ultra[ultra]`) | **LGPL-3.0** | Weak copyleft with real conditions if you ship binaries. See the LGPL section of `THIRD-PARTY-LICENSES.md`. |
| NVIDIA CUDA wheels (`deploy/env/gpu-learned/`, `deploy/env/served-ocr-server/`) | **Proprietary NVIDIA EULAs** | Not open-source. Installing either environment means accepting NVIDIA's terms. |
| `google/embeddinggemma-300m` | **Gemma license** — opt-in, non-default, no Gemma-backed Provider ships here | Permits commercial use but carries a Prohibited Use Policy and a flow-down obligation to every downstream recipient. |
| `opendatalab/OmniDocBench` | **Research-only, non-commercial** | Gated behind `--accept-non-commercial` in the benchmark harness. Never usable in a commercial benchmark. |
| `naver/splade-v3` weights | **CC-BY-NC-SA-4.0** | Gated behind `--accept-noncommercial-license`. The SPLADE *code path* is this repository's; those *weights* carry their own terms. |
| `Babelscape/REDFM` | CC BY-SA 4.0 | Commercial use permitted with attribution; ShareAlike binds adaptations of the corpus. |
| Every other runtime dependency and default model checkpoint | Permissive (Apache-2.0 / MIT / BSD / ISC / PSF / MPL-2.0) | No restriction on commercial use. |

You are responsible for your own compliance with those third-party terms.
`latence env matrix` machine-checks the CUDA disclosure, and the benchmark
harness's licence gate refuses a non-commercial corpus unless you opt in
explicitly — but neither is a substitute for your own review.

## Contributions

[`CLA.md`](./CLA.md) at the repository root is a Contributor Licence Agreement,
and `CONTRIBUTING.md` makes a recorded signature a precondition of every merge.
Under the proprietary terms it is load-bearing again: Section 2 is the grant
that lets the copyright holders include a contribution in the work they license
on their own terms. `CLA.md` has not been reviewed by a lawyer, and it says so
at the top. That review is tracked as `RELEASE-GATES.md` **Gate 3 — Licence
coherence**, and it is open.

## Legal review

**This is not legal advice, and it has not been reviewed by counsel.** Before
relying on it, have a lawyer confirm at minimum:

1. the LGPL-3.0 `easydict` interaction, specifically LGPL-3.0 §4 and §6 if
   anything is ever shipped as a binary or a container image;
2. the boundary between the Apache-2.0 snapshot of 2026-11-11 and the
   proprietary line after it (the facts are recorded in [`NOTICE`](./NOTICE)
   and in [ADR-0076](./docs/adr/0076-relicence-the-private-line-to-proprietary.md));
3. the contributor position above, and the open points the CLA's own
   [Notes for the reviewer](./CLA.md#notes-for-the-reviewer) list.
