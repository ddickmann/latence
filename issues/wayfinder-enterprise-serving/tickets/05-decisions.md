# 05 — the six things only Dennis can decide

Everything else from `05-part-a-comparison.md` has been landed. These six were held back on
purpose: each is a policy or legal call, not a fix, and picking one for you would be exactly the
kind of quiet decision this repository's ADR process exists to prevent.

Each item states the **options**, what the **five reference repos** actually do, the **cost of
not deciding**, and a **recommendation**. They are ordered so you can answer them in one pass —
D1 constrains D6, D4 constrains everything with a URL in it.

Nothing here is blocked on more research. Answer inline (or in an ADR) and the follow-up work is
mechanical.

---

## D1 — Publish to PyPI at all, under PolyForm Noncommercial 1.0.0?

**Options**

1. **Signed GitHub Releases only.** Wheels + sdists + SBOM + Sigstore bundle attached to the
   Release; installation is `pip install <URL>` or from a clone. `release.yml` already does all
   of this; the `pypi-publish` job stays dead (`if: ${{ false && … }}`).
2. **Publish to PyPI.** Flip the `pypi-publish` job on, using the already-scaffolded OIDC
   trusted-publishing path and the protected `pypi` environment.
3. **Publish a subset** — e.g. `latence-retrieval` (which retains an outbound Apache-2.0 grant on
   seven modules) to PyPI, everything else by Release only.

**What the references do.** None of the five is a clean precedent, because all five are
permissively licensed. qdrant ships binaries + cosign-signed Docker images and no PyPI. Hermes
has **no PyPI workflow at all** — a curl installer plus Docker — while sitting at 230k stars.
DeepSeek publishes npm with no tags and no provenance. Only docling and Scrapling do OIDC
trusted publishing, and both are MIT/BSD. So "signed GitHub Releases, no PyPI" is an established
top-of-field shape, not a consolation prize.

**Facts that matter.** PyPI's ToU do not require an OSI licence and PolyForm-NC is royalty-free,
so the upload is *permitted*. The tension is downstream: `pip install` is overwhelmingly a
corporate action, PolyForm-NC does not licence commercial use, and corporate SCA scanners
hard-block a `PolyForm-Noncommercial-1.0.0` SPDX identifier. Publishing puts the artifact in a
channel whose majority consumer is, by the licence's own terms, unlicensed — support load rather
than adoption. Namespace, verified: `latence-core` is free; `latence` already exists (0.2.0, MIT)
under your own `latenceainew` account. Note also `LICENSE:68-99` — the retained Apache-2.0 subset
in `latence-retrieval` is invisible at distribution level, since a scanner reads package metadata
and sees only PolyForm.

**Cost of not deciding.** M6 was landed as a caveat ("not yet on PyPI, here is the clone path"),
which is honest indefinitely, so nothing is blocked *today*. What stays open: `docs/RELEASING.md`
§"Open decisions" discusses namespace and auto-publish but never raises the licence question at
all, and the `pypi-publish` enable checklist in `release.yml:228-248` presumes the answer is yes.

**Recommendation: option 1.** Signed GitHub Releases only, and write it down as an ADR so it
reads as a decision rather than an omission. Revisit only if a dual-licence or a commercial SKU
changes who the `pip install` audience is. If you take option 1, `release.yml`'s enable checklist
and `docs/RELEASING.md` should be reworded from "pending the maintainer's go-ahead" to "closed:
not publishing, and here is why".

---

## D2 — The CLA instrument, and how it is administered *(this is must-fix M4)*

`CONTRIBUTING.md` §"Before you write code" makes a signed CLA mandatory and then says the text
"is a legal document maintained separately from this page". **Nothing in the tree is that
document**, and there is no CLA automation. On day one of a public repo, the first PR cannot be
merged — the stated gate is unsatisfiable.

**Options**

1. **Apache ICLA/CCLA derivative** — `CLA.md` at root, adapted names/jurisdiction, plus
   [cla-assistant](https://github.com/cla-assistant/cla-assistant) (a GitHub App: contributor
   comments to sign, the bot records it and sets a status check).
2. **Same text, manual administration** — signed by email/e-signature before first merge,
   recorded by you, with the procedure written into `CONTRIBUTING.md`. Zero infrastructure, does
   not scale past a handful of contributors, but that may be the whole population.
3. **Drop the CLA, use DCO instead** (`Signed-off-by`, enforced by a bot). **This changes the
   licensing model** — a DCO does not give you the rights to relicense or to grant commercial
   terms over contributed code, which is the entire reason the CLA is there. Only viable if you
   accept that contributed code cannot be included in a commercial grant.
4. **Accept no external contributions**, DeepSeek-style ("we cannot accept external pull requests
   at the moment" → write a plugin). Blunt, but honest, and this repo's Provider seam makes the
   plugin route genuinely first-class.

**What the references do.** None of the five uses a CLA. docling uses **DCO 1.1** with a
`dco-advisor.yml` bot that turns a failing check into a copy-pasteable remediation command.
qdrant, Scrapling and Hermes rely on inbound=outbound licensing only. DeepSeek accepts no PRs.

**Recommendation: option 1's text, option 2's administration, for now.** Land `CLA.md` (an Apache
ICLA derivative — a lawyer should read it, but a lawyer reading a draft is much cheaper than a
lawyer drafting one) and administer it manually with the procedure spelled out in
`CONTRIBUTING.md`. Add cla-assistant only when a second contributor exists. Do not take option 3
unless you are also giving up the commercial grant. **This one genuinely needs the legal review
that `RELEASE-GATES.md` Gate 3 already lists** — it is on the same review as the LGPL interaction
and the retained Apache subset, so it costs no extra round trip.

---

## D3 — Enable GitHub Discussions on publication?

**Options**

1. **On**, and make it the front door: `blank_issues_enabled: false` (already set) + Discussions
   as contact link #1.
2. **Off**, and route questions through the issue templates only — the current landed state.
3. **Off**, route to a chat channel (Discord/Matrix) instead.

**What the references do.** 4/5 have it on. Scrapling and docling deliberately make it the front
door with blank issues off — which is exactly the shape `.github/ISSUE_TEMPLATE/config.yml` here
already assumed. Hermes has it **off** and routes to Discord. DeepSeek shows the failure mode:
with issues disabled, Discussions became an unstructured bug tracker — 1,553 threads in ~24h.

**Landed already:** both dead `/discussions` links are gone (`config.yml` now points at
`SUPPORT.md`; `SUPPORT.md` states plainly that there is no forum). So the *defect* is fixed and
nothing is blocked. This decision is now purely "do you want the channel".

**Recommendation: option 2 for launch, option 1 later if demand appears.** For a one-maintainer
project with an explicit no-SLA `SUPPORT.md`, an empty Discussions tab reads worse than no tab,
and an unmoderated one reads worse still. Turning it on later is a two-line change to two files;
turning it off after people have posted is not. **Whatever you choose, the invariant is that
`config.yml` and `SUPPORT.md` must agree with the repository's actual settings** — that is what
broke before.

---

## D4 — The final org / repo name

`ddickmann/latence-framework` is hardcoded in `mkdocs.yml:7-9`, `.github/ISSUE_TEMPLATE/config.yml`,
`SUPPORT.md`, `.github/CODEOWNERS`, `docs/RELEASING.md:65`, `release.yml`'s enable checklist, the
new `CITATION.cff`, and `[project.urls]` (Homepage / Repository / Issues / Changelog /
**Documentation**) in all 33 `pyproject.toml`s — and it currently **404s**. The PyPI `latence`
name sits under a *different* account, `latenceainew`.

**Options:** keep `ddickmann/latence-framework`; move to an org (`latence-ai/…`, matching the
existing `latenceainew` PyPI identity); or something else entirely.

**Cost of not deciding.** Everything above moves together, *including the Sigstore
`--cert-identity` string in `docs/RELEASING.md:65`* — which means a release cut before the rename
is signed under an identity that no longer exists after it. The mechanical part is cheap; the
signing identity is not.

**Recommendation: decide before publish, not after, and before the first tag.** A rename after a
signed release leaves a permanent seam in the verification story. No preference on the name
itself — that is a branding call. Two guardrails now exist to make the rename mechanical and
safe: `test_publish_readiness.py` asserts that every package's `Documentation` URL equals
`mkdocs.yml`'s `site_url`, and that `CITATION.cff` names the same repository as `latence-core`'s
`[project.urls]`, so a half-finished rename turns the suite red instead of shipping.

---

## D5 — The AI-contribution posture, and how it interacts with Part B

**Landed already (the uncontroversial half):** `CONTRIBUTING.md` now has a
`## Contributing with AI` section — disclose it, you own it, write the PR prose yourself — and
the PR template has a matching disclosure checkbox. That closes the "conspicuous silence where
5/5 peers have a policy" gap without deciding anything about *agent-facing* material.

**What is still open** is the bigger half: does this project also ship a **public, user-facing
agent-instruction file** (`AGENTS.md` and/or `CLAUDE.md`)?

**What the references do.** 4/5 ship agent-instruction files *publicly*. docling: `AGENTS.md` +
`CLAUDE.md` + `.claude/skills/` + `.agents/skills/` + `.codex/` + `.opencode/` at repo root.
DeepSeek: `AGENTS.md` (15.7 KB) and a `CLAUDE.md` whose entire content is the nine bytes
`AGENTS.md`. Hermes: `AGENTS.md` at **81 KB** — the largest governance document in that repo,
explicitly dual-audience for humans and its own triage bot. qdrant: `.github/review-rules.md`
written *for* AI review bots, plus a `docs/CONTRIBUTING.md` §"Contributing with AI" that says
outright that "I asked claude, and it generated this" is not an acceptable answer.

**The tension with Part B.** Part B's instruction is to remove explicit assistant references from
examples and internal documents. That instinct is right about *internal operating material* —
DeepSeek is the cautionary case, having published an internal mirror with Chinese-only issue
templates, an inherited GitLab CI file, and an `AGENTS.md` still asserting "no external consumers"
to 94,000 stargazers. But it is the opposite of current practice for *deliberate, externally
written* agent instructions.

**Options:** (a) ship a purpose-written public `AGENTS.md`; (b) ship nothing agent-facing, keeping
only the `CONTRIBUTING.md` section that just landed; (c) defer until after publication.

**Recommendation: (b) for launch, (a) shortly after.** The distinction that matters is not
"assistant material vs no assistant material" — it is **written for an external reader vs
inherited from internal operation**. Part B should scrub the second category ruthlessly. A public
`AGENTS.md` would be genuinely useful here (this repo's gates, contracts and Provider seam are
exactly what an agent gets wrong), but it must be *written*, not *untracked-then-restored* — so
it is a follow-up ticket, not part of the scrub.

---

## D6 — OpenSSF Best Practices enrolment (`bestpractices.dev`)

Not a gap yet, and it is here only so it does not get lost. `.github/workflows/scorecard.yml`
sets `publish_results: true`, uploading to the public OpenSSF API — but there is no badge and no
bestpractices.dev project. docling carries both the OpenSSF Best Practices badge (#10101) and the
LF AI badge, and cites the badge programme in its `SECURITY.md` as evidence.

**This is deliberately NOT actionable before publication.** The Scorecard badge is meaningless
until the repo is public, and adding either badge now would give the README exactly the
untruthful badge its own disclaimer refuses to carry.

**Recommendation: after publication**, enrol and add the badge — the questionnaire is
substantially already satisfied (SECURITY.md, CODEOWNERS, signed releases, SBOM, pinned actions,
a documented test regime), so it is close to free evidence. Sequence it with D4: enrolment is
tied to the repository URL.

---

## What was landed rather than asked

For completeness, so you know what you are *not* being asked to approve: M1 (macOS CI lane +
narrowed OS classifiers, with the Windows conclusion written down and evidenced), M2 (a py3.13
lane, run locally first), M3 (the false "first public release" changelog entry), M5 (the dead
Discussions links), M6 (the install block), and all twelve should-fix items except the parts of
S1 and S11 called out above. The six axes where `05-part-a-comparison.md` §3 records this repo as
already **above** the union bar were not touched.
