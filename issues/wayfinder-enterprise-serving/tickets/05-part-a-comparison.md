# 05 Part A — the SOTA-OSS comparison and gap analysis

Status: **research complete, nothing changed.** This document is the analysis deliverable for
ticket `05-oss-release-bar-and-public-scrub.md` Part A. **No repo file was modified.** A later
pass lands the fixes; this pass only establishes what the bar is, where we sit against it, and
what is a decision rather than a fix.

Analysis date: **2026-08-14**. Local repo state: branch `hypergraph-m1`, HEAD `e34e339`.

---

## 1. Method and what was studied

Each reference repo was resolved to its canonical GitHub name via the GitHub API and read at the
revision below (raw files at the default branch, `.github/workflows` enumerated via the contents
API). Where a claim is about a workflow, the workflow file and job/step name are quoted.

| Repo studied | Canonical name | Default branch | Revision marker (`pushed_at`) | Licence | Stars | Language |
|---|---|---|---|---|---|---|
| Qdrant | `qdrant/qdrant` | `master` (dev branch is `dev`) | 2026-08-14T18:06:03Z | Apache-2.0 | 33,976 | Rust |
| docling | `docling-project/docling` — `DS4SD/docling` is a **redirect** (repo id 826168160) | `main` | 2026-08-14T16:19:56Z | MIT (LF AI & Data) | 64,772 | Python |
| Scrapling | `D4Vinci/Scrapling` | `main` | 2026-08-11T18:45:32Z | BSD-3-Clause | 73,954 | Python |
| Hermes agent | `NousResearch/hermes-agent` | `main` | 2026-08-14T19:18:01Z | MIT | 230,597 | Python |
| DeepSeek harness | `deepseek-ai/deepseek-harness` | `master` | 2026-08-13T13:00:21Z | MIT | 94,226 | TypeScript |

Two framing notes that change how their practice should be read:

- **`deepseek-ai/deepseek-harness` is a push-mirror of an internal repo, published 2026-08-13.**
  12,293 commits, merge commits referencing a *different* org (`deepseek-harness`), `has_issues:
  false`, PRs unavailable (`GET /pulls` → 404), zero tags, zero releases. Its thinness is
  amputation, not immaturity — and its visible defects (Chinese-only issue/PR templates, an
  `AGENTS.md` still asserting "no external consumers" to 94k stargazers, a `.gitlab-ci.yml` for an
  internal pipeline, a Pages workflow that publishes nothing) are **exactly the failure mode
  Part B of this ticket exists to prevent.** It is the cautionary reference, not the model.
- **`NousResearch/hermes-agent` has Discussions disabled** and routes community to Discord.

---

## 2. Comparison, axis by axis

### Axis 1 — README

| | Qdrant | docling | Scrapling | Hermes | DeepSeek harness | **This repo** |
|---|---|---|---|---|---|---|
| Badges | 6, `flat-square`: workflow-status (`rust.yml`), OpenAPI docs, licence, Discord, Roadmap-2025, Qdrant Cloud | 16: Trendshift, arXiv `2408.09869`, docs-live, PyPI version, PyPI pyversions, uv, Ruff, Pydantic v2, prek, licence, pepy downloads, Apify, Dosu, Discord, **OpenSSF Best Practices #10101**, **LF AI & Data** | 11 (raw HTML block): Trendshift, Tests workflow, PyPI version, Docker pulls, downloads, Agent-Skill, Clawhub, Discord, X, pyversions | 6, **all hand-made static** `for-the-badge`: Docs, Discord, licence, "Built by Nous Research", 3 language links. No CI/PyPI/coverage badge | **zero badges** | 5 static: licence (PolyForm), Python 3.11+, `tests 5,139 passing`, `mypy --strict`, `status pre-1.0` |
| Above the fold | logo (`<picture>` light/dark) → tagline → badges → 4 prose paras → nav strip (Quick Start · Agent Skills · Clients · Demos · Integrations · Contact) | banner `docs/assets/docling_processing.png` → Trendshift → badge block → one-sentence pitch | cover SVG → badges → doc-nav row → 3 prose paras → **code sample** → 2nd code sample → sponsor tables (`# Platinum Sponsors` at L90, above `## Key Features` at L230) | banner → title → nav → badges → one bolded claim → 7-row HTML capability table | language switcher → one-sentence identity → plugin pitch → **stability warning** | title → bolded value claim → badges → **explicit no-CI/no-PyPI-badge disclaimer** → `## The problem` → `## 60-second quickstart` |
| Benchmarks in README | **linked out, zero inline numbers** — one line, L26, → `qdrant.tech/benchmarks/` | **none at all**; delegated to arXiv 2408.09869 | **two inline tables** with absolute ms + ratio-vs-Scrapling; no dates, no hardware | **none**; capability claims are prose only | none (`BENCHMARK.md` is 231 B, a pointer) | inline `### The measured retrieval result` + `### Enterprise serving, measured (2026-08)` (README L258, L280), each carrying date/corpus, plus a `Check our numbers` row pointing at `docs/BENCHMARKS.md` + `docs/running-the-benchmarks.md` |
| Quickstart | Docker one-liner `docker run -p 6333:6333 qdrant/qdrant` + **an immediate insecure-default caveat** + 3-line Python | 3 steps: `pip install docling` → CLI one-liner → 5-line Python | 5-line snippet + a `Spider` snippet | `curl -fsSL … \| bash` installer (+ PowerShell); `pip install` never shown | `npx @deepseek-ai/dsh web` | `git clone` → `uv sync` → `uv run latence-demo`, with the **expected output pasted inline** and a stated non-zero-exit contract |
| Honesty device | insecure-default caveat next to the quickstart | "Python 3.9 support was dropped in 2.70.0" note | none | ~30-line antivirus false-positive block | **"THERE WILL BE COMPATIBILITY-BREAKING CHANGES."** as §2, before install | `## Status & maturity` with a *Not yet proven* list, `RELEASE-GATES.md`, and `PENDING:` tokens that fail the build under `LATENCE_RELEASE_CHECK=1` |

**Read:** on README honesty this repo is the strongest of the six — nobody else publishes a
"what is not yet true" section, and nobody else machine-enforces it. On badges it is mid-pack and
deliberately so; Hermes at 230k stars proves hand-made static badges are acceptable at the top of
the field. The one real defect is that `tests 5,139 passing` is a *static* badge over a number
that drifts every commit.

### Axis 2 — Contribution surface

| | Qdrant | docling | Scrapling | Hermes | DeepSeek | **This repo** |
|---|---|---|---|---|---|---|
| `CONTRIBUTING.md` | 7-line stub → `docs/CONTRIBUTING.md`. Rule stated 4× : PRs target `dev`, not `master` | thin, delegates policy to `docling-project/community`; keeps uv/ruff/`ty`/`prek` mechanics | full: PR must target `dev`; tox on all Pythons; mypy **and** pyright; conventional commits; "don't put your name in the code" | **48,835 B**. Ranked contribution priorities; *search-first* not issue-first; routes memory providers + vendor integrations out to standalone plugins | 1,574 B: **"We are sorry that we cannot accept external pull requests at the moment."** → write a plugin, tag it `dsh-plugin` | 349 lines: CLA requirement, uv workspace setup, `verify-local.sh` as the merge gate, test-first rule, DoD tiers, doc-truth gates, ADR process, Provider-without-touching-core |
| Inbound IP | inbound=outbound Apache-2.0. **No DCO, no CLA** | **DCO 1.1** (`Signed-off-by`), `.github/dco.yml` `allowRemediationCommits: individual: true` | none stated | **none** — one line: contributions licensed MIT | none (no PRs) | **CLA required, no exceptions** (`CONTRIBUTING.md` §"Before you write code") |
| `CODE_OF_CONDUCT.md` | `docs/CODE_OF_CONDUCT.md`, Covenant **v2.0**, contact `andrey@vasnetsov.com` | stub → community repo; Covenant **v2.0**, `deepsearch-core@zurich.ibm.com` | present, verbatim Covenant | **MISSING** | **MISSING** | present, Covenant **v2.1**, contact `ddickmann81@gmail.com` |
| `SECURITY.md` | **MISSING** (root, `.github/`, and org `.github` repo all 404) | `.github/SECURITY.md`; GitHub private advisory preferred; **"acknowledged and analyzed … within 3 working days"** | **MISSING** | present, 15,750 B — a real threat model (trust model, in/out of scope, 90-day disclosure) | **MISSING** — for a repo shipping a sandbox/landlock escape surface | present, 210 lines: private advisory + email, ack ≤3 business days, triage ≤10, 90-day embargo target, trust model, PII floor, "what is not yet true" |
| Issue templates | 4 legacy `.md`; **no `config.yml`** (404) → Discord/Discussions not surfaced in the chooser | 3 legacy `.md`; `config.yml` has **only** `blank_issues_enabled: false`, no contact_links | **4 `.yml` forms**, zero `.md`; `config.yml` → Discussions + Discord | 4 `.yml` forms; `config.yml` → Discord, Docs, Contributing | 6 templates, **all Chinese**, inert (issues disabled); `contact_links: []` | **3 `.yml` forms** (`bug_report`, `feature_request`, `provider_request`) + `config.yml` with `blank_issues_enabled: false` and 3 contact_links (Discussions, private advisory, CONTRIBUTING) |
| PR template | 3 checklist blocks; leads with the `dev`-branch rule | 4 HTML-comment steps + 3-item checklist; requires conventional-commit **title** | "DO NOT DELETE ANY TEXT"; 9-box type checklist; 8-item author checklist | heavy: type checkboxes, Code checklist, Docs & Housekeeping checklist, "For New Skills" block | Chinese; "非 Draft 人类 PR 至少引用一个同仓库 Issue" — distinguishes human from agent PRs | 8 sections / ~30 checkboxes: Licence(CLA), DoD tier, **"The test that was red"** + anti-false-green, Gates (exit code by redirect, not pipe), Contracts/packaging/licences, Security/secrets/PII, Docs & decisions, "Not run locally" |
| `CODEOWNERS` | **MISSING** | **MISSING** | — | **MISSING** | — | **present**, 8 rules routing core / ADRs / SECURITY / NOTICE / licences / `.github` / `issues` |
| **AI-contribution policy** | **`docs/CONTRIBUTING.md` §"Contributing with AI"** — 4 sub-rules: no AI-written PR prose to humans; "'I asked claude, and it generated this' is not an acceptable answer"; `[AI]`/`[manual]` commit prefixes; **disclose your prompt**. Plus `.github/review-rules.md` written *for* AI review bots | `AGENTS.md` + `CLAUDE.md` at repo root; `.claude/skills/`, `.agents/skills/`, `.codex/`, `.opencode/`, `.plans/` | **`AI_POLICY.md`** — disclosure mandatory; undisclosed AI output gets the public **`AI-Slop` label** and is closed | `AGENTS.md` **81,342 B**, the largest governance doc in the repo; dual-audience (humans + the triage sweeper), explicitly fences which closes a bot may not make | `AGENTS.md` 15,737 B; `CLAUDE.md` whose entire content is the 9 bytes `AGENTS.md`; `.agents/notes/` decision records mandatory per PR | **NONE.** `grep -rni "AI assist\|AI-generated\|AI tool\|copilot\|claude\|chatgpt" CONTRIBUTING.md .github/ CODE_OF_CONDUCT.md SUPPORT.md` returns **zero hits** |

**Read:** this repo's contribution surface is the strongest of the six on paper (only repo with
CODEOWNERS; newest CoC; issue forms better than qdrant's and docling's; SECURITY.md matched only
by Hermes and docling). Two real holes: the **CLA text does not exist** — `CONTRIBUTING.md` says
"The CLA text is a legal document maintained separately from this page" and nothing in the tree
is it, so the stated gate is unsatisfiable on day one — and there is **no AI-contribution
policy**, which 5/5 references have in some form.

### Axis 3 — CI

| | Qdrant | docling | Scrapling | Hermes | DeepSeek | **This repo** |
|---|---|---|---|---|---|---|
| OS matrix | **`ubuntu-latest`, `windows-latest`, `macos-latest`** in `rust.yml` job `rust-tests` | `ubuntu-latest` default; `windows-latest` (`run-tests-windows`) + `macos-latest` (`run-tests-macos`) in `checks.yml`, nightly/label-gated | tests **macOS only** (4 matrix entries); lint on `ubuntu-latest` | `ubuntu-latest` + a dedicated `tests-os.yml` on `macos-latest`/`windows-latest` by marker | ubuntu / `windows-2025` / `macos-latest` / self-hosted pools | **`ubuntu-latest` only.** `grep -rn "macos\|windows" .github/workflows/` → no match |
| Interpreter matrix | n/a (Rust) | `'["3.10"]'` on cheap PRs; `'["3.10","3.11","3.12","3.13","3.14"]'` on nightly/`tests:full` label | 3.10, 3.11, 3.12, 3.13 via tox | **3.11 only**, but `requires-python = ">=3.11,<3.14"` caps the claim | Node 24 primary; 22.19 + 26 compat | `["3.11","3.12"]` (`ci.yml` L96-97) with **uncapped `requires-python = ">=3.11"`** in all 33 packages |
| Caching | `Swatinem/rust-cache@e18b4977…` | `astral-sh/setup-uv@v7 enable-cache: true`; APT cache; **HF/modelscope/EasyOCR model cache** keyed on `uv.lock`+`pyproject.toml`; `~/.cache/prek` cache | 3-layer `actions/cache` (Playwright binaries, `.tox`), save gated on `main` | `astral-sh/setup-uv` SHA-pinned + `uv cache prune --ci`; test-duration cache for LPT slicing | pnpm store cache + wine apt cache | `astral-sh/setup-uv@11f9893b… enable-cache: true` in every job |
| Action pinning | full commit SHA + `# vX` comment | full commit SHA + `# vX` comment | tag-pinned (`@v6`) | full commit SHA | full SHA on policy workflows | **full commit SHA + `# vX` comment** everywhere |
| Concurrency | **none in any of 18 workflows** | `${{ github.workflow }}-${{ PR number \|\| ref }}`, cancel-in-progress | `${{workflow}}-${{ref}}`, cancel | `ci-${{ github.ref }}`, cancel on PR only | cancel on PR, **not** on push (self-hosted drills) | `${{ github.workflow }}-${{ github.ref }}`, `cancel-in-progress: true` (ci.yml L12-14) |
| Single required check | — | `check` "All Green" via `.github/scripts/check_needs_results.py --allowed-skips` | — | `all-checks-pass`, comment: "Branch protection should require ONLY this check" | `all-checks-passed`, **fails on `skipped` too** | **none** — branch protection would have to enumerate every job/matrix leg by name |
| Coverage | `coverage.yml` on `workflow_dispatch` + nightly cron, **never on PR**; codecov project target 50, patch off | codecov, **both project and patch `informational: true`**, split into 7 named components | `--cov` via tox, uploaded | — | — | surfaced to `$GITHUB_STEP_SUMMARY` + uploaded artifact, **never hard-gated** (ci.yml L106-108 comment: "a % threshold flakes") |
| Release automation | `release-artifacts.yml` on `release: published` (Linux/macOS/Windows/musl/deb/AppImage); `docker-image.yml` on `push: tags: '*'` → Docker Hub + GH Packages, **cosign-signed** | `cd.yml` (`workflow_dispatch`) → python-semantic-release → `gh release create` → `pypi.yml` on `release: published` → **OIDC trusted publish, `attestations: true`**, 3 packages, `environment: pypi` | `release-and-publish.yml` on **`pull_request: closed`+merged+owner**, version from PR **title**, notes from PR **body**, OIDC trusted publish, sdist+wheel | **no PyPI workflow at all**; only `docker.yml` → Docker Hub multi-arch by digest. PyPI `hermes-agent` sits at 0.19.0 while `pyproject.toml` says 0.20.1 | `release.yml` publishes the *packed artifact bytes* (no rebuild) behind `environment: npm-publish`; **no `--provenance`** | `release.yml` on `v*` tag: build all 33 → **CycloneDX SBOM + SHA256SUMS + keyless Sigstore** (`release-integrity.sh --sign`) → GitHub Release with generated notes + `VERIFY.md`; `pypi-publish` job present and **deliberately dead** via `if: ${{ false && … }}` (L254) behind a protected `pypi` environment |
| Supply chain | dependabot (cargo/actions/docker) → `target-branch: dev` | `uv.lock` + `uv-lock` pre-commit hook; **no dependabot** | dependabot n/a; bandit in CI | dependabot **actions-only** by design; exact-pinned deps ("Mini Shai-Hulud worm", 2026-05-12) | dependabot npm/uv/actions; **`allowBuilds` deny-by-default** with per-package justifications | dependabot (pip + actions, grouped minor/patch); `codeql.yml` (weekly cron, python); `scorecard.yml` (weekly, `publish_results: true`); `detect-secrets` audited-baseline gate as the **first** CI job |
| Local hooks | `.rusty-hook.toml` | `.pre-commit-config.yaml` + `prek`, and **`prek run --all-files` IS the CI lint step** | `.pre-commit-config.yaml` mirroring CI exactly | **none** | `lefthook.yml` installed by `postinstall` | **none** — `.pre-commit-config.yaml` absent |

**Read:** this repo's release pipeline is the most rigorous of the six (nobody else ships SBOM +
keyless Sigstore + a signing-identity refusal guard + a self-reporting publish gate). Its **CI
breadth is the weakest**: it is the only repo of the six that never leaves Linux, and the only one
whose `requires-python` floor has no ceiling and no matching test lane.

### Axis 4 — Docs

| | Qdrant | docling | Scrapling | Hermes | DeepSeek | **This repo** |
|---|---|---|---|---|---|---|
| Site | **separate repo** `qdrant/landing_page` | in-repo `docs/`, `mkdocs.yml` config but built by **`zensical build`** | in-repo `docs/`, **`zensical.toml`**, built on readthedocs (no CI workflow) | **Docusaurus** in `/website/`, deployed to Vercel + GH Pages | VitePress in `/website/`; **`has_pages: false`, the Pages URL 404s** | in-repo `docs/`, `mkdocs.yml` (319 lines), **mkdocs-material**, GH Pages via `docs.yml` |
| API reference | **OpenAPI**, hand-authored ytt templates in `/openapi/`, generated by `tools/generate_openapi_models.sh`, committed to `docs/redoc/<version>/openapi.json` — **41 versioned dirs in-tree** | `mkdocstrings[python]` + `griffe_pydantic` (`schema: true`), google style; `reference/cli.md` generated by `scripts/render_cli_reference.py` | `mkdocstrings` + griffe plugins, sphinx docstring style, 7 generated reference pages | **none** — hand-written CLI/env-var pages | generated catalogs (`gen-config-catalog`, `gen-tool-catalog`) each with a paired `verify-* --check` drift gate | **`mkdocstrings`** (`mkdocs.yml` L64-79), sphinx docstring style, over `latence-core` + 3 CPU reference Providers; `docs/reference/` has 9 pages |
| Versioned docs | 41 OpenAPI versions in-tree | **no `mike`** — single-version + `mkdocs-redirects` | single-version (readthedocs `/latest/`) | single-version | single-version | single-version. **Not a gap: 0/5 references use `mike`.** |
| Strict build | — | `ci-docs.yml` builds on PR, paths-filtered | RTD builds; no PR gate | `docs-site-checks.yml` | `verify-md-links`, `verify-doc-refs`, `doc-typecheck` (typechecks code blocks) | **`uv run mkdocs build --strict`** on every PR (`docs.yml`, step "Build (strict — fail on warnings)"), deploy job gated on `push` to `main` |
| Examples that run | — | **`docs/examples/*.py` executed in CI** — `run-examples-light` in `checks.yml`, heavy ones in `ci-heavy-examples.yml`, with explicit `EXAMPLES_HEAVY` / `EXAMPLES_UNSUPPORTED_IN_CI` regex lists so **every example is classified, never silently unrun** | `pytest.ini` sets `--doctest-modules` (docstring examples execute) | `hermes --toolsets skills -q "…"` as an E2E check in the PR template | doc code blocks typechecked | `stacks/*.yaml` fully gated (9 validated in `ci.yml` job `stacks`, all globbed in `test_envspec.py:738`, `test_stack_validate.py:90`). **`examples/*.yaml` (15 files) are not** — only 2 are touched, and only for a PII property (`test_example_corpus_pii_default.py:28`) |
| Contributing/Changelog in the docs nav | — | yes | **yes** — nav's last two entries are external links to `CONTRIBUTING.md` and `/releases` | yes | — | **no** — `grep -n "CONTRIBUTING\|CHANGELOG\|SECURITY" mkdocs.yml` returns nothing |

### Axis 5 — Release discipline

| | Qdrant | docling | Scrapling | Hermes | DeepSeek | **This repo** |
|---|---|---|---|---|---|---|
| `CHANGELOG.md` | **absent** | **auto-generated** by python-semantic-release ~7.32 (angular parser; `perf` → patch) | **absent by design** — `pyproject` `Changelog` URL → `/releases` | **absent** (code search `filename:CHANGELOG.md` → 0) | **absent** | present, **Keep a Changelog 1.1.0 + SemVer**, hand-written, 554 lines |
| Tag convention | `vMAJOR.MINOR.PATCH`; `master` release-only, `dev` accumulates | `vMAJOR.MINOR.PATCH`, ~2 releases/week, currently v2.120.x | `vX.Y.Z`, every 1–2 weeks | **CalVer tag** `v2026.8.13` + SemVer release *name* `v0.20.1`; ~1 tag/7–8 days | `<semver>-rc.<n>`; **zero tags, zero releases** on the public mirror | `vX.Y.Z` documented in `docs/RELEASING.md` L14. **Zero `v*` tags exist** (`git tag -l 'v*'` → empty) |
| Notes | handwritten `# Change log` with emoji sections; PR URL + human summary; `[[docs](…)]` per feature. **Zero auto-generated markers** | semantic-release-generated, supplied via `gh release create -F`; `"name": null` on every release | **handwritten long-form**, 4–6 KB each, every bullet carries a runnable snippet + a docs deep-link | **two-tier**: minors are 40–58 KB themed essays ("The Herald Release"); patches ~1 KB and self-describe as mechanical | none | `generate_release_notes: true` (auto from merged PRs) **plus** `VERIFY.md` appended, generated by `release-integrity.sh` from the same tag that was signed |
| Release integrity | cosign-signed Docker images | PyPI attestations (PEP 740) via trusted publishing | trusted publishing, sdist+wheel | Docker digests | packed-bytes publish, **no provenance** | **CycloneDX SBOM + SHA256SUMS + keyless Sigstore bundle attached to the Release**; `github-release` job hard-fails if `SHA256SUMS.sigstore.json` is missing; a "Refuse to publish under a non-tag signing identity" step blocks a mis-signed release |

### Axis 6 — Community affordances

| | Qdrant | docling | Scrapling | Hermes | DeepSeek | **This repo** |
|---|---|---|---|---|---|---|
| Discussions | **enabled**, 327, 5 categories | **enabled**, 461, incl. a **`Roadmap`** category | **enabled**, and made the front door (blank issues off, Discussions is contact link #1) | **disabled** → Discord | enabled; 1,553 in ~24h, and General is functioning as an unstructured bug tracker | **not enabled** — the repo `ddickmann/latence-framework` returns **404** (unpublished), yet `config.yml:4` and `SUPPORT.md:13` both link `/discussions` |
| Roadmap | **`docs/roadmap/README.md`** in-tree + 2022/2023/2024 archives; surfaced only by a README badge | Discussions `Roadmap` category + README `### Coming soon` | **`ROADMAP.md`** at root, flat 12-item checklist, 8 done | **none** | **none** | **none.** `RELEASE-GATES.md` is a gate checklist, not a roadmap; the word "roadmap" appears nowhere in README/CONTRIBUTING/SUPPORT/issue templates |
| Benchmark reproduction | **separate repo** `qdrant/vector-db-benchmark`; qdrant/qdrant does not link it, only the results page | **none** — `…/benchmarks/` 404s; numbers delegated to arXiv. The one recipe (`docs/examples/parquet_images.py`) names the dataset + exact `vllm serve` command | **`benchmarks.py` at repo root**, linked from README and `docs/benchmarks.md`; real script, `timeit` 100 reps, fairness note in-code. **No pinned comparator versions, no hardware, no run instructions** | **none** | none | **`docs/running-the-benchmarks.md`** — `uv run latence-benchmark run --suite all`, per-suite table with a **"Published today: yes/no"** column, licence check + pinned dataset revision per suite, and every results section dated. `docs/BENCHMARKS.md` separates *our measurements* from *vendor claims* |
| Citation | — | **`CITATION.cff`** + arXiv badge + `@techreport` bib block with DOI | `## 🎓 Citations` BibTeX section | — | — | **none** |
| Funding | — | — | `FUNDING.yml` (3 platforms) + sponsor tables in README **and** appended to every release body | — | — | `FUNDING.yml` present, **all entries commented out** by decision (ADR-0001, non-commercial) |

---

## 3. The union bar, and where this repo already clears it

Before the gap list, the things a later pass **must not "fix"**, because they are already at or
above the union bar and several are best-in-class across the six:

1. **README honesty.** `## Status & maturity` with an explicit *not yet proven* list, machine-
   enforced by `PENDING:` tokens that fail under `LATENCE_RELEASE_CHECK=1`. No reference repo does
   this. Keep the "no CI or PyPI badge, because no release exists" disclaimer.
2. **Release integrity.** SBOM + keyless Sigstore + signing-identity refusal + a self-reporting
   PyPI gate (`scripts/pypi-gate-status.sh`, run in a job that *always* executes precisely because
   a skipped job emits nothing). Strictly ahead of all five.
3. **SECURITY.md.** Only Hermes and docling have a comparable one; qdrant, Scrapling and DeepSeek
   have **none**.
4. **CODEOWNERS.** The only repo of the six with one.
5. **Issue forms + `config.yml` contact links.** Better than qdrant (no config at all) and docling
   (empty contact_links).
6. **Keep-a-Changelog + SemVer.** 4/5 references have no changelog at all.
7. **Benchmark reproduction.** Licence-checked, revision-pinned, per-suite published/not-published
   column, dated sections. The best of the six by a wide margin.
8. **Legal/continuity surface.** `COMMERCIAL.md`, `NOTICE`, `THIRD-PARTY-LICENSES.md`,
   `docs/CONTINUITY.md` (bus factor, fork posture, CRA readiness), `docs/THREAT-MODEL.md`,
   `RELEASE-GATES.md`. No reference repo carries this much, and `RELEASE-GATES.md` Gate 3 already
   anticipates most of the licensing work.
9. **Versioned docs are not a gap** — 0/5 references use `mike`.

---

## 4. Gap list

Every gap names the exact file to change and what the change is. Ranking is
`must-fix` (blocks going public / a truthfulness defect) / `should-fix` (below the union bar,
non-blocking) / `flag-for-Dennis` (needs a decision before it can be a fix).

### must-fix

**M1 — CI never leaves Linux, but all 33 packages claim `Operating System :: OS Independent`.**
`grep -rn "macos\|windows" .github/workflows/` returns no match; every `runs-on:` in all seven
workflows is `ubuntu-latest`. Meanwhile `packages/latence-core/pyproject.toml:18` (and all 33)
carries `"Operating System :: OS Independent"`. 5/5 reference repos exercise ≥2 OSes.
*Change:* `.github/workflows/ci.yml` — add a macOS (and ideally Windows) lane. Cheapest shape that
matches docling: keep `ubuntu-latest` as the PR default and add `macos-latest` + `windows-latest`
legs gated on a label or the nightly `ci-heavy.yml` schedule (Hermes' `tests-os.yml` is the model,
including its guard that **fails the job on pytest exit code 5** so a mis-selected marker cannot
pass green). If Windows is genuinely unsupported (`deploy/env` glinker wheels are Linux/x86-64
only per `docs/compatibility-matrix.md:28`), then the classifier must change instead — in all 33
`packages/*/pyproject.toml`.

**M2 — `requires-python = ">=3.11"` is unbounded but only 3.11/3.12 are tested.**
All 33 packages declare `requires-python = ">=3.11"` (e.g. `packages/latence-core/pyproject.toml:10`);
`ci.yml:96-97` matrixes `["3.11", "3.12"]`; classifiers stop at `Python :: 3.12`. So 3.13 and 3.14
installs are permitted and never tested. Hermes solves this with `>=3.11,<3.14` and a 10-line
justification comment; docling matrixes through 3.14 on the nightly.
*Change:* either cap to `>=3.11,<3.13` in all 33 `packages/*/pyproject.toml`, **or** add 3.13 to the
`ci.yml` matrix and the classifier lists. Do not leave the claim wider than the test.

**M3 — `CHANGELOG.md` records a "first public release" that never happened, under the wrong licence.**
`CHANGELOG.md:488-492`:
```
## [0.1.0] - 2026-07-14

First public, coherent release of the workspace. All publishable distributions
(`latence-core` plus the provider packages) are pinned to `0.1.0` and licensed
Apache-2.0.
```
This contradicts `README.md:381` ("**Nothing has been released.** No public artifact exists and
nothing is on PyPI"), it predates the 2026-07-29 PolyForm relicence recorded in `LICENSE`, and
`git tag -l 'v*'` is empty. The two link refs at `CHANGELOG.md:553-554` point at
`compare/v0.1.0...HEAD` and `releases/tag/v0.1.0`, both of which will 404 the day the repo is public.
*Change:* `CHANGELOG.md` — either relabel the `[0.1.0]` entry as an internal, never-published
version bump (and drop/redirect the two link refs), or fold it into `[Unreleased]` and let the
first real tag be the first entry. Strike "First public" and "licensed Apache-2.0" either way.

**M4 — the CLA is a stated gate with no artifact behind it.**
`CONTRIBUTING.md` §"Before you write code" makes a signed CLA mandatory for every contribution,
then states: "The CLA text is a legal document maintained separately from this page. This section
states the *requirement*; it is not itself the agreement and grants nothing." Nothing in the tree
is that document, and no CLA-assistant automation exists (`.github/workflows/` has no CLA check;
compare docling's `dco-advisor.yml`, which turns a failing check into a copy-pasteable
`git commit --allow-empty -s -m "DCO Remediation Commit …"` comment). On day one of a public repo,
the first PR cannot be merged.
*Change:* add the CLA text (e.g. `CLA.md`, or an Apache ICLA/CCLA derivative) and either a
CLA-assistant workflow or an explicit manual procedure in `CONTRIBUTING.md`. **The choice of CLA
instrument is a decision — see D2.**

**M5 — `SUPPORT.md` and the issue chooser route users to Discussions that do not exist.**
`.github/ISSUE_TEMPLATE/config.yml:4` and `SUPPORT.md:13` both link
`https://github.com/ddickmann/latence-framework/discussions`. `GET /repos/ddickmann/latence-framework`
returns **404** — the repo is unpublished, so Discussions is not enabled and the link is dead.
Hermes ships exactly this defect today (a `CONTRIBUTING.md` "GitHub Discussions" bullet against
`has_discussions: false`) and it is called out as a defect there.
*Change:* enable Discussions when the repo is published (decision D3), or remove/replace both
links. Whichever, the two files must agree with the repo's actual settings before publish.

**M6 — the `## Install` section shows ten `pip install` commands for distributions that do not exist.**
`README.md:389-406` lists `pip install latence-core`, `latence-demo`, `latence-console`,
`latence-retrieval`, and six opt-in packages. `https://pypi.org/pypi/latence-core/json` → **404**;
nothing is published. `test_front_door_truthfulness.py::test_readme_install_lines_name_real_distributions`
checks the *names* match workspace distributions, not that they are installable. The honest line
does exist at `README.md:381`, eight lines above, but the install block itself reads as live.
*Change:* `README.md` — add an inline caveat inside `## Install` (one line, e.g. "*Not yet on PyPI —
see [Status & maturity](#status--maturity). Until a release is cut, install from a clone.*") and
give the from-clone path. Depends on D1.

### should-fix

**S1 — no AI-contribution policy.** 5/5 references have one (qdrant `docs/CONTRIBUTING.md`
§"Contributing with AI" + `.github/review-rules.md`; Scrapling `AI_POLICY.md` + the public
`AI-Slop` label; Hermes `AGENTS.md` §"Contribution Rubric"; docling and DeepSeek ship `AGENTS.md`
+ `CLAUDE.md` at repo root). This repo has zero hits for AI-assistance terms across
`CONTRIBUTING.md`, `.github/`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`.
*Change:* add a `## Contributing with AI` section to `CONTRIBUTING.md` (disclosure requirement,
"you own the output", no AI-written PR prose aimed at a human reviewer), and reference it from
`.github/PULL_REQUEST_TEMPLATE.md`. **Interacts directly with Part B — see D5.**

**S2 — no single aggregate required check.** 3/5 references (docling `check`, Hermes
`all-checks-pass`, DeepSeek `all-checks-passed`) end CI with one job that `needs:` everything, so
branch protection requires exactly one status. Here, `ci.yml` has five independent jobs plus a
2-leg matrix, so protection must enumerate `test · conformance · coverage (py3.11)`,
`… (py3.12)`, `lint · type (fast gate)`, `secret scan (detect-secrets)`, `stack validate (Goal A
— end-to-end)`, `build wheels + sdists (all packages)` by name — and adding a matrix leg silently
leaves it unprotected.
*Change:* `.github/workflows/ci.yml` — add a final `all-green` job with `if: always()` that fails
on any `needs.*.result` other than `success` (DeepSeek's version also fails on `skipped`; docling
uses an explicit `--allowed-skips` allowlist).

**S3 — no `.pre-commit-config.yaml`.** 4/5 references run local hooks (docling `prek`, Scrapling
`pre-commit`, qdrant `rusty-hook`, DeepSeek `lefthook`), and docling makes the hook run *be* the CI
lint step so the two cannot drift. Here `scripts/verify-local.sh` is the full mirror, which is
excellent but is a whole-gate run, not a commit-time one.
*Change:* add `.pre-commit-config.yaml` with ruff check + ruff format + the `detect-secrets` gate,
and note it in `CONTRIBUTING.md` §"Development setup". Keep `verify-local.sh` as the merge gate.

**S4 — `examples/*.yaml` are not validated.** 15 files in `examples/`; only `graph-slice.yaml` and
`demo-pipeline.yaml` are touched, and only for a PII-ancestor property
(`packages/latence-core/tests/test_example_corpus_pii_default.py:28`). `stacks/*.yaml` by contrast
are globbed and validated (`test_envspec.py:738`, `test_stack_validate.py:90`, plus the `stacks`
job in `ci.yml:175-185`). docling's rule is that every example is *classified* (light / heavy /
unsupported-in-CI) so none is silently unrun.
*Change:* add a globbing test over `examples/*.yaml` that at minimum parses and `stack validate`s
each, with an explicit skip-list carrying a reason per entry. README quotes
`diff examples/demo-pipeline.yaml examples/demo-pipeline.swap.yaml` as a proof, so these are
front-door artifacts.

**S5 — no roadmap.** 3/5 references publish one (qdrant `docs/roadmap/README.md` with year
archives; Scrapling `ROADMAP.md`; docling a Discussions Roadmap category + README "Coming soon").
`grep -rn "roadmap"` over `README.md CONTRIBUTING.md SUPPORT.md .github/ISSUE_TEMPLATE/*.yml`
returns nothing.
*Change:* cheapest honest version — a `## Roadmap` row in the README "Where to go next" table
pointing at `RELEASE-GATES.md` + the *not yet proven* list, or a thin `ROADMAP.md`. Do **not**
invent dates.

**S6 — the docs site never surfaces Contributing / Changelog / Security.**
`grep -n "CONTRIBUTING\|CHANGELOG\|SECURITY" mkdocs.yml` returns nothing; `not_in_nav` excludes only
`PITCH-SLIDES.md`. Scrapling's nav ends with two external links (Contributing → the GitHub file,
Changelog → `/releases`), which costs two lines.
*Change:* `mkdocs.yml` — add external nav entries for `CONTRIBUTING.md`, `CHANGELOG.md`,
`SECURITY.md` (external links avoid `--strict` complaining about files outside `docs_dir`).

**S7 — no `Documentation` URL in package metadata.** `grep -l "Documentation" packages/*/pyproject.toml`
→ 0 of 33. `[project.urls]` carries Homepage / Repository / Issues / Changelog only
(`packages/latence-core/pyproject.toml:66-70`), while `mkdocs.yml:7` declares
`site_url: https://ddickmann.github.io/latence-framework/`. Every reference Python package links
its docs from PyPI.
*Change:* add `Documentation = "<site_url>"` to `[project.urls]` in all 33
`packages/*/pyproject.toml`. Blocked on D1 only insofar as the site URL depends on the final org.

**S8 — no `CITATION.cff`.** docling ships one plus an arXiv DOI badge and a `@techreport` bib
block; Scrapling ships a `## 🎓 Citations` section. This repo publishes benchmark methodology
(`docs/BENCHMARKS.md`, `docs/running-the-benchmarks.md`) and ADRs that are citable work.
*Change:* add `CITATION.cff` at root (cff-version 1.2.0, author, title, licence, repo URL).
GitHub renders a "Cite this repository" button from it automatically.

**S9 — the `tests 5,139 passing` badge is static and drifts.** `README.md:9` is a hardcoded
shields.io badge; `README.md:329-331` correctly says the count "grows as work lands — the command
is the source of truth, not this line". The badge does not carry that caveat and will be wrong
within a day of any merge. Hermes proves static badges are acceptable; a *stale count* is not.
*Change:* `README.md` — replace with a non-numeric badge (`tests passing`, or a `mypy --strict`-style
capability badge), or wire a shields.io endpoint from the coverage/test summary the CI job already
emits (`ci.yml:133-143` already prints a badge-ready number).

**S10 — `RELEASE-GATES.md` Gate 3 carries a now-stale parenthetical.** `RELEASE-GATES.md:64-65`
reads "All 33 `packages/*/pyproject.toml` declare the same SPDX identifier — **one package still
declaring the old licence** undermines the whole relicence". Verified:
`grep -h "^license = " packages/*/pyproject.toml | sort | uniq -c` → `33 license = "PolyForm-Noncommercial-1.0.0"`,
and `grep -L "PolyForm-Noncommercial-1.0.0" packages/*/pyproject.toml` is empty. The condition is met.
*Change:* `RELEASE-GATES.md` — tick the box or reword to state the invariant rather than an
outstanding defect.

**S11 — no OpenSSF Best Practices enrolment despite running Scorecard.**
`.github/workflows/scorecard.yml` sets `publish_results: true` (line ~44), which uploads to the
public OpenSSF API — but there is no badge and no bestpractices.dev project. docling carries both
the OpenSSF Best Practices badge (#10101) and the LF AI badge, and its `.github/SECURITY.md` cites
the badge programme as evidence.
*Change:* after publication, enrol at bestpractices.dev and add the badge to `README.md`. Note the
Scorecard badge itself will be meaningless until the repo is public — do not add either badge
before the repo exists, or the README acquires exactly the untruthful badge its own disclaimer at
`README.md:13-15` currently refuses to carry.

**S12 — the `[Unreleased]` backlog is larger than the released history and there is no cadence.**
`CHANGELOG.md` is 554 lines; `[Unreleased]` spans lines 10–487 with 33 top-level entries, versus
the single `[0.1.0]` section. All 33 packages sit at `version = "0.1.0"`. Reference cadences:
docling ~2 releases/week, Scrapling 1–2 weeks, Hermes ~7–8 days.
*Change:* not a file edit on its own — it is the consequence of M3 + D1. Once the first real tag is
cut, the `[Unreleased]` block becomes that release's section and the versions bump off `0.1.0`.

### flag-for-Dennis (blocked on a decision, listed in §5)

- **F1** — whether to publish to PyPI at all under PolyForm-Noncommercial (→ D1). This blocks M6,
  S7, the `pypi-publish` enable checklist in `release.yml:228-248`, and `docs/RELEASING.md`
  §"Open decisions", which currently discusses namespace and auto-publish but **does not raise the
  licence question at all**.
- **F2** — CLA instrument and mechanism (→ D2). Blocks M4.
- **F3** — whether to enable GitHub Discussions (→ D3). Blocks M5.
- **F4** — final org/repo name, which every URL in the tree hardcodes (→ D4).
- **F5** — the AI-policy posture, which is the constructive counterpart to Part B's scrub (→ D5).

---

## 5. Decisions for Dennis — not fixes, and not decided here

**D1 — Publish to PyPI at all, given PolyForm-Noncommercial 1.0.0?**
Facts gathered, no recommendation made:
- PyPI's Terms of Use do **not** require an OSI licence. Content carrying a royalty-free "Included
  License" may be redistributed by the PSF as uploaded; otherwise the uploader grants the PSF and
  *all other users* an irrevocable worldwide licence to reproduce and distribute. PolyForm-NC is
  royalty-free, so the upload itself is permitted.
- The tension is downstream, not at upload: `pip install` is overwhelmingly used inside companies,
  and PolyForm-NC does not licence commercial use. Publishing puts the artifact in a channel whose
  majority consumer is, by the licence's own terms, unlicensed — and corporate SCA scanners will
  hard-block a `PolyForm-Noncommercial-1.0.0` SPDX identifier, generating support load rather than
  adoption.
- Namespace reality, verified: `https://pypi.org/pypi/latence-core/json` → **404** (free);
  `https://pypi.org/pypi/latence/json` → **exists**, version 0.2.0, MIT, repo
  `https://github.com/latenceainew/latence-trace-python` — i.e. the lead name is already yours
  under the `latenceainew` account. `docs/RELEASING.md` §"Open decisions" flags the namespace but
  not the licence question.
- Note also `LICENSE:68-99`: seven `latence-retrieval` modules keep an outbound Apache-2.0 grant.
  A scanner reading package metadata sees only PolyForm, so the exception is invisible at the
  distribution level — `RELEASE-GATES.md:80-84` already flags this for legal review.
*Options seen in the reference set:* GitHub Release assets only (qdrant does binaries+Docker,
DeepSeek does npm-only-with-no-tags, Hermes has no PyPI workflow at all and ships via a curl
installer + Docker). "Signed GitHub Releases, no PyPI" is an established SOTA shape, not a
consolation prize.

**D2 — CLA instrument, and how it is administered.**
None of the five references uses a CLA. docling uses **DCO 1.1** with a bot that auto-generates the
remediation command; qdrant, Scrapling and Hermes rely on inbound=outbound only; DeepSeek accepts
no PRs. A CLA here is structurally forced by the licence (the owner cannot grant commercial rights
over code they do not hold rights in), so the question is not *whether* but *which and how*:
which text (Apache ICLA/CCLA derivative? a bespoke one?), and administered by a bot (cla-assistant),
by a signed document, or manually in-review. Until this is settled, M4 cannot be landed.

**D3 — Enable GitHub Discussions on publication?**
4/5 references have it on; Scrapling and docling make it the deliberate front door with blank
issues disabled — which is exactly the shape `config.yml` here already assumes. Hermes uses Discord
instead. DeepSeek shows the failure mode: with issues off, Discussions becomes an unstructured bug
tracker (1,553 threads in 24 hours). For a one-maintainer project with an explicit no-SLA
`SUPPORT.md`, "on" and "off" are both defensible; what is not defensible is the current state,
where two files link a Discussions tab that does not exist (M5).

**D4 — The final org/repo name.** `ddickmann/latence-framework` is hardcoded in `mkdocs.yml:7-9`,
`.github/ISSUE_TEMPLATE/config.yml`, `SUPPORT.md:13`, `.github/CODEOWNERS`, `CHANGELOG.md:553-554`,
`docs/RELEASING.md:65`, `release.yml`'s enable checklist, and `[project.urls]` in all 33
`pyproject.toml`s — and it currently 404s. The PyPI `latence` name sits under `latenceainew`. If the
repo lands under a different owner, every one of those URLs and the Sigstore `--cert-identity`
string in `docs/RELEASING.md:65` change together. Decide before publish, not after.

**D5 — AI-contribution posture, and how it interacts with Part B.**
Part B instructs removing explicit Claude/assistant references from examples, internal documents
and markdowns. The reference set shows the *opposite* is current SOTA practice at the very top of
the field, and it is worth deciding deliberately rather than by omission:
- **4 of 5 ship agent-instruction files publicly.** docling: `AGENTS.md` + `CLAUDE.md` + `.claude/skills/`
  + `.agents/skills/` + `.codex/` + `.opencode/` at repo root. DeepSeek: `AGENTS.md` (15.7 KB) and a
  `CLAUDE.md` whose entire content is the nine bytes `AGENTS.md`. Hermes: `AGENTS.md` at **81 KB**,
  the largest governance document in that repo, explicitly dual-audience for humans and its own
  triage bot. qdrant: `.github/review-rules.md` written *for* AI review bots.
- **3 of 5 have an explicit AI-contribution policy** (qdrant, Scrapling, Hermes), and all three are
  about *disclosure*, not concealment. qdrant's names the failure mode directly: "'I asked claude,
  and it generated this' is not an acceptable answer, and we might discard the PR". Scrapling
  attaches a public `AI-Slop` label to undisclosed AI output.
- DeepSeek is the counter-example that supports Part B's instinct: it published an internal mirror
  without an external-reader pass, and shipped Chinese-only issue templates, an inherited GitLab CI
  file, a Pages workflow that publishes nothing, and an `AGENTS.md` line still asserting "no
  external consumers" to 94,000 stargazers.
So the question for Dennis is not "scrub or don't scrub" — the internal operating material
absolutely should go (that is Part B's real target). It is: **do we also ship a deliberate,
user-facing AI story (an `AGENTS.md` and/or a `## Contributing with AI` section) rather than
leaving a conspicuous silence where every peer repo has a policy?** S1 is written as a
should-fix on the assumption the answer is yes, but it is a decision, not a fix.

---

## 6. Cross-dependency with Part B (noted, not acted on)

Two Part-A facts constrain Part B and should be read before the scrub runs:

- **`README.md` makes 7 links into `issues/`** — lines 221, 300, 315, 341, 362 (×2), 373, reaching
  `issues/e2e-extended/DECISIONS.md`, `issues/wayfinder-enterprise-serving/tickets/01-report.md`
  and `03-report.md`, `issues/gate/G1-RESULTS.md`, and `issues/e2e-extended/README.md`. If Part B
  untracks the `issues/` tree, those links dangle on the public repo. `README.md` is in the
  `_FRONT_DOOR` set of `packages/latence-core/tests/test_front_door_truthfulness.py:52`, and
  `test_no_dangling_cross_references_in_the_front_door` (line 274) resolves links against
  **git-tracked paths** — so untracking `issues/` will turn that test red, exactly as ticket 05
  §Part B item 5 anticipates. Plan the README rewrite and the untrack as one change.
- **`scripts/` contains at least one clearly-internal artifact** — `scripts/_pod_probe_gliner.py`,
  alongside eleven `validate-*-on-gpu.md` runbooks. Classification is Part B's job; flagging here
  only because `scripts/` is not on Part B's named suspect list (`issues/`, wave reports,
  FINDINGS, campaign STATE).

---

## 7. Verification appendix — every claim about this repo

| Claim | Verified by |
|---|---|
| 7 workflows, all `runs-on: ubuntu-latest`, no macOS/Windows | `grep -rhn "runs-on:" .github/workflows/` → 16 hits, all `ubuntu-latest`; `grep -rn "macos\|windows" .github/workflows/` → no match |
| Python matrix `["3.11","3.12"]` | `.github/workflows/ci.yml:96-97` |
| All 33 packages `requires-python = ">=3.11"` | `grep -h "requires-python" packages/*/pyproject.toml \| sort \| uniq -c` → `33 requires-python = ">=3.11"` |
| All 33 packages claim `OS Independent`, none lists Python 3.13 | `grep -l "OS Independent" packages/*/pyproject.toml \| wc -l` → 33; `grep -l "Python :: 3.13" …` → 0 |
| All 33 packages PolyForm, all at `0.1.0` | `grep -h "^license = " packages/*/pyproject.toml \| sort \| uniq -c` → 33 PolyForm; same for `^version = ` → 33 × `0.1.0` |
| No `Documentation` URL in any package | `grep -l "Documentation" packages/*/pyproject.toml \| wc -l` → 0; `[project.urls]` at `packages/latence-core/pyproject.toml:66-70` |
| `pypi-publish` is dead by construction | `.github/workflows/release.yml:254` — `if: ${{ false && github.event.inputs.dry_run != 'true' }}`; enable checklist at L228-248; gate self-report step at L116-117 |
| Release signs and refuses a mis-signed identity | `release.yml:127-139` step "Refuse to publish under a non-tag signing identity"; `release.yml:204-207` step "Fail if the release is not signed" |
| Docs build strictly on PR | `.github/workflows/docs.yml` step "Build (strict — fail on warnings)" → `uv run mkdocs build --strict --site-dir site`; deploy job `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` |
| API reference is mkdocstrings over 4 packages | `mkdocs.yml:61-79` |
| No Contributing/Changelog/Security in the docs nav | `grep -n "CONTRIBUTING\|CHANGELOG\|SECURITY" mkdocs.yml` → no match; `not_in_nav` at L318-319 lists only `PITCH-SLIDES.md` |
| Coverage surfaced, never gated | `ci.yml:106-108` comment + steps "Coverage summary → job summary + badge number" (L133) and "Upload coverage report" (L145) |
| Secret scan is the first CI job | `ci.yml:25-51`, job `secret-scan`, step "detect-secrets (audited baseline gate)" → `scripts/secret_scan_gate.py` |
| 9 stacks validated end-to-end in CI | `ci.yml:175-185`, job `stacks` |
| `examples/*.yaml` not validated | 15 files in `examples/`; only `test_example_corpus_pii_default.py:28` references two of them, for a PII property |
| `CHANGELOG.md` "first public release", Apache-2.0 | `CHANGELOG.md:488-492`; link refs `CHANGELOG.md:553-554` |
| No `v*` tags | `git tag -l 'v*'` → empty (48 tags exist, all `archive/*`) |
| README says nothing released | `README.md:381` |
| README install block | `README.md:389-406` |
| `latence-core` not on PyPI; `latence` is, under `latenceainew` | `https://pypi.org/pypi/latence-core/json` → 404; `https://pypi.org/pypi/latence/json` → 0.2.0, MIT, `github.com/latenceainew/latence-trace-python` |
| GitHub repo does not exist | `https://api.github.com/repos/ddickmann/latence-framework` → **404** |
| Discussions linked from two files | `.github/ISSUE_TEMPLATE/config.yml:4`; `SUPPORT.md:13` |
| No AI policy anywhere in the contribution surface | `grep -rni "AI assist\|AI-generated\|AI tool\|LLM-generated\|generative ai\|copilot\|claude\|chatgpt" CONTRIBUTING.md .github/ CODE_OF_CONDUCT.md SUPPORT.md` → no match |
| Absent files | `.pre-commit-config.yaml`, `CITATION.cff`, `GOVERNANCE.md`, `MAINTAINERS.md`, `ROADMAP.md`, `AGENTS.md`, `CLAUDE.md`, `.editorconfig`, `.github/release-drafter.yml` — all confirmed absent |
| CoC is Covenant v2.1 | `CODE_OF_CONDUCT.md:117-119`, contact at L63 |
| SECURITY response targets | `SECURITY.md:9-37` |
| `RELEASE-GATES.md` Gate 3 stale item | `RELEASE-GATES.md:64-65` vs the verified 33/33 PolyForm result above |
| README links into `issues/` | `grep -on "issues/[A-Za-z0-9/._-]*" README.md` → lines 221, 300, 315, 341, 362 (×2), 373 |
| Front-door link gate resolves against tracked paths | `packages/latence-core/tests/test_front_door_truthfulness.py:52` (`_FRONT_DOOR`), `:58` (`_tracked_paths`), `:274` (`test_no_dangling_cross_references_in_the_front_door`) |
| Scorecard publishes results, no badge | `.github/workflows/scorecard.yml:44` — `publish_results: true`; the 5 badges at `README.md:7-11` contain no Scorecard or OpenSSF badge |
