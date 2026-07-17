# Release Fleet — the single execution plan for tickets #178–#199

**What this is.** One self-contained document indexing every open pre-release ticket (#178–#199, all
labelled `ready-for-agent`), the exact dependency graph, the *file-conflict clusters* that constrain
parallelism beyond the blocking edges, and a self-contained implementation brief per ticket. It is the
launch contract for an agent fleet: each ticket can be handed to a fresh subagent that, together with
this document and the ticket's own GitHub issue body (the authoritative What-to-build + Acceptance
Criteria), has everything it needs — no other context required.

**Status:** authored, **not launched.** The fleet fires on the maintainer's explicit "go" (after the
multi-hop benchmark completes). Two source waves converge here:
- **P0-launch wave** (#178–#189) — from the competitive gap analysis; distribution/proof/integration.
- **Pre-release fix wave** (#190–#199) — from the live 519-page German-corpus run; bugs + reproducible env.

Two **external** references (not in #178–#199, must be handled): **#72** (pre-publish polish — a hard
blocker of #189) and **#156** (sink-connector umbrella — #183 ticks its first box, does not replace it).

---

## 1. The dependency graph (blocking edges only)

```
FRONTIER (no blockers — 15 tickets can start at once):
  #178 #179 #180 #181 #182 #183 #184 #185 #186   (P0-launch)
  #190 #191 #192 #193 #194 #195                   (pre-release fix)
  [#72 external — must also start; gates #189]

SECOND LAYER (unblock when their edge lands):
  #187 ← #181            (benchmark: NER/RE/ER/redaction ← parse-harness tracer)
  #188 ← #186            (console deepen ← console tracer)
  #196 ← #195            (served-OCR env ← env tracer)
  #197 ← #192            (wizard redact decision ← gliner2 loadable)
  #198 ← #195, #191, #192 (GPU learned-provider env ← tracer + device-fix + gliner2-pin)

THIRD LAYER:
  #199 ← #196, #198      (Blackwell/CUDA-13 matrix + full-stack verify)
  #189 ← #178, #72       (flip public ← signed-release flow + polish)
```

---

## 2. File-conflict clusters (the real parallelism limiter)

The blocking graph is shallow, so *most* tickets are "independent" by dependency — **but several touch
the same files** and would collide if run concurrently in the same tree. The fleet resolves this two
ways: **worktree isolation** (each agent in its own git worktree) + **cluster serialization** (tickets
that edit the same load-bearing file are done by ONE agent, in order). Clusters:

| Cluster | Tickets | Shared surface | Rule |
|---|---|---|---|
| **Wizard** | #193, #194, #197 | the `latence setup` enterprise-sota profile emission (one generator) | **One agent, serialized**: #193 then #194 immediately; #197 appended once #192 lands. Never three parallel writers. |
| **Env / lockfile** | #195 → #196, #198 → #199 | pins, extras, deploy assets, compat docs | Already serialized by the DAG; keep it so. #196 and #198 (both ← #195) are disjoint *enough* (OCR-serving vs learned-provider) to run parallel **after** #195, in separate worktrees. |
| **Core seams** | #184, #185, #190 | `latence-core` (Export / Runner / the Provider perf seam) | Different core files (export vs runner vs perf-seam) → low collision; run in **isolated worktrees**, integrate in this order (#190 seam first, then #184, then #185) so the shared-seam change lands before the feature adds. |
| **Front-door docs** | #178, #179, #180, #189 | README + mkdocs nav | #179/#180 add nav entries (small); #189 rewrites the README and is **last** (← #178,#72). Worktree-isolate; integrate #189 after the others. |
| **Benchmark** | #181 → #187 | the benchmark harness | Serial by DAG. |
| **Console** | #186 → #188 | the run console | Serial by DAG. |

Genuinely independent (new packages / disjoint files — safe to parallelize freely): **#182** (SharePoint
Source pkg), **#183** (Qdrant sink pkg), **#191** (gliner_pii pkg), **#192** (gliner2-pii pkg),
**#186**/#181 tracers, **#178**/#179/#180 (mostly disjoint doc/CI files).

---

## 3. Shared implementation standard (applies to EVERY ticket — stated once)

Every fleet agent gets this preamble; the per-ticket brief in §5 adds only the specifics.

- **Repo:** `/Users/dennisdickmann/Developer/latence-framework`. Branch off latest `main` into
  `agent/<issue-number>-<slug>` (one branch per ticket). Do **not** push, open a PR, or merge — the
  integration step and the maintainer own that.
- **Read the issue first:** `gh issue view <N>` is the authoritative What-to-build + Acceptance Criteria.
  Implement **every** checkbox. This document gives you the *where* and the *standard*; the issue gives
  the *what*.
- **Code-quality standard (mattpocock codebase-design):** deep modules — a lot of behaviour behind a
  small interface; seams only where behaviour actually varies (two adapters, not one); accept
  dependencies, don't create them; the interface is the test surface (test observable behaviour, not
  internals).
- **Project bar (non-negotiable):** `ruff` clean; `mypy --strict` clean on every touched package (src +
  tests); pytest green; ship `py.typed`; heavy deps live in the per-Provider package, never in
  `latence-core` (ADR-0016). New Providers register under `latence.providers`; retrieval components under
  `latence.retrieval`.
- **Anti-false-green (mandatory):** every fix ships a **regression test that fails on the pre-fix code and
  passes after** — state that you observed it fail. Offline fakes must faithfully mirror **and reject**
  what the real dependency does (a fake that swallows what a real API rejects is forbidden). Assert
  behaviour, never a flag.
- **License diligence (ADR-0012):** any named model/store gets weights **and** code license verified
  separately + cited, or flagged `UNVERIFIED` behind a placeholder — never a hardcoded unverified model.
- **Verify before commit:** run `bash scripts/verify-local.sh` (the full merge gate: secret-scan · ruff ·
  mypy-strict-per-package · pytest · conformance · stack-validate · bake-off). If it's too slow for a
  single package during iteration, run the touched-package subset, then the full gate once before the
  final commit. **`verify-local` green is the definition of done** (GitHub Actions is billing-blocked, so
  it is the only gate).
- **Exemplars to mirror:** a Provider package → `packages/latence-embedder-st/`. The retrieval library →
  `packages/latence-retrieval/`. The wizard → `packages/latence-core/src/latence_core/setup/wizard.py`.
  The Export/signal seam → `packages/latence-core/src/latence_core/stages/export.py`. The device/perf
  seam → `packages/latence-core/src/latence_core/providers/`.
- **Return** a structured report: what you built, exact commands + real results, the regression-test name
  + your evidence it failed pre-fix, any AC unmet (honestly), any `UNVERIFIED`.

---

## 4. Launch waves (how the fleet is orchestrated)

The fleet runs as a Workflow (pre-authored, held). Each ticket = one implementer agent in an **isolated
worktree**, then an adversarial verifier, then integration onto a shared release-integration branch with
`verify-local` as the wave gate. Order:

- **Wave 1 (parallel, ~13 lanes):** #178, #179, #180, #181, #182, #183, #184, #185, #186, #190, #191,
  #192, #195 — plus the **Wizard-cluster lane** doing #193+#194 as one serialized agent. (#72 is a
  maintainer/human polish task done alongside.)
- **Wave 2 (parallel, unblocked):** #187 (←181), #188 (←186), #196 (←195), #198 (←195,191,192), #197
  (←192, appended to the Wizard lane).
- **Wave 3:** #199 (←196,198); then **#189** (←178,#72) — the launch, whose final visibility-flip +
  PyPI push stay **maintainer-manual**.

Integration order within a wave respects §2's cluster rules (seam-first: #190 → #184 → #185; wizard
serialized; #189 last).

---

## 5. Per-ticket briefs (issue = What/ACs; this = where + specifics)

> Each row: **Issue** · **Blocked-by** · **Cluster** · **Where to build / seam** · **Governing ADRs** · **the one thing not to get wrong**.

### P0-launch wave

- **#178 — Sign releases + per-release SBOM.** blocked-by: none · cluster: front-door docs · **where:**
  `.github/workflows/release.yml` + the existing SBOM script (`docs/SBOM.md` references it) + `docs/RELEASING.md`;
  no package. · ADRs: — · **don't get wrong:** demonstrate the whole flow on a dry-run tag; sign real artifacts, don't just document intent.
- **#179 — Continuity & security-process statement.** none · front-door docs · **where:** new
  `docs/CONTINUITY.md` (or similar) + README link + mkdocs nav. · **don't get wrong:** every claim
  verifiable in-repo — no aspirational security posture (anti-false-green applied to prose).
- **#180 — Measured throughput envelopes.** none · front-door docs · **where:** extend
  `docs/PERF-RESULTS.md` + README/mkdocs link. · **don't get wrong:** only measured numbers with
  hardware/profile/date/repro; an explicit "not measured" list; do not anticipate #64's parked 100k/1M numbers.
- **#181 — Benchmark harness (Parse tracer).** none · benchmark (→#187) · **where:** new benchmark
  harness (its own dir; not a pipeline Provider) + a published results page in docs. · ADRs: ADR-0028
  (adopter gold-set is separate; this is the *public* harness). · **don't get wrong:** verify the public
  dataset's license before inclusion; pin dataset revision + stack config; our numbers reproducible,
  competitor numbers labelled CLAIMS.
- **#182 — SharePoint Source connector.** none · independent · **where:** new package
  `latence-source-sharepoint` mirroring `latence-embedder-st`; Source Capability behind the existing seam. ·
  ADRs: **ADR-0014** (Source seam, thin connectors), Delta/manifest per ADR-0018. · **don't get wrong:**
  content-hash manifest so Delta skips unchanged; secrets env-only; faithful auth/pagination/rate-limit fakes.
- **#183 — Qdrant sink loader (first slice of #156).** none · independent · **where:** new package
  `latence-sink-qdrant`; a one-shot loader (NOT a held index). · ADRs: **ADR-0017** (files stay source of
  truth), **ADR-0048** (stateless). · **don't get wrong:** idempotent upserts keyed by record id;
  Retraction/Purge → verifiably absent after re-load; **tick #156's checkbox with a pointer, don't duplicate it.**
- **#184 — Iceberg-table Export.** none · core-seams (integrate 2nd) · **where:**
  `latence-core/stages/export.py` — optional output alongside Parquet, via an Iceberg REST catalog. ·
  ADRs: **ADR-0017**. · **don't get wrong:** off by default → byte-identical existing export when
  disabled; verify against a local/OSS REST catalog offline (no cloud dep).
- **#185 — OpenLineage run events.** none · core-seams (integrate 3rd) · **where:**
  `latence-core/runner.py` + `observability.py`. · ADRs: **ADR-0034** (counts-only, thin tracing). ·
  **don't get wrong:** counts-only asserted by test (no content/PII/secrets in any facet); off by default;
  emission failure never fails the run; verify against local Marquez.
- **#186 — Run console tracer.** none · console (→#188) · **where:** new read-only localhost console over
  Storage files (its own package/tool). · ADRs: ADR-0010/0017 (read-only over files, no server state),
  ADR-0041 (adjacent). · **don't get wrong:** strictly read-only, offline, no new state; legible to a
  non-engineer (headline verdicts, not JSON).
- **#187 — Benchmark: NER/RE, ER, redaction recall.** **blocked-by #181** · benchmark · **where:** extend
  the #181 harness with the remaining suites onto the same results page. · **don't get wrong:** synthetic
  PII only (no real PII in-repo); verify each dataset license; folded P1 German slice only if free.
- **#188 — Console deepen (Quarantine/chunk/KG+Evidence).** **blocked-by #186** · console · **where:**
  extend the #186 console. · **don't get wrong:** the chunk inspector resolves Provenance→page/offsets;
  every KG edge shows its Evidence; still read-only/offline.
- **#189 — Flip public (repo + PyPI + docs site).** **blocked-by #178, #72** · front-door docs (integrate
  LAST) · **where:** README, mkdocs, package metadata, go-live runbook. · **don't get wrong:** fix the
  gap-analysis inconsistencies (a Runner adapter claimed in README that doesn't exist; stale ADR count;
  dangling cross-refs) *before* the flip; rehearse PyPI in a clean env; **the visibility-flip + PyPI push
  stay maintainer-manual** — this ticket prepares and verifies up to that trigger.

### Pre-release fix wave

- **#190 — gliner2 JIT/nvrtc crash on Blackwell.** none · core-seams (integrate 1st) · **where:**
  `latence-extract-gliner2` + the **shared Provider seam** so every mdeberta-backed Provider inherits it
  (the perf/compile seam in `latence-core/providers/`). · ADRs: **ADR-0036** (W9 compile-guard reasoning),
  ADR-0004. · **don't get wrong:** fix at the shared seam, not per-Provider; throughput-only degradation,
  output byte-identical; regression test exercises/asserts the JIT-neutralization. (Proven workaround:
  `PYTORCH_JIT=0` — encode it in the seam, don't leave it to operators.)
- **#191 — redaction.gliner_pii ignores cuda.** none · independent · **where:** `latence-pii-gliner`
  package. · ADRs: **ADR-0036** (device profiles), ADR-0007. · **don't get wrong:** place the model on the
  resolved device at load; regression test asserts model-parameter device == resolved device
  (device-parameterized so CPU-CI still runs it); redaction spans identical across devices.
- **#192 — redaction.gliner2 load fails vs gliner2 1.3.2.** none · independent (feeds #197,#198) ·
  **where:** `latence-pii-gliner2` package + its dependency pin. · ADRs: **ADR-0036** (W9), **ADR-0016**
  (per-Provider pin), ADR-0044 (financial-PII floor). · **don't get wrong:** pin the supported gliner2
  version in the package; an unsupported version fails at resolution or with a typed error, never a native
  AttributeError; ADR-0044 floor intact.
- **#193 — enterprise-sota omits graph_features.** none · **Wizard cluster (agent does 193+194)** ·
  **where:** `latence-core/setup/wizard.py` enterprise-sota profile. · ADRs: **ADR-0041**, **ADR-0051**;
  machinery exists (#161). · **don't get wrong:** wire `graph_features.pagerank` into the emitted graph
  Stage; test the emitted config includes it AND the columns appear in a run's export; deterministic/additive.
- **#194 — enterprise-sota omits BM25 signal.** none · **Wizard cluster (same agent as 193)** · **where:**
  same wizard profile, corpus Export block. · ADRs: **ADR-0041**, **ADR-0049**; exporter exists (#164). ·
  **don't get wrong:** wire the `bm25` block; test the emitted config includes it AND the artifact
  materializes.
- **#195 — Reproducible-env tracer (CPU/text stack).** none · env (→#196,#198) · **where:** pinned
  lockfile + install recipe (and/or container) + a one-command smoke test; builds on `deploy/`. · ADRs:
  **ADR-0016** (per-Provider extras are the mechanism), ADR-0007, ADR-0041. · **don't get wrong:** the
  smoke test IS the regression guard — future dependency drift turns it red; pins are enforced
  constraints, not folklore.
- **#196 — Served-vLLM OCR env.** **blocked-by #195** · env · **where:** pinned image/recipe for the vLLM
  LightOn-OCR endpoint, isolated from the text stack via the ADR-0030 endpoint seam; `deploy/`. · **don't
  get wrong:** the incompatible prebuilt wheels (flash-attn, torchcodec) **cannot** end up installed —
  enforce by pins/exclusions, not "uninstall" instructions; document the two-environment split.
- **#197 — Wizard/profile redact-provider disagreement.** **blocked-by #192** · **Wizard cluster (append
  to 193/194 agent after #192)** · **where:** wizard profile emission + the profile design doc. · ADRs:
  ADR-0041, ADR-0044, ADR-0036. · **don't get wrong:** decide the blessed redactor *in light of #191/#192
  fixes*, record the reasoning, make emission and docs agree; ADR-0044 floor preserved.
- **#198 — GPU learned-Provider env.** **blocked-by #195, #191, #192** · env · **where:** pinned GPU
  extras/lockfile for GLiNER2/GLiNER-PII/GLinker/Granite on CUDA-13; resolve the transformers <5 vs ≥5
  split via the ADR-0030 endpoint boundary or a documented two-phase install; fold in #190's JIT fix. ·
  ADRs: ADR-0016, ADR-0007, ADR-0045, ADR-0046. · **don't get wrong:** exactly one supported resolution;
  the GPU smoke asserts **real GPU placement** (so the #191-class silent-CPU-fallback can't return);
  per-Provider extras, not a monolith.
- **#199 — Blackwell/CUDA-13 compat matrix + full-stack verify.** **blocked-by #196, #198** · env · **where:**
  a compat matrix in the deploy/install docs + one documented full-stack end-to-end run. · ADRs: ADR-0016,
  ADR-0036. · **don't get wrong:** every cell measured or explicitly `UNVERIFIED`; the live-run failure
  modes (FlashInfer sm75 check, nvrtc/JIT, wheel ABI breaks) appear as matrix entries with resolutions.

---

## 6. External prerequisites (not #178–#199)

- **#72 — pre-publish polish sweep.** Hard blocker of **#189**. Install-story phrasing matches what's
  installable; package license metadata in clean SPDX; third-party license snapshot refreshed. Can run in
  Wave 1 alongside the fleet (doc/metadata work).
- **#156 — sink-connector umbrella.** Not a blocker. **#183** executes its first checkbox and must update
  it (tick + pointer), not duplicate it.

---

## 7. Launch

The fleet Workflow is authored and held at the session scratch path (see the maintainer's chat). On
"go" it runs Wave 1 → 2 → 3 per §4, isolated worktrees, `verify-local` as each wave's gate, integrating
onto a `feat/release-fleet-integration` branch. Nothing is pushed or merged; the maintainer reviews the
integration branch and runs the two manual steps (#189's visibility-flip + PyPI push). Total scope: 22
tickets across 3 waves; realistic parallel width ~13 lanes in Wave 1.
