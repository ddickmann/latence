# Ticket 04 report — enterprise readiness pass

Status: **done** (2026-08-13). Every checkmark below is backed by a command that was actually
executed on this Mac (the pod untouched throughout — map ruling 5). Where a claim could not be
verified here, it is stated as deferred with the reason, not checked.

## Per-area status

| # | Area | Status | Proof (commands actually run) |
|---|---|---|---|
| 1 | README | **fixed + verified** | Quickstart from a true clean clone (fresh `uv` venv in a scratch dir): `git clone <repo> && uv sync` → exit 0; `uv run latence-demo` → exit 0, output byte-matches the README block; `uv run latence-demo --entity-provider entity.gliner` without the package → exit 1 naming `latence-ner-gliner` (the documented refusal). New "Enterprise serving, measured (2026-08)" section sourced ONLY from opened result files (tickets 01–03 + S3 metrics.json + `benchmark/serving/results/*full__ef512.summary.json`); the in-flight S5 graph-rescue recall cells are explicitly marked sweep-completing and no recall number is quoted from an unlanded cell. Stale "not yet proven" bullets rewritten to what is now true. |
| 2 | Docs | **fixed + verified** | `uv run mkdocs build --strict --site-dir site` in the clean clone (CI-verbatim): **exit 0** (was exit 1 — four `../COMMERCIAL.md` links in CONTINUITY.md, fixed). Journey coverage completed: NEW `docs/guides/serving-qdrant.md` (ticket-02 compose/loader/backend + full-set parity tables) and NEW `docs/guides/benchmark-campaign.md` (the campaign wizard — previously zero docs); incremental-ingest guide placement confirmed (Operate a stack + guides index, per ticket 03); deployment.md gains the per-stage compute story. Doc-truthfulness gates green: `pytest packages/latence-core/tests/test_front_door_truthfulness.py test_benchmarks_page_truthfulness.py` → all passed. |
| 3 | Deployment | **fixed + verified** | Local install: clean-clone `uv sync` (above) + wheels-path: fresh `python3 -m venv`, `pip install` of the six built core wheels → `latence --help` OK, full `latence-demo` run exit 0. Docker: `docker build -f deploy/Dockerfile` — **found the shipped production image's CLI dead on arrival** (editable workspace venv; `latence --help` = ModuleNotFoundError); fixed (`--no-editable`/`--no-sources` + an in-build `RUN latence --help` smoke), rebuild exit 0, CLI + imports verified in the final image (480 MB, 0 editable .pth). `docker compose config --quiet` clean for both compose files; pinned Qdrant compose live: `/readyz` → HTTP 200 "all shards are ready", and `LATENCE_QDRANT_URL=… pytest packages/latence-sink-qdrant` → 110 passed incl. the live-gated legs. Cloud-agnostic story written into deployment.md from the S3 measured numbers; `latence-runner-airflow` stated honestly: tested (11 passed + 4 airflow-gated skips here; the real `build_dag` lane runs in ci-heavy with constraints-pinned apache-airflow), **not yet operated**. GPU compose full-up not runnable on this Mac (no NVIDIA toolkit) — deferred with reason; config validated + image built + smoked instead. |
| 4 | Licenses | **fixed + verified** | `pytest packages/latence-core/tests/test_publish_readiness.py` → 25 passed (24 pre-existing + 1 new gate). Audit of all 33 `packages/*/pyproject.toml`: uniform `license = "PolyForm-Noncommercial-1.0.0"`, `version = "0.1.0"`. **Found + fixed 16 stale "framework code is Apache-2.0" claims** the 2026-07-29 relicence left in package README prose (the grant-line regex never matched them); `latence-retrieval`'s README now names the seven retained Apache-2.0 modules; NEW test `test_no_package_readme_claims_its_own_code_is_apache` pins the prose. Third-party model/dataset lanes: THIRD-PARTY-LICENSES.md gains the seven e2e-campaign dataset lanes (mirroring the machine-enforced CORPUS_PLAN) — one place. NOTICE clean (release-integrity + publish-readiness suites green). |
| 5 | Repo hygiene | **fixed + verified** | `git ls-files` junk scan: no pyc/DS_Store/tmp; the five `*.interrupted.log` files under tickets/01 are committed parity evidence (kept deliberately). `.gitignore` now covers the S3 export copies, campaign run logs, and the whole `benchmark/sota-corpus/` tree (nothing under it was ever tracked). Ticket statuses: enterprise-serving 01–03 done, 04 done (this commit); e2e-sota-run s5/s5b left claimed per instruction (sweep running), s1–s3 status lag logged as FINDINGS F3. Stale branches inventoried for Dennis in FINDINGS F5 — **nothing deleted**. Another session's in-flight files (`STATE.json` drift, chunk-validation scripts) left exactly as found (F4/F6). |
| 6 | Publishing + release | **verified, prepared-not-fired** | `uv build --package <name>` for every `packages/*/` at the final tree → **66 artifacts (33 wheels + 33 sdists), 0 failures** (twice: pre- and post-changes). Built wheels install and run in a fresh venv (area 3). Changelog discipline: `[Unreleased]` now carries the whole enterprise-serving wave (Added + Fixed). RC summary below. **No tag created, nothing published, release.yml untouched** — PyPI publishing remains hard-gated off by design. |
| 7 | CI | **fixed + verified** | Full gate present in `.github/workflows/ci.yml` (secret scan; `ruff check packages`; `scripts/typecheck.sh` mypy --strict src+tests all 33 packages; full pytest incl. publish-readiness + doc-truthfulness + conformance; 9-stack validate; build-all) with `PYTHONHASHSEED: "0"` pinned in ci/ci-heavy/release. Commands run locally verbatim: ruff clean; all 9 `latence stack validate` OK; build loop 0 fails; mkdocs strict exit 0. **Found CI's mypy gate RED** (55 errors across 5 packages, reproduced in a CI-identical env: clean clone + `uv sync` + the heavy-source `--no-deps` list from `.github/actions/setup-workspace`); fixed annotation-only in 7 test files → `STRICT TYPECHECK PASSED for all 33 package(s)`, and the full CI-env pytest run is green, exit 0. |

## The full gate (final tree)

- **CI-locked env** (clean clone, CI-identical setup, commit `1101395`):
  `PYTHONHASHSEED=0 uv run pytest -q` → **exit 0 — `5139 passed, 36 skipped, 4 warnings in
  623.41s`** (summary line from the `--override-ini addopts=` run, since the repo's `addopts =
  "-q"` double-quiets it away). `MYPY_CMD="uv run mypy" bash scripts/typecheck.sh`
  → `STRICT TYPECHECK PASSED for all 33 package(s) (src + tests)`, exit 0.
  `uv run ruff check packages` → All checks passed. `uv run mkdocs build --strict` → exit 0.
- **Host roster env** (this Mac's venv with every heavy provider installed):
  `PYTHONHASHSEED=0 .venv/bin/python -m pytest packages -q` → one failure:
  `test_real_weights_embeds_at_declared_width` (order-dependent torch-state pollution — passes
  standalone and in every targeted combination tried; FINDINGS F7), then the known
  native-teardown segfault (exit 139, post-test-phase — the ticket-03 pattern). Before this
  ticket the same run had **five** failures: the other four were real test-honesty defects
  (embedder-st asserting the host venv instead of pinning the torch-free contract; the ULTRA
  faithful-fake never modelling device moves) and are fixed and green.
- Repo-wide lint: `ruff check .` → 0 errors on the tracked tree (7 live stragglers fixed;
  frozen evidence scripts declared excluded with rationale in pyproject; the only residue is
  another session's untracked WIP, F6).
- `mypy --strict` on everything touched: clean (part of the all-33 PASS above).

## Release-candidate summary (prepared, NOT fired — Dennis's explicit go required)

- **What**: all 33 distributions at `0.1.0`, versioned and released together.
- **Proven**: every package builds (66 artifacts, 0 fails); wheels install + run in a fresh
  venv; the production Docker image builds and self-smokes; docs build strict; the full CI
  gate is green in a CI-identical env.
- **One discrepancy to decide at cut time**: CHANGELOG carries a `[0.1.0] — 2026-07-14`
  section (whose link targets a `v0.1.0` release tag), but **no v0.1.0 tag exists and nothing
  was ever published** — and `[Unreleased]` has since accumulated the enterprise-serving wave.
  Recommendation: fold `[Unreleased]` into the 0.1.0 section (re-dated) OR bump to 0.2.0,
  then `git tag v0.1.0 && git push origin v0.1.0` fires the release workflow (build → SBOM →
  Sigstore-signed → GitHub Release); PyPI stays off until the maintainer flips the gated job.
- **Not done here**: no tag, no push of a tag, no registry publish, no branch deletion.

## Findings (logged, not chased)

F1 graph sources re-read parquet per call · F2 goldset-property test costs ~288 s CPU ·
F3 e2e-sota-run ticket statuses lag their stages · F4 local campaign STATE.json is a stale
parallel-session snapshot · F5 stale-branch inventory for Dennis (31 + archive/*) ·
F6 chunk-validation untracked WIP · F7 the order-dependent real-weights flake on roster hosts.
