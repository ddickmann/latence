# 04 — Enterprise readiness pass: docs, deployment, release, CI — the final push

Type: task
Status: open
Blocked by: 01, 02, 03

## Why (Dennis, 2026-08-12)

"The goal is when I raise the enterprise readiness question that you cannot answer like
always with 'here are a few caveats to consider'. I don't want to hear that anymore."
After 01–03 close the technical caveats, this pass makes the repo itself releasable.

## Scope (audit → fix → verify, each area)

1. **README**: current, honest, benchmark numbers from the completed S5 sweep (full-query,
   with the published-baseline comparison framing), quickstart that actually works from a
   clean clone (verified by running it).
2. **Docs**: the mkdocs site builds clean; guides cover the full journey (ingest → pipeline →
   graph → retrieval → serving with Qdrant → incremental churn → campaign wizard); every
   guide's claims match shipped behavior (the doc-truthfulness gates extended to the new
   surfaces).
3. **Deployment on the stacks**: reproducible deployment paths documented AND smoke-verified —
   local (uv/pip), Docker (compose incl. Qdrant from ticket 02), and the cloud-agnostic story
   (what runs where, GPU vs CPU requirements per stage). Airflow runner package: state its
   real status honestly (exercised or explicitly experimental).
4. **Licenses**: root LICENSE + the seven-file Apache-2.0 exception set intact
   (test_publish_readiness green), per-package license metadata correct, third-party model/
   dataset license lanes documented in one place, NOTICE hygiene.
5. **Repo hygiene**: no stray artifacts/dead files in tracked tree, .gitignore complete
   (derived benchmark data stays local), issues/ tickets statuses current, stale branches
   noted for Dennis (never deleted unilaterally).
6. **Publishing + release**: package build (`uv build` / wheel) succeeds for every
   publishable package; version + changelog discipline; a tagged release candidate prepared
   (tag pushed only with Dennis's approval).
7. **CI**: the full gate (tests all packages, ruff, mypy --strict, publish-readiness,
   doc-truthfulness) runs green in CI config, not just locally; PYTHONHASHSEED pinned;
   offline-CI fakes cover what live deps cover.

## Non-negotiables

- Every "verified" claim = actually executed here (quickstart run, docker build run, docs
  build run). No aspirational checkmarks.
- Nothing outward-facing (publishing to registries, tagging public releases, deleting
  branches) happens without Dennis's explicit go — prepared, not fired.
- Findings that are out of scope get logged in the campaign FINDINGS pattern, not chased.

## Gate artifact

A readiness report: per-area status table (green/fixed/deferred-with-reason), the exact
commands run as proof, and the release-candidate summary. Dennis inspects → approves the
architecture pass (byte-identical gate) and the S5 re-run.
