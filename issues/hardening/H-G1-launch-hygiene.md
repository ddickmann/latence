# H-G1 — OSS launch hygiene: provider namespace, hardcoded paths, pins

**Source: hostile 360 audit, THEME G (S3/S4).** Small, mechanical pre-publish cleanup. Two THEME-G
items are already resolved and OUT of scope here: the NOTICE/THIRD-PARTY EmbeddingGemma overclaim
(fixed in H-B1 #104) and the 43 stray merged remote branches (already pruned; `debt-mypy` intentionally
kept — a superseded no-merged-PR branch left for the maintainer to review).

## 1. Unify the Parse provider namespace (`parse.*` → `parser.*`)
The Parse capability has inconsistent provider IDs: `parser.plaintext`, `parser.document`,
`parser.pdfplumber` (all `parser.*`) but `parse.endpoint` (the odd one out). The audit flagged this
breaks the headline "swap one line" story — an adopter flipping `parser.document` → `parser.endpoint`
gets a `KeyError`.
- **Rename `parse.endpoint` → `parser.endpoint`** in `packages/latence-parser-endpoint/pyproject.toml`
  (the `latence.providers` entry point), and update EVERY reference: the package README, any example
  YAML/pipeline config, docstrings, and tests that name `parse.endpoint`.
- Grep the whole repo for `parse.endpoint` and `"parse\.` to catch stragglers; after the change the
  only Parse IDs are `parser.*`. The registry keys on the exact entry-point name, so this is a clean
  rename (no compat shim needed — nothing is published yet). Confirm the conformance suite + any
  endpoint tests still pass with the new name.

## 2. Parameterize hardcoded maintainer paths
Tracked files embed the maintainer's home directory (`/Users/dennisdickmann/...`), which is
non-portable and leaks a local path into a public repo:
- `orchestration/*.workflow.js` — the `repo`/`worktree` **default** constants (e.g.
  `const repo = '/Users/dennisdickmann/Developer/latence-framework'`). Replace the hardcoded absolute
  defaults with a portable derivation: default `repo` to the git toplevel (e.g. derive from the
  script's own location / `process.cwd()` / a `LATENCE_REPO` env var), and default `worktree` relative
  to it. These are Claude-workflow harness scripts (dev tooling); keep them working when driven by
  args (as they are today) but make the *defaults* portable, not machine-specific.
- `issues/gate/G1-RESULTS.md` — replace the absolute `/Users/dennisdickmann/...` path with a
  repo-relative path or a `<REPO_ROOT>` placeholder.
- Grep the tracked tree for `/Users/dennisdickmann` and `/Users/` and fix every hit that ships in the
  repo (source, docs, configs, orchestration). Do NOT touch untracked/scratch files.

## 3. Dependency pins on fast-moving model libs (optional, low-risk)
The packaging audit noted lower-bound-only pins on libraries whose APIs move across majors
(`gliner`, `openai`). Add conservative upper caps (e.g. `openai>=1.0,<2`, `gliner>=0.2.5,<0.3` —
use the actual current-major boundary) on the model-lib deps in the provider packages that import
them, so a downstream install can't silently pull an API-breaking major. Leave the stable libs
(pydantic/pyarrow/fsspec) as-is (standard lower-bound library practice). If a cap would be guesswork
about the next major's compatibility, note it and skip rather than pin arbitrarily.

## Definition of Done (Baseline bar + these)
- `parser.endpoint` replaces `parse.endpoint` everywhere; repo grep for `parse.endpoint` / `"parse\.`
  is clean; endpoint + conformance tests green with the new ID; the swap-a-provider docs are correct.
- No tracked file contains `/Users/dennisdickmann`; the orchestration workflow defaults derive a
  portable repo root and still run when given args; `G1-RESULTS.md` uses a portable path.
- Model-lib deps carry sensible upper caps (or a documented skip); no stable-lib over-pinning.
- `mypy --strict`/`ruff`/pytest green; the workflow scripts remain valid JS (a quick `node --check`
  or equivalent on the edited `.workflow.js` files).

## Reviewer should check
- The `parse.endpoint`→`parser.endpoint` rename is COMPLETE (no dangling old ID in any config/doc/test)
  and the provider still loads + passes conformance under the new name.
- No `/Users/` path remains in any tracked file; the orchestration defaults are genuinely portable
  (would work on a fresh clone in a different location) yet unchanged when driven by explicit args.
- Any added dep cap matches the library's real current-major boundary (not an arbitrary guess).
