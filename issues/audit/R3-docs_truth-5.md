# R3-docs_truth-5 — README summarises THIRD-PARTY-LICENSES.md as "all permissive" after this stream added ~40 proprietary CUDA wheels to it

**Severity:** major
**Round:** R3 · lane `docs_truth`
**Site:** `README.md:188` (vs `THIRD-PARTY-LICENSES.md:14,161,194-240`, `docs/CONTINUITY.md:53-58`)
**Relates to:** #189 AC-2 ("nothing published may be false on day one"), #196, #198, ADR-0012

## Failure scenario

`git show main:THIRD-PARTY-LICENSES.md | grep -c -i nvidia` returns `0`. On
`feat/release-fleet-integration` the same file gained 192 lines and now opens (line 14) with
"NVIDIA CUDA wheels that are **not** open-source", tabulates ~40 wheels as
`LicenseRef-NVIDIA-Proprietary` / "proprietary (CUDA EULA)" (lines 206-240), and records
`google/embeddinggemma-300m` as "Not OSI open-source" carrying Google's Prohibited Use Policy plus
a flow-down obligation (line 161). `docs/CONTINUITY.md:53-58` — added by this same stream — names
"the one disclosed exception … the CUDA wheel set … which carry NVIDIA's proprietary EULAs".

`README.md:188` nevertheless still reads:

```
- [`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md) — runtime dep + model licenses (all permissive).
```

A procurement or legal reviewer who reads only the README's front-door summary concludes the
dependency tree is licence-clean and skips the EULA review the repo's own disclosure file says is
required. The stream deliberately swept the surrounding README lines (ADR count 32→52, the
Runner-adapter overclaim, the meta-package line) while leaving this one.

No test guards it: `packages/latence-core/tests/test_front_door_truthfulness.py` checks the ADR
count, Runner adapters, dangling links and install lines but makes no licence assertion, and
`test_continuity_statement.py` only checks that cited paths exist, never their content.

## What to build

1. Correct `README.md:188` so the front-door summary matches the disclosure file: permissive **by
   default**, with the two disclosed exceptions named (the NVIDIA CUDA wheels under the pinned GPU
   environments, and opt-in EmbeddingGemma).
2. Add a general regression guard to `test_front_door_truthfulness.py`: no published page may
   summarise `THIRD-PARTY-LICENSES.md` with a blanket permissiveness claim while that file itself
   discloses non-permissive entries. The guard must derive its ground truth from
   `THIRD-PARTY-LICENSES.md`, not from a hard-coded string, so a *future* proprietary addition
   re-arms it automatically.

## Acceptance criteria

- **AC-1** — No front-door page (README, root governance pages, every tracked `docs/**/*.md`)
  contains a block that both references `THIRD-PARTY-LICENSES.md` and makes an unqualified
  "all permissive" / "permissive only" claim, unless that same block discloses the carve-out
  (proprietary / EULA / NVIDIA / CUDA / exception / opt-in / restricted).
- **AC-2** — The guard is non-vacuous: it asserts up front that `THIRD-PARTY-LICENSES.md` really
  does disclose non-permissive entries, so it cannot pass by finding nothing to check. It fails on
  the pre-fix `README.md:188` and passes after.
- **AC-3** — The README still *points at* the licence file (the claim is corrected, not deleted).
- **AC-4** — `make test` stays green at or above 2985 passed, 28 skipped.
