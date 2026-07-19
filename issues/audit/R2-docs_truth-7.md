# R2 — docs/RELEASING.md's "Last verified local dry run" evidence block describes a tree that never existed

- **Severity:** major
- **Dimension:** docs-truth / anti-false-green (the only cited end-to-end proof of the release+SBOM machinery)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), issue #178 (SBOM + release integrity)
- **Status:** fixed

## What is wrong

`docs/RELEASING.md` closes the *Rehearsing a release (dry run)* section with an evidence
block presented as a **measurement**:

> **Last verified local dry run** (tag `v0.1.0`, all 25 packages): 56 distributions built,
> a 45-component CycloneDX 1.6 SBOM generated from `uv.lock` and stamped as
> `latence-framework 0.1.0`, 58 assets checksummed, and `shasum -a 256 -c SHA256SUMS`
> verified all 58.

Every number in it is wrong for the current tree, and the package count was wrong when it
was written:

| claim | reality (measured on this branch) |
| --- | --- |
| "all 25 packages" | `ls -d packages/*/` = **32**; `.github/workflows/release.yml:68-73` builds `for pkg in packages/*/`, i.e. all of them |
| "56 distributions built" | **64** — all 32 packages declare `build-backend = "hatchling.build"`, so `uv build` emits one sdist + one wheel each |
| "58 assets checksummed" | **66** — the 64 distributions plus `sbom.cdx.json` and `VERIFY.md`, which `scripts/release-integrity.sh` stages before writing `SHA256SUMS` |
| "45-component … SBOM" | **66 components** — `uv export --frozen --no-editable --no-emit-project` now yields 54 pinned third-party requirements plus 12 `./packages/...` path lines |

"25" was already wrong at authoring time: commit `891fa2d` (the commit that added the block)
had **28** package directories, and 28 × 2 = 56 exactly reproduces the quoted distribution
count — so the prose count and the measured count disagreed in the same sentence.

Nothing guards these figures. Worse,
`packages/latence-core/tests/test_release_integrity.py:370-374` explicitly *defers* to this
block:

> These are static guards, and deliberately so: actually running generate-sbom.sh needs the
> network […] **The end-to-end proof is the release dry run documented in
> docs/RELEASING.md.**

So the SBOM generator's only claimed end-to-end evidence is a block of unreproducible
numbers.

## Failure scenario

A maintainer or an external auditor runs the documented rehearsal verbatim:

```bash
for pkg in packages/*/; do uv build --package "$(basename "$pkg")" --out-dir dist; done
bash scripts/release-integrity.sh --tag v0.1.0 --dist dist --out release-assets --no-sign
(cd release-assets && shasum -a 256 -c SHA256SUMS)
```

and gets 32 packages / 64 distributions / 66 checksummed assets / a 66-component SBOM,
against a document asserting 25 / 56 / 58 / 45 as a *measured* result. The document's only
end-to-end evidence that the release and SBOM machinery works is therefore unreproducible,
which means a **real** regression — a package silently failing to build, or dropping out of
the `packages/*/` loop — is indistinguishable from documented drift. The reviewer's only
options are to disbelieve the doc or to disbelieve the run; neither yields a merge gate.

This is the same class of failure the block's own footnote warns about ("Release tooling
that is only exercised at release time must pin its tools"): evidence that is only produced
once and never re-checked stops being evidence.

## What to build

1. **Re-run the rehearsal for real** and replace the block with what was actually measured.
2. **Make the counts derivable, and machine-check them.** The three counts that drifted are
   pure functions of the tree — packages = `packages/*/pyproject.toml`, distributions =
   2 × packages (every package is hatchling → sdist + wheel), assets = distributions + 2
   (`sbom.cdx.json` + `VERIFY.md`). State the derivation next to each number and add a
   regression test that recomputes all three from the tree and fails the merge gate when the
   document and the tree disagree — including when a *new package is added*, which is
   precisely how the block rotted.
3. **Stop the deferral from being load-bearing on unguarded prose.** Update the comment at
   `test_release_integrity.py:370-374` so it points at a block that is now enforced.

## Out of scope (recorded, not fixed here)

The generated SBOM contains **12 components named `unknown`** (e.g. `bom-ref:
requirements-L20`, `description: "requirements line 20: ./packages/latence-demo"`). These are
the workspace path requirements that `uv export --no-emit-project` does not strip and that
`cyclonedx-py requirements` cannot resolve to a name/version. They are inert for CVE
matching — the usable third-party surface is the 54 pinned components — but they pad the
component count and would confuse a Dependency-Track import. Fixing that is a change to
`scripts/generate-sbom.sh`, not to this document, and belongs to its own ticket.

## Acceptance criteria

- [x] `docs/RELEASING.md`'s evidence block states counts that a verbatim re-run of the
      documented commands reproduces on the current tree (32 / 64 / 66, SBOM stamped
      `latence-framework 0.1.0`, all 66 verified by `shasum -a 256 -c SHA256SUMS`).
- [x] Each count is accompanied by the derivation that makes it checkable, not asserted bare.
- [x] A regression test recomputes packages / distributions / assets **from the tree** and
      compares them to the numbers written in the document; it **fails on the pre-fix
      document** and passes on the fixed one.
- [x] The same test fails if a package is added to or removed from `packages/` without the
      document being updated, and if a package stops using hatchling (which would break the
      "2 distributions per package" derivation).
- [x] The internal arithmetic of the SBOM sentence is checked (total components =
      third-party + workspace path entries).
- [x] The deferral comment in `test_release_integrity.py` no longer points at unguarded prose.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped; `bash scripts/verify-local.sh`
      passes (docs touched).
