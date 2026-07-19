# R4 — the signed release SBOM omits the dependencies of 18 of the 32 distributions the release ships

- **Severity:** major
- **Dimension:** security / supply chain (SBOM authenticity)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178 (signed release + per-release SBOM); `docs/SBOM.md`, `docs/RELEASING.md`
- **Status:** fixed

## What is wrong

`scripts/generate-sbom.sh` resolves the workspace lockfile with

```
uv export --quiet --frozen --no-editable --no-emit-project --format requirements-txt
```

Without `--all-packages`, `uv export` resolves only the **root** `latence-framework-dev`
meta-project. That project's dependency list deliberately excludes every heavy Provider
package (ADR-0016, thin core), so the resulting requirements file — and therefore the
CycloneDX document built from it — covers a small subset of what the release actually
ships.

Measured read-only on this tree (same `uv.lock`, no re-resolution):

| export | pinned third-party requirements | `./packages/...` path entries |
| --- | --- | --- |
| as shipped (no `--all-packages`) | 54 | 12 |
| with `--all-packages` | 147 | 32 |

The release builds and publishes **32** distributions (`docs/RELEASING.md`, "packages
built: 32"), but the SBOM only ever saw 12 of them. `torch`, `transformers`, `gliner`,
`gliner2`, `openai`, `presidio-analyzer`, `sentence-transformers`, `peft`,
`huggingface-hub`, `glinker`, `flashdeberta`, `python-docx`, `openpyxl` and
`python-pptx` — all **required** (not optional-extra) runtime dependencies of released
packages — appear nowhere in the document.

That document is nonetheless stamped `latence-framework@<version>`, digest-recorded in
`SHA256SUMS`, and Sigstore-signed: it is published as *authenticated* supply-chain
evidence. The script's own header comment claimed the export covers "dev + all workspace
members", which was false.

Nothing caught it:

- `scripts/release-integrity.sh`'s fail-closed guard rejects only a **zero-component**
  SBOM, so 66 components passes trivially.
- `test_releasing_sbom_component_arithmetic_is_consistent` only asserts `54 + 12 == 66`;
  it never cross-checks the 12 workspace path entries against the 32 packages the very
  same evidence block records as built.

## Failure scenario

A CVE lands in `transformers` (or `torch` / `gliner` / `presidio-analyzer` / `openai`).
An adopter who installed the released `latence_ner_gliner`, `latence_pii_presidio`,
`latence_embedder_st`, `latence_parser_glm`, `latence_disambig_glinker`,
`latence_splade`, `latence_schema_inducer` or any of the three `-endpoint` wheels
downloads the release's signed `sbom.cdx.json` and runs `grype sbom:sbom.cdx.json`, or
ingests it into Dependency-Track — which is exactly what `docs/SBOM.md` instructs. It
reports **CLEAN**, because those packages' required dependencies are not in the document
at all. The adopter concludes they are unaffected and does not patch. This is precisely
the failure the script's own comment names: "an SBOM nobody can answer 'am I affected by
CVE-X?' from is worse than none, because it looks like evidence."

## What to build

1. Pass `--all-packages` to `uv export` in `scripts/generate-sbom.sh`, so the export
   resolves every workspace member, and correct the header comment.
2. Make the omission **impossible to reintroduce silently**: `generate-sbom.sh` itself
   fails closed when the exported requirements do not contain one `./packages/<name>`
   path entry for every `packages/*/pyproject.toml` in the workspace. A dropped flag
   then breaks the release run loudly instead of publishing a hollow SBOM.
3. Replace the hollow arithmetic guard with a real one: the SBOM evidence block's
   workspace-path-entry count must equal the package count derived from the tree, and
   the recorded third-party count must be the one the shipped flags actually produce.
4. Correct the counts and the `unknown`-components note in `docs/RELEASING.md`, and
   state the workspace-wide scope in `docs/SBOM.md`.

## Acceptance criteria

1. `scripts/generate-sbom.sh` invokes `uv export` with `--all-packages`.
2. Running the exact export the script performs yields a requirement set that contains
   every **required** third-party dependency declared in every `packages/*/pyproject.toml`
   `project.dependencies` — including `torch`, `transformers`, `gliner`, `openai`,
   `presidio-analyzer`, `sentence-transformers`.
3. `generate-sbom.sh` exits non-zero, before writing any SBOM, if the export covers fewer
   workspace packages than the tree contains.
4. `docs/RELEASING.md`'s SBOM evidence block records the workspace path-entry count equal
   to the number of packages built (32), and a test derives that from the tree rather than
   only checking internal arithmetic.
5. No regression: `make test` stays green at or above 2985 passed / 28 skipped.
