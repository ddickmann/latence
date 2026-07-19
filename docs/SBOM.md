# Software Bill of Materials (SBOM)

A Software Bill of Materials is a machine-readable inventory of every dependency
that ships in (or is resolved for) the framework. It lets a consumer answer "am I
affected by CVE-X?" without guessing, and is increasingly expected of any software
that processes untrusted enterprise data (see [`docs/THREAT-MODEL.md`](THREAT-MODEL.md)
→ *Boundary 6 — Supply chain*).

The single source of truth for versions is the committed **`uv.lock`** (the uv
workspace lockfile). The SBOM is derived from it, so it always matches what a
`uv sync` installs — it is never hand-maintained.

## Generating an SBOM

```bash
bash scripts/generate-sbom.sh            # writes sbom.cdx.json (CycloneDX JSON)
bash scripts/generate-sbom.sh out.json   # custom output path
```

The script (`scripts/generate-sbom.sh`):

1. runs `uv export --frozen --all-packages` to resolve `uv.lock` into a fully-pinned,
   hashed requirements list covering **every workspace package**, not just the root dev
   project, then
2. checks that the export really covers every package in `packages/` — and **fails
   closed** if it does not, before any SBOM is written (see *Scope* below), then
3. runs `cyclonedx-py requirements` (from `cyclonedx-bom`) to emit a **CycloneDX
   1.6** SBOM in JSON.

## Scope — why `--all-packages` is load-bearing

The root `latence-framework-dev` project deliberately does **not** depend on the heavy
Provider packages (ADR-0016, thin core). So a plain `uv export` resolves only that
meta-project and yields a requirements list covering a fraction of what a release ships:
on this tree, 12 of the 32 distributions, with `torch`, `transformers`, `gliner`,
`openai`, `presidio-analyzer`, `sentence-transformers`, `peft`, `huggingface-hub` and
others absent — even though they are **required** runtime dependencies of released
packages. An adopter scanning that document with `grype` or Dependency-Track would get a
clean report for a CVE they are in fact exposed to.

`--all-packages` fixes the scope; the coverage check makes the omission impossible to
reintroduce silently. `uv export --no-emit-project` leaves one `./packages/<name>` path
requirement per resolved member, so the script compares that set against
`packages/*/pyproject.toml` and aborts on any shortfall.

Both tools run via `uv run --with … --no-project`, so nothing is added to the
workspace's own dependency set.

## Where it fits (and where it deliberately does not)

- **Not a CI merge gate.** The script is intentionally **not** wired into
  `.github/workflows/ci.yml` or `scripts/verify-local.sh`. Generating an SBOM does
  not gate a merge; it is a **release-time** artifact.
- **Release usage — automatic.** `.github/workflows/release.yml` runs
  `scripts/release-integrity.sh` on every `v*` tag. That script calls
  `scripts/generate-sbom.sh`, **stamps** the resulting document with the release identity
  (`metadata.component` = `latence-framework` at the tag's version, plus
  `latence:release:tag` / `latence:release:repository` properties), and attaches it to the
  GitHub Release as `sbom.cdx.json` alongside the wheels. The SBOM's digest is recorded in
  the release's `SHA256SUMS`, which is signed with keyless Sigstore — so the SBOM is not
  just published, it is **authenticated**. See [`RELEASING.md`](RELEASING.md) →
  *Release integrity*.
- **Fail-closed.** A document that is not CycloneDX, has no `specVersion`, or lists **no
  components** is rejected rather than attached. An empty SBOM is worse than none: it looks
  like evidence while answering nothing.
- **Format.** CycloneDX JSON is emitted because it is the OWASP-standard,
  tool-friendly format (consumable by `grype`, Dependency-Track, GitHub's
  dependency graph, etc.). Add SPDX output later if a consumer needs it.

## Related supply-chain controls

- **Dependabot** (`.github/dependabot.yml`) — weekly `pip` + `github-actions`
  updates, grouped minor/patch; surfaces dependency advisories.
- **CodeQL** (`.github/workflows/codeql.yml`) — SAST over the Python sources.
- **OpenSSF Scorecard** (`.github/workflows/scorecard.yml`) — supply-chain posture
  scoring.
- **detect-secrets gate** — `scripts/secret_scan_gate.py`, run in CI and
  `scripts/verify-local.sh`.

Dependency-CVE **gating** (hard-failing a build on a known-vulnerable dependency)
remains a Phase-2 hardening item, tracked in `SECURITY.md`. The SBOM is the input
that makes such a gate possible.
