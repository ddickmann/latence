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

1. runs `uv export --frozen` to resolve `uv.lock` into a fully-pinned, hashed
   requirements list, then
2. runs `cyclonedx-py requirements` (from `cyclonedx-bom`) to emit a **CycloneDX
   1.6** SBOM in JSON.

Both tools run via `uv run --with … --no-project`, so nothing is added to the
workspace's own dependency set.

## Where it fits (and where it deliberately does not)

- **Not a CI merge gate.** The script is intentionally **not** wired into
  `.github/workflows/ci.yml` or `scripts/verify-local.sh`. Generating an SBOM does
  not gate a merge; it is a **release-time** artifact.
- **Release usage.** The release workstream calls `scripts/generate-sbom.sh` and
  attaches the resulting `sbom.cdx.json` to the GitHub Release (and/or publishes it
  alongside the built wheels). A release workflow can shell out to this same script.
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
