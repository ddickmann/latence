#!/usr/bin/env bash
# Generate a CycloneDX SBOM (Software Bill of Materials) from the uv lockfile.
#
# This is a RELEASE-time tool, intentionally NOT wired into CI (ci.yml) or
# scripts/verify-local.sh — the release workstream calls it to attach an SBOM to a
# release, and any workflow that wants it can shell out to this script. It has no
# effect on the merge gate.
#
# What it does:
#   1. `uv export` resolves the workspace lockfile (uv.lock) to a fully-pinned,
#      hashed requirements list (dev + all workspace members, no editable installs).
#   2. `cyclonedx-py requirements` converts that into a CycloneDX 1.6 SBOM.
#
# Both tools are invoked via `uv run --with ... --no-project` so nothing is added to
# the workspace's own dependency set.
#
# Usage:
#   bash scripts/generate-sbom.sh [output-path]
# Default output: sbom.cdx.json (CycloneDX JSON) in the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-sbom.cdx.json}"
REQ="$(mktemp -t latence-sbom-requirements.XXXXXX.txt)"
trap 'rm -f "$REQ"' EXIT

echo "==> Exporting fully-pinned requirements from uv.lock"
# --frozen: do not re-resolve; use the committed lockfile exactly.
# --no-emit-project / --no-editable: describe installed third-party deps, not the
# local editable workspace packages themselves.
uv export \
  --quiet \
  --frozen \
  --no-editable \
  --no-emit-project \
  --format requirements-txt \
  -o "$REQ"

echo "==> Building CycloneDX SBOM -> $OUT"
# cyclonedx-bom is PINNED to a major range on purpose: this is release machinery, and an
# unpinned tool silently changing its CLI is exactly how a release stops producing an SBOM.
# (It did: `--outfile` was removed in favour of `-o`/`--output-file`.)
#
# --spec-version 1.6      the version this repo documents (docs/SBOM.md); the tool's default
#                         floats forward, which would silently change the emitted schema.
# --output-reproducible   drop the wall-clock timestamp + random serial number, so the same
#                         uv.lock always yields byte-identical output. Required here: the
#                         SBOM's digest is recorded in the release's signed SHA256SUMS, so a
#                         non-reproducible SBOM makes the release unreproducible too.
uv run --with 'cyclonedx-bom>=7,<8' --no-project -- \
  cyclonedx-py requirements "$REQ" \
  --output-format JSON \
  --spec-version 1.6 \
  --output-reproducible \
  --output-file "$OUT"

echo "==> SBOM written: $OUT"
