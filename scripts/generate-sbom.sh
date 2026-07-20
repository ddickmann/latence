#!/usr/bin/env bash
# Generate a CycloneDX SBOM (Software Bill of Materials) from the uv lockfile.
#
# This is a RELEASE-time tool, intentionally NOT wired into CI (ci.yml) or
# scripts/verify-local.sh — the release workstream calls it to attach an SBOM to a
# release, and any workflow that wants it can shell out to this script. It has no
# effect on the merge gate.
#
# What it does:
#   1. `uv export --all-packages` resolves the workspace lockfile (uv.lock) to a
#      fully-pinned, hashed requirements list covering the root dev project AND every
#      workspace member (no editable installs).
#   2. It fails closed if that export does not cover every package in `packages/`.
#   3. `cyclonedx-py requirements` converts that into a CycloneDX 1.6 SBOM.
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
# --all-packages: resolve EVERY workspace member, not just the root `latence-framework-dev`
#   meta-project. This is load-bearing, not a nicety: the root project deliberately does not
#   depend on the heavy Provider packages (ADR-0016, thin core), so without this flag the
#   export covers 12 of the 32 distributions the release ships and the SBOM silently omits
#   torch, transformers, gliner, openai, presidio-analyzer, sentence-transformers and the
#   rest — while still being stamped, checksummed and Sigstore-signed as evidence.
uv export \
  --quiet \
  --frozen \
  --no-editable \
  --no-emit-project \
  --all-packages \
  --format requirements-txt \
  -o "$REQ"

echo "==> Checking the export covers every workspace package"
# Fail closed BEFORE writing an SBOM. `uv export --no-emit-project` leaves one
# `./packages/<name>` path requirement per resolved workspace member, so the export's own
# coverage is directly observable. If it covers fewer packages than `packages/` contains,
# the resolution scope is wrong (a dropped --all-packages, a member missing from the
# workspace globs) and the document would understate the release's dependency surface.
# An SBOM nobody can answer "am I affected by CVE-X?" from is worse than none, because it
# looks like evidence.
REQ="$REQ" python3 - <<'PY'
import os
import pathlib
import re
import sys

req = pathlib.Path(os.environ["REQ"]).read_text(encoding="utf-8")
exported = {
    m.group(1) for m in re.finditer(r"^\./packages/([^\s;\\]+)", req, flags=re.MULTILINE)
}
tree = {p.parent.name for p in pathlib.Path("packages").glob("*/pyproject.toml")}

missing = sorted(tree - exported)
if missing:
    sys.exit(
        "generate-sbom: the uv export covers "
        f"{len(exported)} of the {len(tree)} workspace packages — an SBOM built from it "
        "would omit the dependencies of the distributions the release ships. "
        f"Missing: {', '.join(missing)}"
    )
print(f"    {len(exported)} workspace packages covered")
PY

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
