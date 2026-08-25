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

echo "==> Folding in the workspace-EXCLUDED packages' own locks"
# ADR-0064. A uv workspace is ONE resolved environment, so a package pinning a dependency major
# disjoint from its siblings' cannot be a member — `uv lock` refuses the set, correctly, and
# `latence-gliner25` (gliner2 2.x, against the other Providers' 1.x) is such a package. It is
# still BUILT and PUBLISHED by the release, so leaving it out of the SBOM would be exactly the
# silent false negative the coverage check below exists to prevent: an adopter scanning for
# "am I affected by CVE-X?" would never see its dependency surface.
#
# So each excluded package carries its OWN committed `uv.lock` and is exported from it here, with
# its `./packages/<name>` path line synthesised so it is identified the same way a member is.
# `--frozen` keeps this offline. Requirements already present (same name AND version) are dropped
# so the union does not double-count the large shared surface; a DIFFERENT resolved version of the
# same distribution is kept, because that is a genuinely different component the release ships.
# Fails closed if an excluded package has no lock of its own — an unexported exclusion is an
# undisclosed gap, which is the thing being fixed.
python3 - "$REQ" <<'PY'
import pathlib
import re
import subprocess
import sys
import tomllib

req_path = pathlib.Path(sys.argv[1])

with pathlib.Path("pyproject.toml").open("rb") as handle:
    workspace = tomllib.load(handle).get("tool", {}).get("uv", {}).get("workspace", {})
excluded = sorted(workspace.get("exclude", []))
if not excluded:
    print("    no workspace exclusions")
    raise SystemExit(0)

_PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s\\]+)")


def pins(text: str) -> set[tuple[str, str]]:
    return {
        (m.group(1).lower().replace("_", "-"), m.group(2))
        for m in (_PIN.match(line) for line in text.splitlines())
        if m
    }


existing = pins(req_path.read_text(encoding="utf-8"))
extra: list[str] = []
for rel in excluded:
    pkg = pathlib.Path(rel)
    if not (pkg / "pyproject.toml").is_file():
        continue
    if not (pkg / "uv.lock").is_file():
        sys.exit(
            f"generate-sbom: {rel} is excluded from the uv workspace but has no uv.lock of its "
            "own, so its dependencies cannot be exported — the release would ship a distribution "
            "whose surface this SBOM does not describe. Run: uv lock --directory " + rel
        )
    exported = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            "uv", "export", "--quiet", "--frozen", "--no-editable", "--no-emit-project",
            "--directory", rel, "--format", "requirements-txt",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    keep: list[str] = []
    skip_block = False
    for line in exported.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = _PIN.match(line)
        if match:
            skip_block = (match.group(1).lower().replace("_", "-"), match.group(2)) in existing
            if not skip_block:
                existing.add((match.group(1).lower().replace("_", "-"), match.group(2)))
        elif line.startswith(("../", "./")):
            # A sibling workspace path the ROOT export already emits under its own name.
            skip_block = True
        if not skip_block:
            keep.append(line)
    # The path line the coverage checks identify a shipped distribution by, written the same way
    # `uv export --no-emit-project` writes a member's.
    extra.append(f"./{pkg.as_posix()}")
    extra.extend(keep)
    print(f"    folded in {rel} ({sum(1 for line in keep if _PIN.match(line))} new pins)")

with req_path.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(extra) + "\n")
PY

echo "==> Checking the export covers every workspace package"
# Fail closed BEFORE writing an SBOM. `uv export --no-emit-project` leaves one
# `./packages/<name>` path requirement per resolved workspace member, so the export's own
# coverage is directly observable. If it covers fewer packages than the RESOLVED WORKSPACE
# contains, the resolution scope is wrong (a dropped --all-packages, a member missing from the
# workspace globs) and the document would understate the release's dependency surface.
# An SBOM nobody can answer "am I affected by CVE-X?" from is worse than none, because it
# looks like evidence.
#
# "Resolved workspace" is `packages/*` MINUS `[tool.uv.workspace] exclude` (ADR-0064). A uv
# workspace is one resolved environment, and `latence-gliner25` pins a `gliner2` major disjoint
# from the one the other Provider packages pin, so it cannot be a member — `uv lock` correctly
# refuses. It still SHIPS, so it is a real, stated gap in this document rather than a silent one:
# the check subtracts only DECLARED exclusions (an undeclared missing package still fails closed)
# and prints every excluded distribution so the gap appears in the release log.
REQ="$REQ" python3 - <<'PY'
import os
import pathlib
import re
import sys
import tomllib

req = pathlib.Path(os.environ["REQ"]).read_text(encoding="utf-8")
exported = {
    m.group(1) for m in re.finditer(r"^\./packages/([^\s;\\]+)", req, flags=re.MULTILINE)
}
tree = {p.parent.name for p in pathlib.Path("packages").glob("*/pyproject.toml")}

with pathlib.Path("pyproject.toml").open("rb") as handle:
    workspace = tomllib.load(handle).get("tool", {}).get("uv", {}).get("workspace", {})
excluded = {pathlib.PurePosixPath(e).name for e in workspace.get("exclude", [])}

missing = sorted(tree - exported)
if missing:
    sys.exit(
        "generate-sbom: the uv export covers "
        f"{len(exported)} of the {len(tree)} workspace packages the release ships — an SBOM "
        "built from it would omit the dependencies of the distributions the release ships. "
        f"Missing: {', '.join(missing)}"
    )
print(f"    {len(exported)} workspace packages covered")
for name in sorted(excluded & tree):
    print(f"    (via its own uv.lock, outside the workspace: {name})")
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
