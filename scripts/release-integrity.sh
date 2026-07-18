#!/usr/bin/env bash
# Release integrity: assemble the signed, SBOM-carrying asset set for one release tag.
#
# One command turns "a directory of built wheels/sdists" into "the exact set of files a
# GitHub Release should carry, plus everything an adopter needs to verify them":
#
#   <out>/                      <- this IS the release asset set; upload the whole directory
#     latence_*.whl / *.tar.gz  <- the built distributions, staged flat
#     sbom.cdx.json             <- per-release CycloneDX SBOM, stamped with the release version
#     SHA256SUMS                <- digest of every file above (bare filenames, flat)
#     SHA256SUMS.sigstore.json  <- keyless Sigstore bundle signing SHA256SUMS  (--sign)
#     VERIFY.md                 <- the copy-pasteable verification instructions
#
# Why only SHA256SUMS is signed: it covers every other asset by digest, so ONE signature
# transitively authenticates the whole release. An adopter runs two commands (see VERIFY.md).
#
# Signing is keyless Sigstore (sigstore-python): the signing identity is the GitHub Actions
# workflow itself, certified by Fulcio and logged in Rekor. There is no long-lived private
# key to store, rotate, or leak. That also means a signature can ONLY be produced where a
# Sigstore-trusted OIDC identity exists — in practice, inside the release workflow. A local
# dry run therefore uses --no-sign and produces everything except the bundle.
#
# Usage:
#   bash scripts/release-integrity.sh --tag vX.Y.Z --dist dist --sign
#   bash scripts/release-integrity.sh --tag vX.Y.Z --dist dist --no-sign        # dry run
#   bash scripts/release-integrity.sh --tag vX.Y.Z --dist dist --sbom s.json --no-sign
#
# Options:
#   --tag vX.Y.Z        Release tag (required). Must be v + SemVer.
#   --dist DIR          Directory of built distributions (default: dist).
#   --out DIR           Output/asset directory (default: release-integrity). Recreated.
#   --sbom PATH         Use this CycloneDX document instead of generating one (skips uv+network).
#   --sign / --no-sign  Required, explicit. There is deliberately no default and no silent
#                       skip: an unsigned release must be an explicit choice, never an accident
#                       caused by a missing OIDC permission.
#   --identity-token T  Pass an explicit OIDC token to sigstore (otherwise ambient detection).
#   --repo OWNER/NAME   Repo slug used in the verification identity (default: ddickmann/latence-framework).
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=""
DIST="dist"
OUT="release-integrity"
SBOM_IN=""
SIGN=""
IDENTITY_TOKEN=""
REPO_SLUG="${GITHUB_REPOSITORY:-ddickmann/latence-framework}"

die() {
  echo "release-integrity: $*" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 ;;
    --dist) DIST="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --sbom) SBOM_IN="${2:-}"; shift 2 ;;
    --sign) SIGN="yes"; shift ;;
    --no-sign) SIGN="no"; shift ;;
    --identity-token) IDENTITY_TOKEN="${2:-}"; shift 2 ;;
    --repo) REPO_SLUG="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# ---- fail-closed input validation -------------------------------------------------------
[ -n "$TAG" ] || die "--tag is required (e.g. --tag v0.1.0)"
# v + SemVer, optional pre-release/build suffix. A malformed tag would silently produce an
# SBOM stamped with a version that matches no release, so reject it here.
if ! printf '%s' "$TAG" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$'; then
  die "tag '$TAG' is not a v-prefixed SemVer tag (expected e.g. v0.1.0, v1.2.3-rc.1)"
fi
VERSION="${TAG#v}"

[ -n "$SIGN" ] || die "pass exactly one of --sign or --no-sign (no default: an unsigned release must be deliberate)"
[ -d "$DIST" ] || die "--dist directory not found: $DIST"

# Collect the distributions. An empty dist is always a build failure, never a valid release.
DIST_FILES=()
while IFS= read -r f; do DIST_FILES+=("$f"); done < <(find "$DIST" -type f \( -name '*.whl' -o -name '*.tar.gz' \) | sort)
[ "${#DIST_FILES[@]}" -gt 0 ] || die "no *.whl or *.tar.gz found in $DIST — nothing to release"

# Assets are staged flat, so two distributions with the same basename would silently
# overwrite each other and the release would ship one of them unrecorded.
DUPES="$(for f in "${DIST_FILES[@]}"; do basename "$f"; done | sort | uniq -d)"
[ -z "$DUPES" ] || die "duplicate distribution filenames in $DIST: $(echo "$DUPES" | tr '\n' ' ')"

rm -rf "$OUT"
mkdir -p "$OUT"

# ---- 1. SBOM ----------------------------------------------------------------------------
SBOM="$OUT/sbom.cdx.json"
if [ -n "$SBOM_IN" ]; then
  [ -f "$SBOM_IN" ] || die "--sbom file not found: $SBOM_IN"
  cp "$SBOM_IN" "$SBOM"
  echo "==> Using provided CycloneDX document: $SBOM_IN"
else
  echo "==> Generating CycloneDX SBOM from uv.lock"
  bash scripts/generate-sbom.sh "$SBOM"
fi

echo "==> Stamping SBOM with release $TAG"
# Validate that this really is a CycloneDX document with a resolved dependency set, then
# stamp the release identity onto it. A structurally wrong or empty SBOM is rejected rather
# than attached: an SBOM nobody can answer "am I affected by CVE-X?" from is worse than none,
# because it looks like evidence.
SBOM="$SBOM" VERSION="$VERSION" TAG="$TAG" REPO_SLUG="$REPO_SLUG" python3 - <<'PY'
import json
import os
import sys

path = os.environ["SBOM"]
try:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
except (OSError, json.JSONDecodeError) as exc:
    sys.exit(f"release-integrity: {path} is not readable JSON: {exc}")

if not isinstance(doc, dict) or doc.get("bomFormat") != "CycloneDX":
    sys.exit(f"release-integrity: {path} is not a CycloneDX document (bomFormat != 'CycloneDX')")
if not doc.get("specVersion"):
    sys.exit(f"release-integrity: {path} has no specVersion")
components = doc.get("components")
if not isinstance(components, list) or not components:
    sys.exit(
        f"release-integrity: {path} lists no components — an empty SBOM would be attached to "
        "the release as if it were evidence"
    )

metadata = doc.setdefault("metadata", {})
metadata["component"] = {
    "type": "application",
    "bom-ref": f"latence-framework@{os.environ['VERSION']}",
    "name": "latence-framework",
    "version": os.environ["VERSION"],
    "description": "Latence Framework — messy enterprise data to AI-ready data.",
    "licenses": [{"license": {"id": "Apache-2.0"}}],
}
properties = [p for p in metadata.get("properties", []) if p.get("name") != "latence:release:tag"]
properties.append({"name": "latence:release:tag", "value": os.environ["TAG"]})
properties = [p for p in properties if p.get("name") != "latence:release:repository"]
properties.append({"name": "latence:release:repository", "value": os.environ["REPO_SLUG"]})
metadata["properties"] = sorted(properties, key=lambda p: p["name"])

with open(path, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"    stamped {len(components)} components as latence-framework {os.environ['VERSION']}")
PY

# ---- 2. Verification instructions (shipped WITH the assets, and reused as release notes) --
# Written BEFORE the checksums so VERIFY.md is itself covered by SHA256SUMS: the instructions
# telling an adopter how to verify must not be the one asset an attacker can rewrite freely.
CERT_IDENTITY="https://github.com/${REPO_SLUG}/.github/workflows/release.yml@refs/tags/${TAG}"
cat > "$OUT/VERIFY.md" <<EOF
## Verifying this release

Every asset of \`${TAG}\` is covered by \`SHA256SUMS\`, and \`SHA256SUMS\` is signed with
[keyless Sigstore](https://www.sigstore.dev/) — the signing identity is this repository's
release workflow, certified by Fulcio and recorded in the Rekor transparency log. There is
no long-lived key to trust or to leak.

Download all the release assets into one directory, then:

\`\`\`bash
# 1. Verify the signature over SHA256SUMS (uvx = uv tool run; pipx run sigstore also works)
uvx sigstore verify identity \\
  --bundle SHA256SUMS.sigstore.json \\
  --cert-identity '${CERT_IDENTITY}' \\
  --cert-oidc-issuer 'https://token.actions.githubusercontent.com' \\
  SHA256SUMS

# 2. Verify every artifact against the now-trusted SHA256SUMS
sha256sum -c SHA256SUMS        # macOS: shasum -a 256 -c SHA256SUMS
\`\`\`

Step 1 failing means the checksum file is not the one this workflow produced — stop.
Step 2 failing means an artifact does not match its recorded digest — stop.

### Software Bill of Materials

\`sbom.cdx.json\` is a **CycloneDX** SBOM for this exact release, generated from the
committed \`uv.lock\`, so it matches what a \`uv sync\` installs. Feed it to \`grype\`,
Dependency-Track, or any CycloneDX consumer to answer "am I affected by CVE-X?".

### Reporting a vulnerability

Please report security issues **privately** — see the coordinated-vulnerability-disclosure
policy in [SECURITY.md](https://github.com/${REPO_SLUG}/blob/${TAG}/SECURITY.md):
GitHub Security Advisories ("Report a vulnerability" on the *Security* tab) or email the
maintainer. Acknowledgement within 3 business days; coordinated disclosure with a 90-day
target embargo.
EOF

# ---- 3. Stage the distributions flat + checksum every asset ------------------------------
for f in "${DIST_FILES[@]}"; do cp "$f" "$OUT/"; done

# GNU coreutils on CI runners, BSD/perl shasum on macOS — both emit the same format.
if command -v sha256sum >/dev/null 2>&1; then
  SHA_GEN=(sha256sum)
else
  SHA_GEN=(shasum -a 256)
fi

echo "==> Writing SHA256SUMS"
(
  cd "$OUT"
  # Bare, sorted filenames so `sha256sum -c SHA256SUMS` works in whatever directory an
  # adopter downloaded the release assets into.
  find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -exec basename {} \; | sort | tr '\n' '\0' \
    | xargs -0 "${SHA_GEN[@]}" > SHA256SUMS
)
echo "    $(wc -l < "$OUT/SHA256SUMS" | tr -d ' ') files checksummed"

# ---- 4. Sign ----------------------------------------------------------------------------
if [ "$SIGN" = "yes" ]; then
  echo "==> Signing SHA256SUMS with keyless Sigstore"
  SIGN_ARGS=(sign --bundle "$OUT/SHA256SUMS.sigstore.json" --overwrite --no-default-files)
  [ -n "$IDENTITY_TOKEN" ] && SIGN_ARGS+=(--identity-token "$IDENTITY_TOKEN")
  uv run --with sigstore --no-project -- sigstore "${SIGN_ARGS[@]}" "$OUT/SHA256SUMS"
  [ -s "$OUT/SHA256SUMS.sigstore.json" ] || die "sigstore produced no bundle"
  echo "    signed: $OUT/SHA256SUMS.sigstore.json"
else
  echo "==> --no-sign: NOT signing. This asset set is a DRY RUN and must not be published."
fi

echo
echo "==> Release assets ready in $OUT:"
ls -1 "$OUT" | sed 's/^/      /'
echo
cat "$OUT/VERIFY.md"
