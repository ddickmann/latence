#!/usr/bin/env bash
# benchmark/corpus/fetch_corpus.sh — acquire the ~100k-file benchmark corpus on a fresh RunPod pod.
#
#   bash benchmark/corpus/fetch_corpus.sh                       # defaults: ~98k files, ~56 GB
#   GOVDOCS_ZIPS=20 bash benchmark/corpus/fetch_corpus.sh        # ~29k files, smoke run
#
# VERIFIED END TO END: zip 000 downloaded, SHA-1 checked, extracted, measured — 981 files,
# 0.8 GB, 22 distinct extensions (pdf 200, html 181, txt 154, doc 111, jpg 89, ppt 88, xls 62).
#   CORPUS_ROOT=/workspace/corpus bash benchmark/corpus/fetch_corpus.sh
#
# THREE AXES, because no single public source covers all of them (issues/e2e-extended/DECISIONS.md
# §D4). The full corpora total ~450 GB; that is disproportionate for a benchmark, so this script
# takes a bounded, reproducible slice of each:
#
#   GovDocs1   file-type diversity   CC0                  90 zips x ~986 files = ~88,700
#   SEC VPRR   scanned realism       SEC public info       ~8,000 scanned PDFs
#   OHR-Bench  scored control        CC-BY-4.0             ~1,000 PDFs + QA ground truth
#
# ============================== READ THIS BEFORE RUNNING =====================================
#
# 1. GovDocs1 CONTAINS LIVE MALWARE, DELIBERATELY RETAINED. From the corpus page: "the malware
#    will not be removed from the files because it is legitimately part of the corpus." We only
#    ever PARSE these files, never execute them — but treat this pod as untrusted: no host mounts,
#    no credential material beyond what the run needs, and expect AV noise. This must be a
#    deliberate posture, not a 2 a.m. surprise.
#
# 2. SEC REQUIRES A DECLARED USER-AGENT. A client without one gets HTTP 403 at the edge — enforced,
#    not advisory — plus a 10 req/s cap counted per entity across all machines, with a 10-minute IP
#    cooldown on violation. Export SEC_UA with a REAL, reachable contact before running.
#
# 3. PUBLICATION. Only GovDocs1 (CC0) and OHR-Bench (CC-BY-4.0) may appear in a public artifact.
#    SEC VPRR is SEC public information (redistributable, but do not use the SEC seal or the EDGAR
#    trademark). Nothing from IIT-CDIP / UCSF IDL is publishable — they are not fetched here.
#
# Idempotent and resumable: every fetch is skipped when its target already exists, so re-running
# after an interruption costs only the manifest rebuild.
set -euo pipefail

CORPUS_ROOT="${CORPUS_ROOT:-/workspace/corpus}"
GOVDOCS_ZIPS="${GOVDOCS_ZIPS:-90}"      # ~986 files each (MEASURED: 981 in zip 000)
VPRR_PDFS="${VPRR_PDFS:-8000}"
FETCH_OHR="${FETCH_OHR:-1}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"             # 1 keeps the .zip after extract (doubles peak disk)
PARALLEL="${PARALLEL:-4}"
SEC_RPS="${SEC_RPS:-8}"                 # under the SEC's 10/s ceiling, deliberately

GOVDOCS_BASE="https://downloads.digitalcorpora.org/corpora/files/govdocs1"
VPRR_BASE="https://www.sec.gov/Archives/edgar/vprr"
OHR_URL="https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/main/pdfs.zip"

C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_B=$'\033[1;36m'; C_0=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$C_B" "$C_0" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s warn%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '%s FAIL%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight

log "preflight"

for tool in curl unzip python3 awk; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
command -v sha1sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
  || warn "no sha1sum/shasum — GovDocs1 checksum verification will be SKIPPED"

# Disk. Refusing up front beats dying 80% through a 30 GB download with a half-extracted tree.
# MEASURED, not estimated: zip 000 is 486 MB and expands to 0.8 GB (1.65x). The published
# corpus averages 333 MB/zip across 1,000 zips, so ~0.55 GB extracted each. 1.7x for headroom.
need_gb=$(( (GOVDOCS_ZIPS * 333 / 1000) * 17 / 10 + 5 ))
if [ "$KEEP_ZIPS" = "1" ]; then need_gb=$(( need_gb + GOVDOCS_ZIPS * 333 / 1000 )); fi

# `df -Pk` (POSIX, 1K blocks) rather than `df -PBG` — the latter is a GNU extension that does not
# exist on BSD/macOS, and under `set -o pipefail` its failure propagates out of the command
# substitution and kills the script before the fallback branch can ever run. Both halves of that
# were real: `|| true` keeps a probe failure a warning instead of a fatal, which is what a probe
# should be.
probe_dir=$(dirname "$CORPUS_ROOT")
[ -d "$probe_dir" ] || probe_dir="/"
avail_gb=$( { df -Pk "$probe_dir" 2>/dev/null || true; } | awk 'NR==2 {printf "%d", $4/1048576}' )
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -gt 0 ] 2>/dev/null; then
  log "disk: need ~${need_gb} GB, have ${avail_gb} GB at ${probe_dir}"
  if [ "$avail_gb" -lt "$need_gb" ]; then
    die "insufficient disk: need ~${need_gb} GB, have ${avail_gb} GB. Lower GOVDOCS_ZIPS (each zip
    is ~986 files / ~333 MB compressed, ~1.65x extracted) or mount a larger volume."
  fi
else
  warn "could not read free disk at ${probe_dir} — proceeding blind"
fi

if [ "$VPRR_PDFS" -gt 0 ] && [ -z "${SEC_UA:-}" ]; then
  die "SEC_UA is not set. The SEC returns HTTP 403 to any client without a declared User-Agent.
    Set a REAL, reachable contact, e.g.
      export SEC_UA='Acme Research contact@acme.com'
    Or skip the scanned axis with VPRR_PDFS=0."
fi

mkdir -p "$CORPUS_ROOT"/{govdocs1,vprr,ohr-bench,_work}
ok "corpus root: $CORPUS_ROOT"

printf '\n%s' "$C_Y"
cat <<'BANNER'
  ┌────────────────────────────────────────────────────────────────────────┐
  │  GovDocs1 retains LIVE MALWARE by design. This pipeline only ever      │
  │  PARSES these files — it never executes them. Treat this pod as        │
  │  untrusted: no host mounts, no stray credentials, expect AV noise.     │
  └────────────────────────────────────────────────────────────────────────┘
BANNER
printf '%s\n' "$C_0"

# ---------------------------------------------------------------- GovDocs1

log "GovDocs1 — ${GOVDOCS_ZIPS} zips ≈ $((GOVDOCS_ZIPS * 986)) files (CC0)"

SUMS="$CORPUS_ROOT/_work/zipfilelist-sha1.txt"
if [ ! -s "$SUMS" ]; then
  curl -fsSL --retry 3 -o "$SUMS" "$GOVDOCS_BASE/zipfilelist-sha1.txt" \
    || warn "checksum manifest unavailable — downloads will not be verified"
fi

fetch_zip() {
  local n="$1"
  local id; id=$(printf '%03d' "$n")
  local zip="$CORPUS_ROOT/_work/${id}.zip"
  local dest="$CORPUS_ROOT/govdocs1/${id}"

  [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ] && return 0   # resumable

  curl -fsSL --retry 3 --retry-delay 2 -o "$zip" "$GOVDOCS_BASE/zipfiles/${id}.zip" || {
    printf 'WARN: %s.zip failed to download\n' "$id" >&2; rm -f "$zip"; return 0; }

  # Verify against the published SHA-1 when we have it. A truncated zip that still unzips is the
  # failure mode this catches — silent partial corpora are worse than loud missing ones.
  if [ -s "$SUMS" ]; then
    local want got
    want=$(awk -v f="${id}.zip" '$0 ~ f {print $1; exit}' "$SUMS" || true)
    if [ -n "$want" ]; then
      if command -v sha1sum >/dev/null 2>&1; then got=$(sha1sum "$zip" | cut -d' ' -f1)
      else got=$(shasum -a 1 "$zip" | cut -d' ' -f1); fi
      if [ "$want" != "$got" ]; then
        printf 'WARN: %s.zip CHECKSUM MISMATCH (want %s got %s) - discarded\n' "$id" "$want" "$got" >&2
        rm -f "$zip"; return 0
      fi
    fi
  fi

  mkdir -p "$dest"
  # -qq quiet, -o overwrite, -j FLATTEN: GovDocs1 zips carry a nested dir we do not need, and a
  # flat tree keeps the parser's source walk simple.
  unzip -qq -o -j "$zip" -d "$dest" 2>/dev/null || printf 'WARN: %s.zip extract had errors\n' "$id" >&2
  [ "$KEEP_ZIPS" = "1" ] || rm -f "$zip"
}
export -f fetch_zip
export CORPUS_ROOT GOVDOCS_BASE SUMS KEEP_ZIPS

seq 0 $((GOVDOCS_ZIPS - 1)) | xargs -P "$PARALLEL" -I{} bash -c 'fetch_zip {}'
ok "GovDocs1: $(find "$CORPUS_ROOT/govdocs1" -type f 2>/dev/null | wc -l | tr -d ' ') files"

# ---------------------------------------------------------------- SEC VPRR

if [ "$VPRR_PDFS" -gt 0 ]; then
  log "SEC VPRR — up to ${VPRR_PDFS} scanned PDFs at ${SEC_RPS} req/s (SEC public information)"
  SEC_UA="$SEC_UA" VPRR_BASE="$VPRR_BASE" CORPUS_ROOT="$CORPUS_ROOT" \
  VPRR_PDFS="$VPRR_PDFS" SEC_RPS="$SEC_RPS" python3 - <<'PY'
import json, os, sys, time, urllib.request
from pathlib import Path

UA   = os.environ["SEC_UA"]
BASE = os.environ["VPRR_BASE"]
DEST = Path(os.environ["CORPUS_ROOT"]) / "vprr"
WANT = int(os.environ["VPRR_PDFS"])
GAP  = 1.0 / float(os.environ["SEC_RPS"])          # the rate limit, honoured on EVERY request

def get(url: str, binary: bool = False):
    # The declared User-Agent is not optional: without it the SEC returns 403 at the edge.
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"})
    time.sleep(GAP)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if binary else json.loads(r.read())

try:
    dirs = [i["name"].strip("/") for i in get(f"{BASE}/index.json")["directory"]["item"]
            if i.get("type") == "dir"]
except Exception as exc:
    print(f"WARN: VPRR index unreachable ({exc}); skipping the scanned axis", file=sys.stderr)
    raise SystemExit(0)

dirs.sort()
got = 0
for d in dirs:
    if got >= WANT:
        break
    try:
        items = get(f"{BASE}/{d}/index.json")["directory"]["item"]
    except Exception:
        continue
    out = DEST / d
    out.mkdir(parents=True, exist_ok=True)
    for it in items:
        if got >= WANT:
            break
        name = it.get("name", "")
        if not name.lower().endswith(".pdf"):
            continue
        target = out / name
        if target.exists() and target.stat().st_size > 0:
            got += 1
            continue                                    # resumable
        try:
            target.write_bytes(get(f"{BASE}/{d}/{name}", binary=True))
            got += 1
        except Exception as exc:
            print(f"WARN: {d}/{name}: {exc}", file=sys.stderr)
    print(f"  vprr/{d}: {got}/{WANT}", flush=True)
print(f"VPRR: {got} PDFs")
PY
  ok "SEC VPRR: $(find "$CORPUS_ROOT/vprr" -type f 2>/dev/null | wc -l | tr -d ' ') files"
else
  warn "VPRR_PDFS=0 — skipping the scanned axis (no OCR-hostile material in this corpus)"
fi

# ---------------------------------------------------------------- OHR-Bench

if [ "$FETCH_OHR" = "1" ]; then
  log "OHR-Bench — scored control with QA ground truth (CC-BY-4.0)"
  ohr_zip="$CORPUS_ROOT/_work/ohr-pdfs.zip"
  if [ -n "$(ls -A "$CORPUS_ROOT/ohr-bench" 2>/dev/null)" ]; then
    ok "already present"
  elif curl -fsSL --retry 3 -o "$ohr_zip" "$OHR_URL"; then
    unzip -qq -o "$ohr_zip" -d "$CORPUS_ROOT/ohr-bench" && rm -f "$ohr_zip"
    ok "OHR-Bench: $(find "$CORPUS_ROOT/ohr-bench" -type f | wc -l | tr -d ' ') files"
  else
    warn "OHR-Bench fetch failed (gated or moved) — try: hf download opendatalab/OHR-Bench \
--repo-type dataset --include 'pdfs.zip' --local-dir $CORPUS_ROOT/ohr-bench"
  fi
fi

# ---------------------------------------------------------------- manifest

log "measuring what actually landed"
CORPUS_ROOT="$CORPUS_ROOT" python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in dir() else "."))
sys.path.insert(0, os.path.join(os.getcwd(), "benchmark", "corpus"))
from manifest import SourceManifest, scan_tree, write_manifest   # noqa: E402

root = Path(os.environ["CORPUS_ROOT"])
spec = [
    ("GovDocs1", "CC0", True, "govdocs1",
     ("Retains live malware by design — parse only, never execute.",
      "2009 harvest: office formats are legacy OLE, not modern OOXML.",
      "33,917 files corpus-wide carry a WRONG extension, deliberately.")),
    ("SEC VPRR", "SEC public information", True, "vprr",
     ("Scanned paper-filed SEC submissions.",
      "Do not use the SEC seal, logos, or the EDGAR trademark.")),
    ("OHR-Bench", "CC-BY-4.0", True, "ohr-bench",
     ("Scored control: ships QA ground truth, so parse QUALITY is measurable, not just throughput.",)),
]
sources = []
for name, lic, pub, sub, notes in spec:
    path = root / sub
    if not path.exists():
        continue
    count, total, hist = scan_tree(path)
    if count == 0:
        continue
    sources.append(SourceManifest(name, lic, pub, str(path), count, total, dict(hist), notes))

if not sources:
    print("no sources acquired — nothing to measure", file=sys.stderr)
    raise SystemExit(1)

m = write_manifest(sources, root)
d = m["diversity"]
print(f"\n  files: {m['totals']['file_count']:,}")
print(f"  bytes: {m['totals']['total_bytes'] / 1e9:.1f} GB")
print(f"  distinct extensions: {d['distinct_extensions']}")
print(f"  office legacy OLE: {d['office_legacy_ole']:,} | modern OOXML: {d['office_ooxml']:,}")
for limit in m["limits"]:
    print(f"  LIMIT: {limit}")
PY

rmdir "$CORPUS_ROOT/_work" 2>/dev/null || true
printf '\n%s==> corpus ready at %s%s\n' "$C_G" "$CORPUS_ROOT" "$C_0"
echo "    manifest.json + MANIFEST.md written there — read the LIMITS section before quoting"
echo "    coverage in any write-up."
echo
echo "    Next: issues/e2e-extended/RUNBOOK.md §5 (parse)."
