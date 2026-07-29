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
# Modern OOXML. Several quarters because one quarter's 10-K/10-Q index yields ~6k filings and the
# per-filing hit rate is ~7/8, so a single quarter caps out below a 5k target.
EDGAR_XLSX="${EDGAR_XLSX:-5000}"
EDGAR_QUARTERS="${EDGAR_QUARTERS:-2024/QTR1 2024/QTR2 2023/QTR1 2023/QTR3}"
# .docx and .pptx, per format. Paced by Zenodo's Crawl-delay: 10 on search, so this axis is slow by
# design — 100 records per search call keeps the number of disallowed-path requests minimal.
ZENODO_OOXML="${ZENODO_OOXML:-2500}"
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

# SINGLE INSTANCE. Two runs of this script against one corpus root will fight over the same download
# paths — MEASURED twice: two `curl -o .../001.zip` processes interleaving writes into one file,
# producing an archive that fails `unzip -t` forever while both look healthy in `ps`. It happens
# whenever an earlier invocation is killed and leaves an orphaned child (killing a shell script does
# not kill its curls), which a supervisor's retry loop makes routine rather than rare.
#
# So the mutual exclusion belongs HERE, not in the caller's discipline. Non-blocking: a second
# instance exits immediately and says why, rather than queueing up to cause the same collision later.
exec 9>"$CORPUS_ROOT/_work/.fetch.lock"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || die "another fetch_corpus.sh already holds $CORPUS_ROOT/_work/.fetch.lock.
    Concurrent runs corrupt shared downloads. Wait for it, or kill its PROCESS GROUP (not just the
    script — orphaned curls are the whole problem) before retrying."
fi

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

  # Download to a PID-unique part file and rename only on success. Belt to the lock's braces: even
  # if two instances somehow coexist, neither can ever see the other's half-written bytes, and an
  # interrupted run leaves a .part behind rather than a plausible-looking corrupt archive.
  local part="${zip}.$$.part"
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "$part" "$GOVDOCS_BASE/zipfiles/${id}.zip"; then
    printf 'WARN: %s.zip failed to download\n' "$id" >&2; rm -f "$part"; return 0
  fi
  mv -f "$part" "$zip"

  # INTEGRITY. Two checks, and which one is AUTHORITATIVE matters — getting this backwards
  # silently discards good data.
  #
  # MEASURED on the live mirror: Digital Corpora's published SHA-1 manifest is STALE relative to
  # the bytes it actually serves. 001.zip re-fetches to a stable 9c23f4e7... against a published
  # a6341dd1..., and `unzip -t` reports "No errors detected" over all 977 entries. So a SHA-1
  # mismatch here does NOT mean corruption; treating it as fatal threw away ~986 good files per
  # hit (observed rate ~1 in 27).
  #
  # So: `unzip -t` is the authority — it is what actually answers "is this archive usable", and it
  # catches the truncation the checksum was there for. The published SHA-1 is kept as ADVISORY and
  # every mismatch is RECORDED, because a stale upstream manifest is a fact about the corpus that
  # belongs in the manifest rather than in a discarded file.
  if [ -s "$SUMS" ]; then
    local want got
    want=$(awk -v f="/${id}.zip" 'index($0, f) {print $1; exit}' "$SUMS" || true)
    if [ -n "$want" ]; then
      if command -v sha1sum >/dev/null 2>&1; then got=$(sha1sum "$zip" | awk '{print $1}')
      else got=$(shasum -a 1 "$zip" | awk '{print $1}'); fi
      if [ "$want" != "$got" ]; then
        printf '%s.zip published=%s served=%s\n' "$id" "$want" "$got" \
          >> "$CORPUS_ROOT/_work/sha1-advisory-mismatches.txt"
      fi
    fi
  fi

  # The authoritative gate. A truncated or corrupt archive fails here and IS discarded.
  if ! unzip -qq -t "$zip" >/dev/null 2>&1; then
    printf 'WARN: %s.zip failed integrity check (unzip -t) - discarded\n' "$id" >&2
    rm -f "$zip"; return 0
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
import gzip, json, os, sys, time, urllib.request, zlib
from pathlib import Path

UA   = os.environ["SEC_UA"]
BASE = os.environ["VPRR_BASE"]
DEST = Path(os.environ["CORPUS_ROOT"]) / "vprr"
WANT = int(os.environ["VPRR_PDFS"])
GAP  = 1.0 / float(os.environ["SEC_RPS"])          # the rate limit, honoured on EVERY request

def get(url: str, binary: bool = False):
    # The declared User-Agent is not optional: without it the SEC returns 403 at the edge.
    #
    # `urllib` does NOT transparently decompress: advertising gzip and then handing the raw body to
    # json.loads fails with "'utf-8' codec can't decode byte 0x8b in position 1" — 0x1f 0x8b being
    # the gzip magic. That decoded, in the first run here, as "index unreachable", which is how a
    # working endpoint came to look like an outage. Decompress by what the server actually says it
    # sent, rather than by what we asked for.
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"})
    time.sleep(GAP)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        encoding = (r.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip":
        raw = gzip.decompress(raw)
    elif encoding == "deflate":
        # Raw-deflate and zlib-wrapped deflate both occur in the wild; wbits=-15 handles the former.
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -15)
    if binary:
        return raw
    # MEASURED: these index.json bodies begin with a literal `Content-type: application/json` line
    # followed by a blank line, before the JSON — a CGI header leaked into the payload. The response
    # header already says application/json, so this is duplication the parser must step over. Slice
    # from the first brace rather than stripping a fixed prefix, so a change in that preamble does
    # not break parsing again.
    brace = raw.find(b"{")
    if brace == -1:
        raise ValueError(f"no JSON object in response from {url} (first 120 bytes: {raw[:120]!r})")
    return json.loads(raw[brace:])


def get_retry(url: str, binary: bool = False, attempts: int = 4):
    """Retry with backoff. A single transient 5xx must not silently delete a whole corpus axis."""
    for attempt in range(1, attempts + 1):
        try:
            return get(url, binary=binary)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}/{attempts - 1} for {url}: {exc}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)


def _is_dir(item: dict) -> bool:
    """MEASURED: EDGAR reports `type` as an ICON NAME — `folder.gif`, `pdf.gif` — never `"dir"`.

    The original `type == "dir"` test matched nothing, so even a correctly parsed index yielded zero
    directories and the axis came back empty with no error. Test the icon prefix, and accept a
    trailing-slash url as corroboration in case the icon vocabulary changes.
    """
    return (str(item.get("type", "")).startswith("folder")
            or str(item.get("url", "")).rstrip().endswith("/"))


try:
    dirs = [i["name"].strip("/") for i in get_retry(f"{BASE}/index.json")["directory"]["item"]
            if _is_dir(i)]
    if not dirs:
        raise ValueError("index parsed but contained no directories — the item filter is wrong")
except Exception as exc:
    # Exit non-zero. The previous version exited 0 here, so the caller printed "ok SEC VPRR: 0 files"
    # and the run continued believing it had scanned material it did not have. A missing axis is a
    # result the operator must see, not a footnote to swallow.
    print(f"ERROR: VPRR index unreachable after retries ({exc!r}); the scanned axis is EMPTY",
          file=sys.stderr)
    raise SystemExit(3)

dirs.sort()
got = 0
for d in dirs:
    if got >= WANT:
        break
    try:
        items = get_retry(f"{BASE}/{d}/index.json")["directory"]["item"]
    except Exception as exc:
        print(f"WARN: vprr/{d} index skipped: {exc}", file=sys.stderr, flush=True)
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
            target.write_bytes(get_retry(f"{BASE}/{d}/{name}", binary=True, attempts=2))
            got += 1
        except Exception as exc:
            print(f"WARN: {d}/{name}: {exc}", file=sys.stderr)
    print(f"  vprr/{d}: {got}/{WANT}", flush=True)
print(f"VPRR: {got} PDFs")
PY
  # An empty axis is never "ok". The first run printed `ok SEC VPRR: 0 files` after the index fetch
  # had failed outright, which is precisely the shape of a green report over absent work.
  vprr_n=$(find "$CORPUS_ROOT/vprr" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$vprr_n" -gt 0 ]; then
    ok "SEC VPRR: ${vprr_n} files"
  else
    printf '%s\n' "vprr: requested ${VPRR_PDFS}, acquired 0" \
      >> "$CORPUS_ROOT/_work/empty-axes.txt"
    warn "SEC VPRR acquired 0 files — the scanned/OCR-hostile axis is ABSENT from this corpus.
      Any OCR claim made from this run covers born-digital material only. Recorded to
      _work/empty-axes.txt and surfaced in the manifest LIMITS."
  fi
else
  warn "VPRR_PDFS=0 — skipping the scanned axis (no OCR-hostile material in this corpus)"
fi

# ---------------------------------------------------------------- SEC EDGAR — modern OOXML
#
# MEASURED, and the reason this axis exists: GovDocs1 is a 2009 harvest, so the corpus arrives with
# 16,021 legacy OLE files (doc/xls/ppt) and **34** modern OOXML. OOXML is not a variant of OLE — it
# is a zipped-XML container parsed by a completely different code path — so without this axis the
# pipeline's modern-Office handling would be exercised by ~nothing while "many file types" implied
# otherwise.
#
# Every 10-K/10-Q filing directory carries a `Financial_Report.xlsx`. VERIFIED against the live
# endpoint: 6,052 such filings in one quarter's form.idx, 7 of 8 sampled URLs returned files whose
# magic bytes are PK\x03\x04 (a real OOXML zip, not a renamed legacy file or an HTML error page).
# The ~1-in-8 404 is expected — not every filer publishes the rendered financial report.
if [ "$EDGAR_XLSX" -gt 0 ]; then
  log "SEC EDGAR — up to ${EDGAR_XLSX} modern .xlsx from ${EDGAR_QUARTERS}"
  SEC_UA="$SEC_UA" CORPUS_ROOT="$CORPUS_ROOT" EDGAR_XLSX="$EDGAR_XLSX" \
  EDGAR_QUARTERS="$EDGAR_QUARTERS" SEC_RPS="$SEC_RPS" python3 - <<'PY'
import gzip, os, re, sys, time, urllib.request, zlib
from pathlib import Path

UA   = os.environ["SEC_UA"]
DEST = Path(os.environ["CORPUS_ROOT"]) / "edgar-xlsx"
WANT = int(os.environ["EDGAR_XLSX"])
QTRS = os.environ["EDGAR_QUARTERS"].split()
GAP  = 1.0 / float(os.environ["SEC_RPS"])

def get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"})
    time.sleep(GAP)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip":
        raw = gzip.decompress(raw)
    elif enc == "deflate":
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -15)
    return raw

DEST.mkdir(parents=True, exist_ok=True)
got = len(list(DEST.glob("*.xlsx")))                       # resumable
row = re.compile(r"edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})\.txt")

for qtr in QTRS:
    if got >= WANT:
        break
    try:
        idx = get(f"https://www.sec.gov/Archives/edgar/full-index/{qtr}/form.idx").decode("latin-1")
    except Exception as exc:
        print(f"WARN: form.idx {qtr} unavailable: {exc}", file=sys.stderr, flush=True)
        continue
    filings = [m.groups() for line in idx.splitlines()
               if line.startswith(("10-K ", "10-Q ")) for m in [row.search(line)] if m]
    print(f"  {qtr}: {len(filings)} 10-K/10-Q filings indexed", flush=True)

    for cik, acc in filings:
        if got >= WANT:
            break
        target = DEST / f"{cik}-{acc.replace('-', '')}.xlsx"
        if target.exists() and target.stat().st_size > 0:
            got += 1
            continue
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
               f"{acc.replace('-', '')}/Financial_Report.xlsx")
        try:
            blob = get(url)
        except Exception:
            continue                        # a filer without a rendered report; expected, not an error
        # Verify it IS OOXML before counting it. Accepting whatever the URL returned is how a corpus
        # ends up full of HTML error pages that only surface as parse failures hours later.
        if blob[:4] != b"PK\x03\x04":
            print(f"WARN: {url} returned non-OOXML ({blob[:4]!r})", file=sys.stderr, flush=True)
            continue
        target.write_bytes(blob)
        got += 1
        if got % 100 == 0:
            print(f"  edgar-xlsx: {got}/{WANT}", flush=True)

print(f"EDGAR xlsx: {got}")
PY
  xlsx_n=$(find "$CORPUS_ROOT/edgar-xlsx" -type f -name '*.xlsx' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$xlsx_n" -gt 0 ]; then
    ok "SEC EDGAR modern .xlsx: ${xlsx_n} files"
  else
    printf '%s\n' "edgar-xlsx: requested ${EDGAR_XLSX}, acquired 0" \
      >> "$CORPUS_ROOT/_work/empty-axes.txt"
    warn "EDGAR acquired 0 .xlsx — modern OOXML remains UNTESTED in this corpus."
  fi
fi

# ---------------------------------------------------------------- Zenodo — .docx / .pptx
#
# The EDGAR axis above closes .xlsx only. .docx and .pptx are separate OOXML parts exercised by
# separate code, and GovDocs1 supplies ~0 of either. Zenodo carries 100,322 CC-BY-4.0 records with
# .docx and 17,594 with .pptx (live-queried), and — unlike EDGAR or Common Crawl — CC-BY/CC0 grant
# REDISTRIBUTION, so this is the only axis here whose files may enter a public artifact.
#
# TWO verification gates, both learned from measurement:
#
#  1. The search index and the record payload disagree about where the licence lives. Filtering on
#     `metadata.rights.id` works for SEARCH, but the returned record has `rights: null` and states
#     its licence at `metadata.license.id`. Trusting the query alone would mean asserting CC-BY over
#     files that never declared it. So every record is re-checked against its OWN payload and
#     skipped if it does not self-declare an allowed licence and open access.
#  2. A malformed licence filter FAILS OPEN on this API — a bare `rights:` facet returns the
#     unfiltered total. Verified that a bogus licence id returns 0 before relying on the filter.
#
# CC-BY-4.0 §3(a) requires attribution (author, title, licence URI, changes note), so an attribution
# CSV is written alongside the files. Without it the corpus is not redistributable in practice.
if [ "$ZENODO_OOXML" -gt 0 ]; then
  log "Zenodo — up to ${ZENODO_OOXML} each of .docx/.pptx (CC-BY-4.0 / CC0, REDISTRIBUTABLE)"
  CORPUS_ROOT="$CORPUS_ROOT" ZENODO_OOXML="$ZENODO_OOXML" ZENODO_TOKEN="${ZENODO_TOKEN:-}" python3 - <<'PY'
import csv, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

DEST = Path(os.environ["CORPUS_ROOT"]) / "zenodo-ooxml"
WANT = int(os.environ["ZENODO_OOXML"])
TOKEN = os.environ.get("ZENODO_TOKEN") or ""
UA = "Latence Research ddickmann81@gmail.com (benchmark corpus acquisition)"

# robots.txt: `Disallow: /api`, `Allow: /api/records/*/files`, `Crawl-delay: 10`. Downloads sit on
# the explicitly allowed path; search does not, so search is paced at the crawl-delay and used as
# sparingly as possible (100 records per call).
SEARCH_GAP, FILE_GAP = 10.0, 1.5
ALLOWED = {"cc-by-4.0", "cc-zero", "cc0-1.0"}


def get(url, binary=False, gap=FILE_GAP):
    headers = {"User-Agent": UA}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    time.sleep(gap)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        raw = r.read()
    return raw if binary else json.loads(raw)


def search(ext, page):
    q = f"filetype:{ext} AND metadata.rights.id:cc-by-4.0"
    return get("https://zenodo.org/api/records?" + urllib.parse.urlencode(
        {"q": q, "size": 100, "page": page}), gap=SEARCH_GAP)


DEST.mkdir(parents=True, exist_ok=True)
attrib_path = DEST / "ATTRIBUTION.csv"
new_file = not attrib_path.exists()
attrib = attrib_path.open("a", newline="", encoding="utf-8")
writer = csv.writer(attrib)
if new_file:
    writer.writerow(["file", "record_id", "title", "creators", "licence", "record_url", "changes"])

for ext in ("docx", "pptx"):
    got = len(list(DEST.glob(f"*.{ext}")))
    page = 1
    rejected = 0
    while got < WANT and page <= 60:
        try:
            hits = search(ext, page)["hits"]["hits"]
        except Exception as exc:
            print(f"WARN: zenodo search {ext} p{page}: {exc}", file=sys.stderr, flush=True)
            break
        if not hits:
            break
        for rec in hits:
            if got >= WANT:
                break
            md = rec.get("metadata", {})
            # GATE: the record must declare its own licence and open access. The search filter is
            # necessary but NOT sufficient — see the header note.
            lic = str((md.get("license") or {}).get("id", "")).lower()
            if lic not in ALLOWED or md.get("access_right") != "open":
                rejected += 1
                continue
            for f in rec.get("files", []):
                if got >= WANT:
                    break
                key = f.get("key", "")
                if not key.lower().endswith(f".{ext}"):
                    continue
                target = DEST / f"{rec['id']}-{Path(key).name}"
                if target.exists() and target.stat().st_size > 0:
                    got += 1
                    continue
                url = (f"https://zenodo.org/api/records/{rec['id']}/files/"
                       f"{urllib.parse.quote(key)}/content")
                try:
                    blob = get(url, binary=True)
                except Exception:
                    continue
                if blob[:4] != b"PK\x03\x04":
                    print(f"WARN: {url} is not OOXML ({blob[:4]!r})", file=sys.stderr, flush=True)
                    continue
                target.write_bytes(blob)
                creators = "; ".join(c.get("name", "") for c in md.get("creators", []))
                writer.writerow([target.name, rec["id"], md.get("title", ""), creators, lic,
                                 f"https://zenodo.org/records/{rec['id']}",
                                 "unmodified; used as pipeline input"])
                attrib.flush()
                got += 1
        print(f"  zenodo {ext}: {got}/{WANT} (page {page}, {rejected} rejected on licence)",
              flush=True)
        page += 1
    print(f"Zenodo {ext}: {got} files, {rejected} records rejected for not self-declaring a licence")

attrib.close()
PY
  for ext in docx pptx; do
    n=$(find "$CORPUS_ROOT/zenodo-ooxml" -type f -name "*.${ext}" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -gt 0 ]; then
      ok "Zenodo .${ext}: ${n} files"
    else
      printf '%s\n' "zenodo-${ext}: requested ${ZENODO_OOXML}, acquired 0" \
        >> "$CORPUS_ROOT/_work/empty-axes.txt"
      warn "Zenodo acquired 0 .${ext} — that OOXML format remains UNTESTED."
    fi
  done
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
    ("SEC EDGAR (Financial_Report.xlsx)", "SEC public information", True, "edgar-xlsx",
     ("The ONLY modern-OOXML axis: GovDocs1 supplies 16k legacy OLE and ~0 docx/xlsx/pptx.",
      "Rendered financial reports from 10-K/10-Q filings; every file magic-checked as PK\\x03\\x04.",
      "Covers .xlsx ONLY — .docx and .pptx remain untested at volume.",
      "Do not use the SEC seal, logos, or the EDGAR trademark.")),
    ("Zenodo OOXML (docx/pptx)", "CC-BY-4.0 / CC0", True, "zenodo-ooxml",
     ("Closes the .docx and .pptx gap EDGAR's .xlsx axis does not cover.",
      "Every record re-verified against its OWN metadata.license.id — the search filter alone is "
      "not sufficient, and a malformed licence filter fails OPEN on this API.",
      "REDISTRIBUTABLE, unlike GovDocs1-adjacent and EDGAR material: see ATTRIBUTION.csv, which "
      "CC-BY-4.0 s3(a) requires to accompany any redistribution.",
      "EU-hosted academic material: may contain personal data — screen before publishing.")),
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
