"""Measure an acquired corpus and emit its manifest (T-CORPUS-1 acceptance).

The acceptance bar for corpus acquisition is not "the download finished" — it is a manifest
carrying per-source counts, bytes, licence, and the **measured** file-type histogram. Measured,
because two of the three claims we want to make about this corpus are falsifiable only by counting:

* that the corpus is genuinely file-type diverse (the reason GovDocs1 was chosen at all), and
* that its office formats are **legacy OLE**, not modern OOXML — a real limit of a 2009 harvest
  that the write-up must state rather than let "many file types" quietly imply.

So this module counts what actually landed on disk and writes it down. A histogram derived from the
publisher's documentation would restate the brochure; this one can contradict it, which is the only
kind worth printing.

**Extension vs content.** GovDocs1 documents that **33,917 of its files carry the wrong
extension** — *"The file extension is a suggestion; it is not part of the corpus."* The histogram
below is therefore by CLAIMED extension only; it does not sniff content. That is a deliberate
limit, not an oversight: the extension/content disagreement is a result the *pipeline* should
produce, and measuring it here with a second, independent detector would pre-empt the very thing
the corpus was chosen to test.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Extensions we group as "office (legacy OLE)" vs "office (OOXML)". The split is the point: a 2009
#: harvest is overwhelmingly the former, and conflating them would overstate our coverage.
LEGACY_OLE = frozenset({"doc", "xls", "ppt", "pps", "dot", "xlt", "pot"})
OOXML = frozenset({"docx", "xlsx", "pptx", "docm", "xlsm", "pptm"})


@dataclass(frozen=True)
class SourceManifest:
    """One acquired source: what it is, what it costs us legally, and what actually landed."""

    name: str
    licence: str
    #: What the licence permits us to PUBLISH. The operative field — several sources here grant
    #: access without granting redistribution, and that distinction decides what can appear in an
    #: artifact. Recorded per source so it cannot be lost in aggregation.
    publishable: bool
    root: str
    file_count: int
    total_bytes: int
    by_extension: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def _extension(path: Path) -> str:
    """The lowercased extension with no dot; ``"(none)"`` when a file has none.

    Deliberately naive — this is the *claimed* type, and its unreliability is what we measure.
    """
    suffix = path.suffix.lstrip(".").lower()
    return suffix if suffix else "(none)"


def scan_tree(root: Path, *, follow_symlinks: bool = False) -> tuple[int, int, Counter[str]]:
    """Walk ``root`` once and return ``(file_count, total_bytes, extension_histogram)``.

    One walk answers all three questions, so a manifest can never disagree with itself about the
    corpus it just measured. Unreadable entries are skipped rather than fatal: a partial measurement
    of a 100k-file tree beats no measurement, and the count itself reveals anything systematic.

    ``follow_symlinks`` defaults False — GovDocs1 is untrusted input and a symlink loop would hang
    the walk. This is the same posture the rest of the acquisition takes toward this corpus.
    """
    count = 0
    total = 0
    histogram: Counter[str] = Counter()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            try:
                if path.is_symlink() and not follow_symlinks:
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            count += 1
            total += size
            histogram[_extension(path)] += 1
    return count, total, histogram


def diversity_summary(histogram: dict[str, int]) -> dict[str, Any]:
    """Roll a raw extension histogram into the claims a reader will actually check.

    Reports the OOXML count **separately** from legacy OLE. If they were summed, a corpus with 77k
    ``.doc`` files and 164 ``.docx`` would read as "78k office documents" and the write-up would
    imply modern-format coverage that 164 files do not support.
    """
    total = sum(histogram.values()) or 1
    legacy = sum(n for ext, n in histogram.items() if ext in LEGACY_OLE)
    ooxml = sum(n for ext, n in histogram.items() if ext in OOXML)
    distinct = len([ext for ext, n in histogram.items() if n > 0])
    top = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return {
        "distinct_extensions": distinct,
        "office_legacy_ole": legacy,
        "office_ooxml": ooxml,
        "ooxml_share_of_office": (ooxml / (legacy + ooxml)) if (legacy + ooxml) else 0.0,
        "top_extensions": [{"ext": e, "count": n, "share": n / total} for e, n in top],
    }


def build_manifest(sources: list[SourceManifest]) -> dict[str, Any]:
    """Assemble the manifest, including the limits the corpus cannot support a claim about."""
    combined: Counter[str] = Counter()
    for source in sources:
        combined.update(source.by_extension)

    summary = diversity_summary(dict(combined))
    limits: list[str] = []

    # OOXML is reported per-format, not as one bucket. The SEC EDGAR axis supplies .xlsx at volume,
    # so a combined count would read as "modern Office covered" while .docx and .pptx — different
    # parts of the OOXML family, exercised by different code — remained at ~zero.
    by_ext = dict(combined)
    docx = by_ext.get("docx", 0) + by_ext.get("docm", 0)
    xlsx = by_ext.get("xlsx", 0) + by_ext.get("xlsm", 0)
    pptx = by_ext.get("pptx", 0) + by_ext.get("pptm", 0)
    thin = [f"{label} ({n})" for label, n in (("docx", docx), ("xlsx", xlsx), ("pptx", pptx))
            if n < 1000]
    if thin:
        limits.append(
            f"Modern OOXML is UNTESTED AT VOLUME for: {', '.join(thin)} — against "
            f"{summary['office_legacy_ole']} legacy OLE (doc/xls/ppt) from the 2009 GovDocs1 "
            "harvest. OOXML is a zipped-XML container parsed by a different code path than OLE, so "
            "legacy coverage says nothing about it. Do not let 'many file types' imply "
            "modern-format "
            "coverage for any format listed here."
        )
    if not any(s.publishable for s in sources):
        limits.append(
            "NO acquired source permits redistribution — nothing here may enter a public artifact."
        )
    # A stale upstream checksum manifest is a fact about the corpus, not a footnote. Digital
    # Corpora's published SHA-1 list does not match the bytes its mirror serves for some zips
    # (measured ~1 in 27); those archives pass `unzip -t` cleanly. Recording it here means a reader
    # can see the provenance caveat instead of trusting an unqualified "verified" claim.
    advisory = Path(sources[0].root).parent / "_work" / "sha1-advisory-mismatches.txt"
    try:
        if advisory.exists():
            n = len([ln for ln in advisory.read_text().splitlines() if ln.strip()])
            if n:
                limits.append(
                    f"{n} GovDocs1 archive(s) did not match the publisher's SHA-1 manifest but "
                    "passed structural integrity (`unzip -t`). The published manifest is stale "
                    "relative to the mirror; integrity was verified structurally, not by checksum. "
                    "See _work/sha1-advisory-mismatches.txt."
                )
    except OSError:
        pass

    # An axis that was requested and came back empty is the most consequential thing a manifest can
    # record, because every downstream claim silently narrows in scope. The scanned-PDF axis failing
    # turns "OCR verified end to end" into "OCR verified on born-digital files" — a different claim.
    empty = Path(sources[0].root).parent / "_work" / "empty-axes.txt"
    try:
        if empty.exists():
            for line in empty.read_text().splitlines():
                if line.strip():
                    limits.append(
                        f"AXIS ABSENT — {line.strip()}. This source contributed nothing; no claim "
                        "resting on it is supported by this corpus."
                    )
    except OSError:
        pass

    unpublishable = [s.name for s in sources if not s.publishable]
    if unpublishable:
        limits.append(
            f"Not publishable, fair-use/access-only posture: {', '.join(unpublishable)}. "
            "Nothing derived from these may appear in a shareable artifact."
        )

    return {
        "schema": 1,
        "totals": {
            "file_count": sum(s.file_count for s in sources),
            "total_bytes": sum(s.total_bytes for s in sources),
        },
        "diversity": summary,
        "sources": [
            {
                "name": s.name,
                "licence": s.licence,
                "publishable": s.publishable,
                "root": s.root,
                "file_count": s.file_count,
                "total_bytes": s.total_bytes,
                "by_extension": dict(sorted(s.by_extension.items())),
                "notes": list(s.notes),
            }
            for s in sources
        ],
        "limits": limits,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    """A human-readable manifest. The limits section is NOT optional — it is the honest half."""
    totals = manifest["totals"]
    div = manifest["diversity"]
    gb = totals["total_bytes"] / 1e9
    lines = [
        "# Corpus manifest",
        "",
        f"**{totals['file_count']:,} files · {gb:.1f} GB · "
        f"{div['distinct_extensions']} distinct extensions**",
        "",
        "## Sources",
        "",
        "| Source | Files | GB | Licence | Publishable |",
        "|---|---:|---:|---|---|",
    ]
    for s in manifest["sources"]:
        lines.append(
            f"| {s['name']} | {s['file_count']:,} | {s['total_bytes'] / 1e9:.2f} | "
            f"{s['licence']} | {'yes' if s['publishable'] else '**no**'} |"
        )
    lines += ["", "## Measured file-type histogram (top 15, by claimed extension)", ""]
    lines += ["| Extension | Files | Share |", "|---|---:|---:|"]
    for row in div["top_extensions"]:
        lines.append(f"| `{row['ext']}` | {row['count']:,} | {row['share'] * 100:.1f}% |")
    lines += [
        "",
        "## Office formats — reported split, deliberately",
        "",
        f"* legacy OLE (doc/xls/ppt): **{div['office_legacy_ole']:,}**",
        f"* modern OOXML (docx/xlsx/pptx): **{div['office_ooxml']:,}**",
        f"* OOXML share of office files: **{div['ooxml_share_of_office'] * 100:.2f}%**",
        "",
        "## Limits this corpus cannot support a claim about",
        "",
    ]
    lines += [f"* {limit}" for limit in manifest["limits"]] or ["* (none recorded)"]
    return "\n".join(lines) + "\n"


def write_manifest(sources: list[SourceManifest], out_dir: Path) -> dict[str, Any]:
    """Write ``manifest.json`` + ``MANIFEST.md`` and return the manifest mapping."""
    manifest = build_manifest(sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "MANIFEST.md").write_text(render_markdown(manifest), encoding="utf-8")
    return manifest
