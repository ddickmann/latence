#!/usr/bin/env python3
"""Secret-scan CI gate that is immune to detect-secrets' volatile baseline fields.

Why this exists (issue #70):
    The obvious gate — ``detect-secrets scan --baseline .secrets.baseline`` followed
    by ``git diff --exit-code .secrets.baseline`` — is broken. ``scan --baseline``
    rewrites volatile fields (the ``generated_at`` timestamp and every finding's
    ``line_number``) into the baseline on *every* invocation, so ``git diff
    --exit-code`` is guaranteed non-zero on a pristine tree with no new secret. That
    makes CI permanently red and drowns a genuinely new secret in constant noise.
    ``detect-secrets-hook --baseline`` has the same failure mode from a different
    angle: it rewrites drifted line numbers into the baseline and then refuses because
    the baseline is "unstaged".

What this does instead:
    Run a fresh ``detect-secrets scan`` of the tracked tree and compare the SET of
    ``(filename, hashed_secret)`` pairs against the audited baseline. The gate fails
    only when the scan surfaces a hashed secret that is not already accounted for
    (audited ``is_secret == false``) in the baseline. Line numbers and the timestamp
    are ignored entirely, so a pristine tree is always green and only a genuinely new
    (or newly-un-audited) secret turns it red.

Exit codes:
    0  every scanned secret is accounted for in the audited baseline
    1  at least one new / un-audited secret was found (details printed to stderr)
    2  usage / environment error (e.g. the baseline is missing)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_BASELINE = Path(".secrets.baseline")
# Paths to keep out of the scan: the lockfile (huge, high-entropy hashes), the local
# virtualenv, build outputs, the baseline itself (its recorded ``hashed_secret``
# SHA1 digests read as high-entropy strings, but they are one-way hashes of already
# audited findings, not live secrets), and GraphML knowledge-graph exports.
#
# On ``*.graphml``: these are not excluded because they are *noisy*, they are excluded
# because every high-entropy string in them is STRUCTURALLY incapable of being a secret.
# A GraphML export is a serialised knowledge graph whose node and edge ids are
# CONTENT-ADDRESSED: a sha256 over the corpus id plus the node's own canonical text and
# type (``contracts.py`` §"node/edge ids are content-addressed"). The id is therefore
# derived FROM the element it names — it reveals nothing the same file does not already
# state in the clear one line below, and it authenticates nothing. There is also no
# credential-shaped field for one to hide in: ``stages/graph_export.py::_render_graphml``
# emits exactly id, label, type, kb_id, confidence and evidence_count — never environment
# or connection material. Auditing them one by one would mean writing ~83k hashes of
# public benchmark text into the baseline (``benchmark/musique-subset/export/graph.graphml``
# alone), bloating the anchor without adding a single bit of review — and a bloated anchor
# is how a real finding gets lost. Kept as one regex for ``--exclude-files``.
_EXCLUDE_FILES = (
    r"(^|/)\.secrets\.baseline$|uv\.lock|\.venv/|dist/|node_modules/|\.graphml$"
)


def _audited_pairs(baseline: dict[str, Any]) -> set[tuple[str, str]]:
    """The (filename, hashed_secret) pairs a human has audited as *not* a real secret."""
    pairs: set[tuple[str, str]] = set()
    for filename, findings in baseline.get("results", {}).items():
        for f in findings:
            # Only findings explicitly audited is_secret==false grant a pass. An entry
            # left un-audited (is_secret null/true) is NOT a free pass.
            if f.get("is_secret") is False:
                pairs.add((filename, f["hashed_secret"]))
    return pairs


def _scanned_pairs(scan: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for filename, findings in scan.get("results", {}).items():
        for f in findings:
            pairs.add((filename, f["hashed_secret"]))
    return pairs


def main() -> int:
    if not _BASELINE.is_file():
        print(f"error: {_BASELINE} not found (secret-scan anchor missing)", file=sys.stderr)
        return 2

    baseline = json.loads(_BASELINE.read_text())

    # A fresh scan (no --baseline) never mutates any tracked file; it just prints JSON.
    proc = subprocess.run(
        ["detect-secrets", "scan", "--exclude-files", _EXCLUDE_FILES],
        capture_output=True,
        text=True,
        check=True,
    )
    scan = json.loads(proc.stdout)

    # Guard against a plugin-set mismatch making the comparison meaningless: if the
    # baseline was generated with a different plugin set, a fresh scan could miss (or
    # invent) findings and the set-diff would be comparing apples to oranges.
    baseline_plugins = {p["name"] for p in baseline.get("plugins_used", [])}
    scan_plugins = {p["name"] for p in scan.get("plugins_used", [])}
    if baseline_plugins != scan_plugins:
        print(
            "error: detect-secrets plugin set drifted from the baseline; regenerate "
            f".secrets.baseline.\n  baseline-only: {sorted(baseline_plugins - scan_plugins)}"
            f"\n  scan-only:     {sorted(scan_plugins - baseline_plugins)}",
            file=sys.stderr,
        )
        return 2

    new = _scanned_pairs(scan) - _audited_pairs(baseline)
    if new:
        print(
            "SECRET-SCAN FAILED: findings not accounted for in the audited "
            ".secrets.baseline:",
            file=sys.stderr,
        )
        for filename, hashed in sorted(new):
            print(f"  {filename}  (hashed_secret={hashed})", file=sys.stderr)
        print(
            "\nIf these are false positives, audit them into the baseline:\n"
            "  detect-secrets scan --baseline .secrets.baseline && "
            "detect-secrets audit .secrets.baseline\n"
            "then commit the updated baseline. If any is a REAL secret, remove it.",
            file=sys.stderr,
        )
        return 1

    print("secret-scan OK: every finding is accounted for in the audited baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
