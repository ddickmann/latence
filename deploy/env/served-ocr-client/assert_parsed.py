"""Assert the served OCR endpoint's markdown actually reached the exported RAG corpus (#196).

Gate 3 of ``smoke.sh``. It scans every exported artifact under the run's storage root for the
sentinel tokens :mod:`make_fixture` rendered onto the page, and fails when they are absent.

Why this is a real assertion and not a flag: the smoke stack's Source accepts image/PDF extensions
only and its single Parse Provider is ``parser.lighton_vllm``. There is therefore NO path by which
this text can appear in the export except "the page was rasterised, POSTed to the vLLM LightOn-OCR
endpoint, and the returned markdown was assembled onto the PageMap". A stubbed, skipped or
silently-failing Parse produces an empty corpus and this exits non-zero.

    python assert_parsed.py <storage-root-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Kept in sync with make_fixture.py by IMPORTING it, so the rendered page and the assertion about
# it cannot drift apart — a second hand-typed copy of the sentinel is how such a gate goes hollow.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixture import SENTINEL_TOKENS


def exported_text(root: Path) -> str:
    """Every exported JSONL the run wrote, concatenated and upper-cased."""
    chunks = []
    for path in sorted(root.rglob("*.jsonl")):
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "".join(chunks).upper()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"FAIL: storage root {root} does not exist — the run produced nothing.")
        return 1

    exports = sorted(root.rglob("*.jsonl"))
    if not exports:
        print(f"FAIL: no exported corpus JSONL under {root}.")
        print("      The run produced no RAG corpus, so no page was parsed. This is the failure")
        print("      shape of a Parse stage that never reached the served OCR endpoint.")
        return 1

    haystack = exported_text(root)
    missing = [token for token in SENTINEL_TOKENS if token.upper() not in haystack]
    if missing:
        print(f"FAIL: the exported corpus is missing {len(missing)} sentinel token(s): {missing}")
        print(f"      Scanned: {', '.join(str(p) for p in exports)}")
        print("      The page text rendered by make_fixture.py did not come back through the")
        print("      served OCR endpoint. Check the vLLM server log and $LATENCE_OCR_BASE_URL.")
        return 1

    print(f"OK: all {len(SENTINEL_TOKENS)} sentinel tokens survived the round trip")
    print(f"    fixture page -> rasteriser -> vLLM LightOn-OCR endpoint -> PageMap -> {exports[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
