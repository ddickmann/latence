"""Mentions in the estate's documents, linked to its entities (ticket 42: modality (b) meets the
estate's rows).

A folder of documents (``.txt`` and ``.md``) is read, and every place a document names a record
of a master list or a value of a value set by its name (its label or another spelling the build
holds) becomes a mention: the document, its digest, the half-open character span, the text and
the row it names. A name is found whole: case-insensitive, never inside a longer word, the
longest name at a place winning. A name that two rows carry is a mention of neither: it is kept
with both candidates and left for a person (no silent over-merge). Names shorter than three
characters are not looked for.

Only rows the export holds are looked for, so a row of an object about persons, which the export
never holds, is never found; the documents themselves are not copied into the export, only the
spans.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

SUFFIXES = (".txt", ".md")
SHORTEST = 3


def _names(export: dict[str, Any]) -> dict[str, set[str]]:
    """Each name (folded) -> the rows that carry it."""
    out: dict[str, set[str]] = {}
    for n in export.get("nodes", []):
        for name in [n.get("label"), *(n.get("aliases") or [])]:
            text = str(name or "").strip()
            if len(text) >= SHORTEST:
                out.setdefault(text.casefold(), set()).add(str(n["id"]))
    return out


def find_mentions(
    export: dict[str, Any], folder: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = _names(export)
    if not names:
        return [], {"documents": 0, "mentions": 0, "linked": 0, "ambiguous": 0}
    pattern = re.compile(
        r"(?<!\w)("
        + "|".join(re.escape(n) for n in sorted(names, key=lambda x: (-len(x), x)))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    mentions: list[dict[str, Any]] = []
    documents = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
    for path in documents:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        for m in pattern.finditer(text):
            rows = sorted(names[m.group(1).casefold()])
            mention: dict[str, Any] = {
                "document": str(path.relative_to(folder)),
                "digest": digest,
                "start": m.start(),
                "end": m.end(),
                "text": m.group(1),
            }
            if len(rows) == 1:
                mention["node"] = rows[0]
            else:
                mention["candidates"] = rows
            mentions.append(mention)
    linked = sum("node" in m for m in mentions)
    return mentions, {
        "documents": len(documents),
        "mentions": len(mentions),
        "linked": linked,
        "ambiguous": len(mentions) - linked,
    }
