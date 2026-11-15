"""The receipt a host's tool writes when it has placed a proposal file, read back.

An approved element the receipt lists as placed, on the claim the element still states, says
since which release of the host's library it is in the layer (``in_layer``): a concept, a
pair, a step between value sets or an equivalence (the element of a ``broader`` or ``match``
entry), and a concept's labels in one language (``labels.in_layer``, by language). What a
receipt lists as already there or as refused says nothing about since when, and is not read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latence_core.errors import ConfigError

RECEIPT = "layer-receipt/1"  # the format of the receipt this package reads


def read_receipts(paths: tuple[Path, ...]) -> dict[tuple[str, str], dict[str, str]]:
    """What a host's receipts say was placed: (element id, the claim it was placed on) -> the
    library and the release that placed it (with the release's digest and the tag it was built
    from, where the receipt states them: the hub's releases, ticket 59). Of several receipts that
    placed the same, the first given counts, so give them oldest first."""
    taken: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fmt = doc.get("format") if isinstance(doc, dict) else None
        if fmt != RECEIPT:
            raise ConfigError(f"{path}: the receipt is {fmt!r}; this build reads {RECEIPT}")
        if not doc.get("library") or not doc.get("version"):
            raise ConfigError(f"{path}: a receipt names the library and the release that "
                              "took the entries up")
        release = {"library": str(doc["library"]), "version": str(doc["version"]),
                   **({"digest": str(doc["digest"])} if doc.get("digest") else {}),
                   **({"tag": str(doc["tag"])} if doc.get("tag") else {})}
        for line in doc.get("placed") or []:
            taken.setdefault((str(line["element"]), str(line["claim"])), release)
    return taken


def in_layer(export: Mapping[str, Any],
             taken: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[dict[str, Any], int]:
    """The export with ``in_layer`` (the library and its release) on each element that a
    receipt says was placed as it states now, and that a person's approval is in force on.
    Returns the export and how many elements say so. Without a receipt nothing is said."""
    def marked(el: dict[str, Any]) -> dict[str, Any]:
        release = taken.get((el["id"], str(el.get("claim"))))
        approved = (el.get("decision") or {}).get("verdict") == "approved"
        return {**el, "in_layer": dict(release)} if release is not None and approved else el

    def worded(c: dict[str, Any]) -> dict[str, Any]:
        said = c.get("labels")
        if not isinstance(said, Mapping):
            return c
        placed = {}
        for lang, claim in sorted((said.get("claims") or {}).items()):
            release = taken.get((f"labels:{c['id']}@{lang}", str(claim)))
            approved = ((said.get("decisions") or {}).get(lang) or {}).get("verdict") == "approved"
            if release is not None and approved:
                placed[lang] = dict(release)
        return {**c, "labels": {**said, "in_layer": placed}} if placed else c

    if not taken:
        return dict(export), 0
    concepts = [worded(marked(c)) for c in export["concepts"]]
    review = [marked(r) for r in export["review"]]
    structure = {
        part: [marked(el) for el in els] if part in ("roll_ups", "equivalent") else els
        for part, els in (export.get("structure") or {}).items()
    }
    count = sum("in_layer" in el for el in concepts + review)
    count += sum("in_layer" in el for part in ("roll_ups", "equivalent")
                 for el in structure.get(part, []))
    count += sum(len((c.get("labels") or {}).get("in_layer") or {}) for c in concepts)
    out = {**export, "concepts": concepts, "review": review}
    if "structure" in export:
        out["structure"] = structure
    return out, count
