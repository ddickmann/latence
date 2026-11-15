"""Persistent IRIs (ticket 57): an element's IRI is minted once and kept in the ledger.

An element's id is built from catalog names (``concept:<schema>.<table>``), so a rename gives
the element a new id. Its IRI is not built from the id each time: the first build that states
the element mints it (an opaque name, ``x`` and 14 hex digits of the id's SHA-1, the name the
Turtle file always used) and appends it to ``iris.jsonl`` in the build folder; every later build
reads it from there.

**A rename keeps the IRI only when a person confirmed it** (``renames.jsonl``, written by
``latence-ontology rename``; the owner's decision is open, option (a) of ticket 57 is built). A
confirmed rename says that the element the ledger knew as ``from`` is the element the build now
states as ``to``: the new id takes the old IRI. It holds only when the old id is gone from the
build and the new one is in it, and the new one has no IRI of its own yet; otherwise it is
refused with the reason. Elements whose ids name the renamed element (a relation from a renamed
table, an attribute of it) take their old IRIs too, where the old id is gone and the id with the
new name in place of the old one is in the build. Drift alone (``drift.py``) carries nothing.

An IRI is never given to two elements: a minted name already held by another element is minted
again with a counter. The same ledger and the same build give the same IRIs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LEDGER = "iris.jsonl"
RENAMES = "renames.jsonl"


def minted(element_id: str, n: int = 0) -> str:
    """The opaque local name minted for an id (the n-th try, where an earlier one is taken)."""
    text = element_id if n == 0 else f"{element_id}\u0000{n}"
    return "x" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]


def read_renames(out: Path) -> list[dict[str, Any]]:
    """The renames people confirmed, oldest first."""
    path = out / RENAMES
    if not path.exists():
        return []
    lines = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not all(str(r.get(k) or "").strip() for k in ("from", "to", "by", "at")):
            raise ValueError(f"{path}:{n}: a rename names the old id, the new id, who and the day")
        lines.append(r)
    return lines


def confirm(out: Path, old: str, new: str, by: str, at: str, reason: str = "") -> dict[str, Any]:
    """Append a person's confirmation that ``old`` was renamed ``new``."""
    if old == new:
        raise ValueError("a rename names two different ids")
    line = {"from": old, "to": new, "by": by, "at": at, **({"reason": reason} if reason else {})}
    out.mkdir(parents=True, exist_ok=True)
    with (out / RENAMES).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
    return line


def _object_of(element_id: str) -> str | None:
    """The catalog object a concept's id names (``concept:s.t`` -> ``s.t``)."""
    if element_id.startswith("concept:") and not element_id.startswith("concept:value:"):
        return element_id.removeprefix("concept:")
    return None


def _named(name: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\w])")


class Iris:
    """The ledger of IRIs of one build folder."""

    def __init__(self, out: Path) -> None:
        self.path = out / LEDGER
        self.of: dict[str, str] = {}
        self.held: dict[str, str] = {}  # a local name to the element holding it
        self.lines: list[dict[str, Any]] = []
        self.renamed: set[tuple[str, str]] = set()  # the renames the ledger already carried
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._hold(str(r["element"]), str(r["iri"]))
                    if r.get("renamed_from"):
                        self.renamed.add((str(r["renamed_from"]), str(r["element"])))

    def _hold(self, element_id: str, local: str) -> None:
        retired = self.held.get(local)
        if retired is not None and retired != element_id:
            self.of.pop(retired, None)  # the IRI went with a rename; the old id holds none
        self.of[element_id] = local
        self.held[local] = element_id

    def _append(self, element_id: str, local: str, **more: Any) -> None:
        self._hold(element_id, local)
        self.lines.append({"element": element_id, "iri": local, **more})

    def settle(self, ids: Iterable[str], renames: list[dict[str, Any]]) -> dict[str, Any]:
        """Give every id of the build its IRI: carried over the confirmed renames, then minted
        for what is new. Returns what was carried and what was refused, with the reason."""
        now = list(dict.fromkeys(ids))
        present = set(now)
        carried: list[dict[str, str]] = []
        refused: list[dict[str, str]] = []
        for r in renames:
            old, new = str(r["from"]), str(r["to"])
            if (old, new) in self.renamed:
                continue  # carried by an earlier build
            why = (
                f"{old} holds no IRI in the ledger"
                if old not in self.of
                else f"{old} is still in the build"
                if old in present
                else f"{new} is not in the build"
                if new not in present
                else None
            )
            if why is not None:
                refused.append({"from": old, "to": new, "reason": why})
                continue
            if new in self.of:
                if self.of[new] != self.of[old]:
                    refused.append(
                        {"from": old, "to": new, "reason": f"{new} already holds an IRI"}
                    )
                continue
            pairs = [(old, new)]
            was, became = _object_of(old), _object_of(new)
            if was and became:
                pattern = _named(was)
                for other in sorted(self.of):
                    if other == old or other in present or not pattern.search(other):
                        continue
                    renamed = pattern.sub(became, other)
                    if renamed in present and renamed not in self.of:
                        pairs.append((other, renamed))
            for a, b in pairs:
                local = self.of[a]
                self._append(b, local, renamed_from=a, by=str(r["by"]), at=str(r["at"]))
                self.renamed.add((a, b))
                carried.append({"from": a, "to": b, "iri": local})
        for element_id in now:
            if element_id in self.of:
                continue
            n = 0
            while minted(element_id, n) in self.held:
                n += 1
            self._append(element_id, minted(element_id, n))
        return {"carried": carried, "refused": refused}

    def save(self) -> int:
        """Append what this build minted or carried; returns how many lines."""
        if self.lines:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                for line in self.lines:
                    fh.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
        n, self.lines = len(self.lines), []
        return n


def element_ids(export: dict[str, Any]) -> list[str]:
    """The ids of an export that the ledger gives IRIs: concepts and their attributes,
    relations, measures, classes and the structure's elements, in the export's order. Rows of
    master data and value sets are named by their keys and are not in it."""
    ids: list[str] = []
    for c in export["concepts"]:
        ids.append(c["id"])
        ids.extend(f"{c['id']}#{a['column']}" for a in c.get("attributes") or [])
    ids.extend(r["id"] for r in export["relations"])
    ids.extend(m["id"] for m in export["measures"])
    classes = export.get("classes") or {}
    ids.extend(c["id"] for part in ("top", "classes") for c in classes.get(part, []))
    for part in ("roll_ups", "part_of", "equivalent", "describes"):
        ids.extend(e["id"] for e in (export.get("structure") or {}).get(part, []) if "id" in e)
    return ids
