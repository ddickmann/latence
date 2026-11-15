"""Erasure when a column turns sensitive (ticket 54; the owner, 14 Nov).

A column read as ordinary in an earlier build may since be tagged sensitive. The build no longer
reads it, but what earlier builds wrote stays in the build folder: its values in the export, in
the decisions on pairs of its values, in files written from them. When a build finds a column
sensitive that the ledger shows was read as ordinary, and the value set over it is gone from
the build, every value of it that an earlier build wrote is erased from the folder:

- a line of a ``.jsonl`` file that holds one (the ledger, the decisions, the calls) is removed;
- any other file that holds one (a proposal file, a staged file, an old Turtle file) is removed;
  the build writes again what it writes;
- an erasure is appended to ``erasures.jsonl``: the column, the objects, the build, how many
  values, lines and files; never a value. A later build that finds nothing more records nothing.

This is the one place where the ledger is not append-only: a legal erasure outranks it (CONTEXT
``Purge``). The values are known from the export the earlier build wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REWRITTEN = (
    "ontology.json",
    "ontology.ttl",
    "ontology.shacl.ttl",
    "ontology.prov.ttl",
)  # the build writes these again from what it reads now


def turned_sensitive(
    prior: dict[str, list[dict[str, Any]]], objects: dict[str, dict[str, bool]]
) -> dict[str, list[str]]:
    """Column -> the objects where the ledger read it as ordinary and the catalog now says it
    is sensitive (``objects``: full name -> column -> sensitive)."""
    out: dict[str, set[str]] = {}
    for statements in prior.values():
        for a in statements:
            p = a["payload"]
            if a["kind"] == "concept" and p.get("kind") == "reference" and p.get("column"):
                # a value set: its values were read from each object it names
                col = str(p["column"])
                out.setdefault(col, set()).update(
                    str(o) for o in p.get("objects") or [] if objects.get(str(o), {}).get(col)
                )
                continue
            if a["kind"] != "concept" or not p.get("object_name"):
                continue
            obj = str(p["object_name"])
            for attr in p.get("attributes") or []:
                col = str(attr["column"])
                if not attr.get("sensitive") and objects.get(obj, {}).get(col):
                    out.setdefault(col, set()).add(obj)
    return {col: sorted(objs) for col, objs in sorted(out.items()) if objs}


def _values(export: dict[str, Any], column: str) -> set[str]:
    concept = f"concept:value:{column}"
    found = {str(n["label"]) for n in export.get("nodes", []) if n.get("concept") == concept}
    found |= {
        str(a)
        for n in export.get("nodes", [])
        if n.get("concept") == concept
        for a in n.get("aliases") or []
    }
    prefix = f"value:{column}:"
    for pair in [
        *export.get("review", []),
        *(p for s in (export.get("label_requests") or {}).values() for p in s),
    ]:
        found |= {
            str(pair[e]).removeprefix(prefix)
            for e in ("a", "b")
            if str(pair.get(e, "")).startswith(prefix)
        }
    return found


def erase(
    out: Path,
    previous: dict[str, Any] | None,
    columns: dict[str, list[str]],
    still_read: set[str],
    build_no: int,
) -> list[dict[str, Any]]:
    """Erase from ``out`` every value of each column that turned sensitive and whose value set
    the build no longer holds (``still_read``: the columns it still reads as a value set).
    Returns the erasures recorded."""
    if previous is None:
        return []
    log = out / "erasures.jsonl"
    done = []
    for column, objs in columns.items():
        if column in still_read:
            continue
        values = _values(previous, column)
        if not values:
            continue
        marks = (
            {json.dumps(v, ensure_ascii=False) for v in values}
            | {f"value:{column}:{v}" for v in values}
            | {f"{column}:{v}~" for v in values}
        )
        lines_removed, files_removed = 0, []
        for path in sorted(out.iterdir()):
            if (
                not path.is_file()
                or path.name in ("erasures.jsonl", "ledger_builds.jsonl")
                or path.name in REWRITTEN
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # a binary file (a records file) is written from what the build reads now
            if not any(m in text for m in marks):
                continue
            if path.suffix == ".jsonl":
                kept = [line for line in text.splitlines() if not any(m in line for m in marks)]
                lines_removed += len(text.splitlines()) - len(kept)
                path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            else:
                path.unlink()
                files_removed.append(path.name)
        record = {
            "column": column,
            "objects": objs,
            "build": build_no,
            "values": len(values),
            "lines_removed": lines_removed,
            "files_removed": files_removed,
        }
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        done.append(record)
    return done
