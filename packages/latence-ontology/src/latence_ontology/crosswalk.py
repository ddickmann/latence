"""The crosswalk to a host's concept keys (ticket 57), read from the host's alias table.

A host keys its concepts ``<kind>:<namespace>.<local>`` and states, per namespace, an alias
table: each served concept's bare id to its key, and for a concept imported from the platform's
metric views the platform ids it was read from (``<kind>:<catalog>.<schema>.<view>.<field>``).
The table is read as the host writes it (``{namespace, keys, platform, renamed}``); nothing of
the host's code is used.

An element of the ontology names a key only where the table says so: a measure read from a
metric view, and a value set whose column a metric view's dimension reads (``source.<column>``
on the view's source object), are matched by the platform id of that view's field. The catalog
part of a platform id is not compared (an estate file need not name the catalog), the rest is
compared whole. An element no platform id of the table names gets no key: a key is never
derived from a name or a label. An element two keys would claim gets none, and is listed. A
member of a matched value set is the pair of the set's key and its code (never one string).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .reader import MetricView

TABLE_FIELDS = ("namespace", "keys", "platform")


def read_alias_tables(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """The alias tables, each with the digest of the file it was read from."""
    tables = []
    for path in paths:
        raw = path.read_bytes()
        table = json.loads(raw)
        if not isinstance(table, dict) or any(f not in table for f in TABLE_FIELDS):
            raise ValueError(
                f"{path}: an alias table states {', '.join(TABLE_FIELDS)} (the host's 297)"
            )
        tables.append({**table, "read_from": hashlib.sha256(raw).hexdigest()[:16]})
    return tables


def _tail(platform_id: str) -> tuple[str, str]:
    """A platform id's kind and what follows its catalog: ``(kind, schema.view.field)``."""
    kind, _, rest = platform_id.partition(":")
    return kind, rest.partition(".")[2]


def _key(table: dict[str, Any], bare: str) -> str | None:
    found = table["keys"].get(bare)
    if found is None:
        kind, _, name = bare.partition(":")
        new = (table.get("renamed") or {}).get(name)
        found = table["keys"].get(f"{kind}:{new}") if new is not None else None
    return str(found) if found else None


def crosswalk(
    export: dict[str, Any], views: list[MetricView], tables: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Per element id its match (``key``, the table's ``namespace`` and the digest of the file
    it was read from), and a note of what was matched and what not."""
    index: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for n, table in enumerate(tables):
        for bare, ids in sorted((table.get("platform") or {}).items()):
            for pid in ids:
                index.setdefault(_tail(str(pid)), []).append((n, str(bare)))
    by_name = {v.name: v for v in views}
    wanted: dict[str, set[tuple[str, str]]] = {}
    for m in export["measures"]:
        v = by_name.get(str(m.get("view")))
        if v is not None and v.schema:
            wanted[m["id"]] = {("measure", f"{v.schema}.{v.name}.{m['name']}")}
    for c in export["concepts"]:
        if c.get("kind") != "reference":
            continue
        tails = {
            ("dimension", f"{v.schema}.{v.name}.{d.name}")
            for v in views
            if v.schema and v.source in (c.get("objects") or [])
            for d in v.dimensions
            if d.expr.strip() == f"source.{c['column']}"
        }
        if tails:
            wanted[c["id"]] = tails
    matches: dict[str, dict[str, Any]] = {}
    several: list[dict[str, Any]] = []
    for element, tails in sorted(wanted.items()):
        keys = {
            (_key(tables[n], bare), n)
            for tail in tails
            for n, bare in index.get(tail, [])
            if _key(tables[n], bare) is not None
        }
        found = sorted({k for k, _ in keys if k})
        if len(found) == 1:
            n = min(n for k, n in keys if k == found[0])
            matches[element] = {
                "key": found[0],
                "namespace": str(tables[n]["namespace"]),
                "read_from": tables[n]["read_from"],
            }
        elif len(found) > 1:
            several.append({"element": element, "keys": found})
    note = {
        "tables": [
            {"namespace": str(t["namespace"]), "read_from": t["read_from"]} for t in tables
        ],
        "matched": len(matches),
        "without_key": sorted(set(wanted) - set(matches) - {s["element"] for s in several}),
        "several_keys": several,
    }
    return matches, note
