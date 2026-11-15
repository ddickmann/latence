"""The crosswalk to a host's concept keys (ticket 57), read from the host's alias table.

A host keys its concepts ``<kind>:<namespace>.<local>`` and hands the builder, per library, an
alias file in its published format ``metis.alias/1`` (the host's ticket 326, written by its
command ``metis-dev concepts alias``)::

    {"format": "metis.alias/1", "catalog": "<library>@<version>:<digest>",
     "version": "sha256:<hex>", "namespace": "...", "keys": {...}, "platform": {...},
     "renamed": {...}}

``keys`` maps each served concept's kind-prefixed bare id (``measure:claim_count``) to its key;
``platform`` maps a concept imported from the platform's metric views to the platform ids it
was read from, ``<kind>:<source>.<view>.<field>`` (``<source>`` is the host's name for the
catalog source in its layer, not the platform's catalog or schema); ``renamed`` maps retired
bare ids to their replacements. ``version`` is the table's digest: ``sha256:`` and the hex
SHA-256 of the RFC 8785 canonical JSON of ``{namespace, keys, platform, renamed}``. The file is
read by the format it states; nothing of the host's code is used. A file of another format, or
whose version is not its table's digest, is refused.

An element of the ontology names a key only where the table says so: a measure read from a
metric view, and a value set whose column a metric view's dimension reads (``source.<column>``
on the view's source object), are matched by the platform id of that view's field, compared on
its kind, view and field (the source part is the host's own name and is not compared). An
element no platform id of the table names gets no key and is listed (every measure and value
set, a value set no view's dimension reads among them): a key is never derived from a name or a
label. An element two keys would claim gets none, and is listed. A view whose
name two schemas of the estate hold is matched by neither, since a platform id cannot say which
it names; it is listed. A member of a matched value set is the pair of the set's key and its
code (never one string).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .reader import MetricView

ALIAS_FORMAT = "metis.alias/1"
TABLE_FIELDS = ("format", "catalog", "version", "namespace", "keys", "platform", "renamed")
KEY = re.compile(
    r"^(dataset|dimension|measure|action):[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
    r"(\.[A-Za-z0-9_]+)*$"
)  # a host's key: its kinds (an `action` key is read, and never matches a measure or value set)


def _canonical(value: Any) -> str:
    """RFC 8785 canonical JSON of a value of strings, lists and objects (an alias table holds
    nothing else): members sorted by the UTF-16 code units of their names, no white space."""
    if isinstance(value, dict):
        names = sorted(value, key=lambda n: str(n).encode("utf-16-be"))
        return "{" + ",".join(f"{_canonical(str(n))}:{_canonical(value[n])}" for n in names) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"an alias table holds strings, lists and objects, not {type(value).__name__}")


def table_version(table: dict[str, Any]) -> str:
    """The version an alias file states: the digest of its table."""
    body = {f: table[f] for f in ("namespace", "keys", "platform", "renamed")}
    return "sha256:" + hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _platform_tail(platform_id: str, where: str) -> tuple[str, str, str]:
    """A platform id's kind, view and field (``<kind>:<source>.<view>.<field>``)."""
    kind, _, rest = platform_id.partition(":")
    parts = rest.split(".")
    if not kind or len(parts) != 3 or not all(parts):
        shape = "<kind>:<source>.<view>.<field>"
        raise ValueError(f"{where}: the platform id {platform_id!r} is not {shape}")
    return kind, parts[1], parts[2]


def read_alias_tables(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """The alias files (``metis.alias/1``), each with the digest of the file it was read from."""
    tables = []
    for path in paths:
        raw = path.read_bytes()
        table = json.loads(raw)
        if not isinstance(table, dict) or table.get("format") != ALIAS_FORMAT:
            stated = table.get("format") if isinstance(table, dict) else None
            raise ValueError(f"{path}: an alias file of format {stated!r}: {ALIAS_FORMAT} is read")
        missing = [f for f in TABLE_FIELDS if f not in table]
        if missing:
            raise ValueError(f"{path}: an alias file ({ALIAS_FORMAT}) states {', '.join(missing)}")
        if table_version(table) != table["version"]:
            raise ValueError(f"{path}: the alias file's version is not its table's digest")
        for bare, key in table["keys"].items():
            if not KEY.match(str(key)):
                raise ValueError(f"{path}: the key of {bare}, {key!r}, is no concept key")
        for bare, ids in table["platform"].items():
            for pid in ids:
                _platform_tail(str(pid), f"{path}: {bare}")
        tables.append({**table, "read_from": hashlib.sha256(raw).hexdigest()[:16]})
    return tables


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
    it was read from), and a note of the tables read (with their version and catalog), of what
    was matched and of what was not."""
    index: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for n, table in enumerate(tables):
        for bare, ids in sorted((table.get("platform") or {}).items()):
            for pid in ids:
                index.setdefault(_platform_tail(str(pid), str(bare)), []).append((n, str(bare)))
    count: dict[str, int] = {}
    for v in views:
        count[v.name] = count.get(v.name, 0) + 1
    twice = sorted(name for name, k in count.items() if k > 1)
    by_name = {v.name: v for v in views}
    wanted: dict[str, set[tuple[str, str, str]]] = {}
    for m in export["measures"]:
        held = by_name.get(str(m.get("view")))
        named = held is not None and held.name not in twice
        wanted[m["id"]] = {("measure", held.name, m["name"])} if held and named else set()
    for c in export["concepts"]:
        if c.get("kind") != "reference":
            continue
        reading = [
            (v, d)
            for v in views
            if v.source in (c.get("objects") or [])
            for d in v.dimensions
            if d.expr.strip() == f"source.{c['column']}"
        ]
        wanted[c["id"]] = {("dimension", v.name, d.name) for v, d in reading if v.name not in twice}
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
            {f: str(t[f]) for f in ("format", "namespace", "version", "catalog", "read_from")}
            for t in tables
        ],
        "matched": len(matches),
        "without_key": sorted(set(wanted) - set(matches) - {s["element"] for s in several}),
        "several_keys": several,
        **({"views_named_twice": twice} if twice else {}),
    }
    return matches, note
