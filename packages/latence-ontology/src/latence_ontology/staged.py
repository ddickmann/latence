"""Approved changes to the catalog, staged for its owner (ticket 52; the owner, 14 Nov).

The engine never writes to Unity Catalog. What a person approved that the catalog should own
(a key it does not declare, a reference between two tables) is written as a file of statements
in the warehouse's dialect, each preceded by the decision it rests on, for the catalog's owner
to review and apply. Nothing here runs a statement or calls the catalog.

Staged: an approved concept's key, as an informational primary key (``NOT ENFORCED``); an
approved relation between two tables, as an informational foreign key. Not staged: what the
catalog already declares; a value set or a relation to one (no catalog object holds it); an
element decided on another statement than the build makes now (stale); an approval by the
engine (only a person's approval goes to the catalog's owner); a key of more than one column of
a concept that is not a table of the estate.
"""

from __future__ import annotations

import re
from typing import Any

HEADER = (
    "-- Staged for the catalog's owner to review and apply. Written from approved decisions of a",
    "-- build; the engine runs none of it. Each statement says the decision it rests on.",
)


def _ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _table(full: str, catalog: str, prefix: str) -> str:
    schema, _, table = full.partition(".")
    return f"{_ident(catalog)}.{_ident(prefix + schema)}.{_ident(table)}"


def _constraint(*parts: str) -> str:
    return _ident(re.sub(r"[^a-z0-9]+", "_", "_".join(parts).lower()).strip("_"))


def staged_statements(
    export: dict[str, Any], decisions: dict[str, dict[str, Any]], catalog: str, prefix: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The statements (each with the element and the decision it rests on), and a report."""
    statements: list[dict[str, Any]] = []
    not_staged: list[dict[str, str]] = []
    concepts = {c["id"]: c for c in export["concepts"]}
    elements = {**concepts, **{r["id"]: r for r in export["relations"]}}
    for subject, d in sorted(decisions.items()):
        el = elements.get(subject)
        if el is None or d.get("verdict") != "approved":
            continue
        declared = (
            any(str(s.get("code", "")).startswith("declared") for s in el.get("signals") or [])
            or el.get("status") == "declared"
        )
        if d.get("by") == "engine":
            not_staged.append({"subject": subject, "why": "not_a_person"})
            continue
        if d.get("claim") != el.get("claim"):
            not_staged.append({"subject": subject, "why": "stale"})
            continue
        if declared:
            not_staged.append({"subject": subject, "why": "declared_in_the_catalog"})
            continue
        origin = f"{subject}, claim {el['claim']}, approved by {d['by']}" + (
            f" on {str(d['at'])[:10]}" if d.get("at") else ""
        )
        if subject in concepts:
            key = list(el.get("key") or [])
            if not el.get("object_name") or not key:
                continue
            table = str(el["object_name"]).partition(".")[2]
            sql = (
                f"ALTER TABLE {_table(str(el['object_name']), catalog, prefix)} ADD CONSTRAINT "
                f"{_constraint(table, 'pk')} "
                f"PRIMARY KEY ({', '.join(_ident(k) for k in key)}) NOT ENFORCED"
            )
        else:
            child, parent = str(el["from"]), str(el["to"])
            if child.startswith("concept:value:") or parent.startswith("concept:value:"):
                continue  # a value set is no catalog object
            child_name, parent_name = (
                child.removeprefix("concept:"),
                parent.removeprefix("concept:"),
            )
            columns, to_columns = list(el.get("columns") or []), list(el.get("to_columns") or [])
            if not columns or len(columns) != len(to_columns):
                not_staged.append({"subject": subject, "why": "no_columns"})
                continue
            sql = (
                f"ALTER TABLE {_table(child_name, catalog, prefix)} ADD CONSTRAINT "
                f"{_constraint(child_name.partition('.')[2], *columns, 'fk')} FOREIGN KEY "
                f"({', '.join(_ident(c) for c in columns)}) "
                f"REFERENCES {_table(parent_name, catalog, prefix)} "
                f"({', '.join(_ident(c) for c in to_columns)}) NOT ENFORCED"
            )
        statements.append({"subject": subject, "origin": origin, "sql": sql})
    return statements, {"statements": len(statements), "not_staged": not_staged}


def to_sql(statements: list[dict[str, Any]]) -> str:
    lines = [*HEADER, ""]
    for s in statements:
        lines += [f"-- {s['origin']}", s["sql"] + ";", ""]
    return "\n".join(lines)
