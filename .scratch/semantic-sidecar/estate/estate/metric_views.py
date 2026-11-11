"""Read metric-view definitions (the YAML Unity Catalog stores, versions 0.1 and 1.1).

Input is either one YAML document per view, or the text several definitions are printed as,
each headed by a line ``-- <schema>.<view>``. A measure or join that cannot be read is kept
with what could be read; nothing is guessed.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from .reader import Field, Join, MetricView


def _strip_catalog(name: str, schema_prefix: str = "") -> str:
    """``catalog.schema.table`` -> ``schema.table``; a schema prefix shared by the catalog's
    layers (``nhtsa_clean`` -> ``clean``) is removed so names match the local mirror."""
    parts = str(name).split(".")
    schema, table = (parts[-2], parts[-1]) if len(parts) >= 2 else ("", parts[-1])
    if schema_prefix and schema.startswith(schema_prefix):
        schema = schema[len(schema_prefix) :]
    return f"{schema}.{table}" if schema else table


def _fields(items: Any) -> tuple[Field, ...]:
    out = []
    for it in items or []:
        if not isinstance(it, dict) or "name" not in it:
            continue
        out.append(Field(
            name=str(it["name"]),
            expr=str(it.get("expr", "")),
            display_name=str(it.get("display_name", "") or ""),
            comment=str(it.get("comment", "") or ""),
            synonyms=tuple(str(s) for s in (it.get("synonyms") or [])),
        ))
    return tuple(out)


def _joins(items: Any, schema_prefix: str) -> tuple[Join, ...]:
    out: list[Join] = []
    for it in items or []:
        if not isinstance(it, dict) or "name" not in it or "source" not in it:
            continue
        # YAML 1.1 reads the bare key `on` as the boolean true; the condition is under either.
        on = it.get("on", it.get(True, ""))
        out.append(Join(str(it["name"]), _strip_catalog(it["source"], schema_prefix), str(on or "")))
        out.extend(_joins(it.get("joins"), schema_prefix))  # nested joins hang off the joined object
    return tuple(out)


def parse_view(name: str, text: str, schema_prefix: str = "") -> MetricView:
    doc = yaml.safe_load(text) or {}
    if not isinstance(doc, dict) or "source" not in doc:
        raise ValueError(f"metric view {name!r} has no source")
    return MetricView(
        name=name,
        source=_strip_catalog(doc["source"], schema_prefix),
        comment=str(doc.get("comment", "") or ""),
        joins=_joins(doc.get("joins"), schema_prefix),
        dimensions=_fields(doc.get("dimensions")),
        measures=_fields(doc.get("measures")),
        raw=doc,
    )


def parse_listing(text: str, schema_prefix: str = "") -> list[MetricView]:
    """Several definitions in one text, each headed by ``-- <schema>.<view>``."""
    views = []
    for block in re.split(r"(?m)^-- (?=\S)", text):
        if not block.strip():
            continue
        head, _, body = block.partition("\n")
        views.append(parse_view(head.strip().split(".")[-1], body, schema_prefix))
    return views


def join_columns(on: str) -> tuple[str, str] | None:
    """``source.dealer_id = internal_dealers.dealer_id`` -> (child column, parent column).
    Only a single equality between the source and the joined object is read."""
    m = re.fullmatch(r"\s*source\.(\w+)\s*=\s*\w+\.(\w+)\s*", on)
    return (m.group(1), m.group(2)) if m else None
