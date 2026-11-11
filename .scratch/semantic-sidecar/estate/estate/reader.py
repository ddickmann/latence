"""What the engine reads from a warehouse, behind one small interface.

A reader gives the objects of a catalog (tables, views, with their columns, comments and
tags), the keys the catalog declares, the metric views, and a way to run an aggregate query
inside the warehouse. Everything later (profiling, induction, export) talks to this interface
only, so the local DuckDB mirror and Unity Catalog are interchangeable.

Object ids are the catalog's own where it has them (Unity Catalog's table id) and are
otherwise built from the object's full name. Every element the engine derives is keyed on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    comment: str = ""
    tags: tuple[str, ...] = ()
    sensitive: bool = False  # never read for values, never shown as instances


@dataclass(frozen=True)
class CatalogObject:
    id: str
    schema: str
    name: str
    kind: str  # "table" | "view"
    columns: tuple[Column, ...]
    comment: str = ""
    rows: int | None = None

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass(frozen=True)
class DeclaredKey:
    object: str  # full name
    columns: tuple[str, ...]
    kind: str  # "primary" | "foreign"
    references: str | None = None  # full name of the referenced object
    referenced_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Field:
    name: str
    expr: str
    display_name: str = ""
    comment: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Join:
    name: str
    source: str  # full name of the joined object, catalog prefix removed
    on: str


@dataclass(frozen=True)
class MetricView:
    name: str
    source: str  # full name of the source object, catalog prefix removed
    comment: str = ""
    joins: tuple[Join, ...] = ()
    dimensions: tuple[Field, ...] = ()
    measures: tuple[Field, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class Reader(Protocol):
    def objects(self) -> list[CatalogObject]: ...

    def declared_keys(self) -> list[DeclaredKey]: ...

    def metric_views(self) -> list[MetricView]: ...

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        """Run one aggregate query inside the warehouse. Callers never select a sensitive column."""
        ...

    def quote(self, obj: CatalogObject) -> str:
        """The object's name as this warehouse's SQL writes it."""
        ...

    def ident(self, name: str) -> str:
        """A column's name as this warehouse's SQL writes it. Dialects differ: what one reads
        as a quoted name another reads as a text value, and counts go wrong without an error.
        Every statement the engine writes takes its names from here."""
        ...

    def as_text(self, expr: str) -> str:
        """The expression cast to text, in this warehouse's name for the text type."""
        ...
