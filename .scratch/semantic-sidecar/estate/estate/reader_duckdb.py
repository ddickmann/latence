"""The reader over a DuckDB file: the local mirror of an estate."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb

from .metric_views import parse_listing
from .reader import CatalogObject, Column, DeclaredKey, MetricView


class DuckDBReader:
    """``schemas`` lists the layers to read (a raw landing layer is usually left out).
    ``sensitive`` names columns as ``schema.table.column``: DuckDB has no tags, so the
    classification the catalog carries is given here. ``metric_views`` is the text of the
    views' definitions, which a DuckDB mirror does not hold."""

    def __init__(
        self,
        path: Path | str,
        schemas: Iterable[str],
        sensitive: Iterable[str] = (),
        metric_views: str = "",
        schema_prefix: str = "",
        name: str | None = None,
    ) -> None:
        self.con = duckdb.connect(str(path), read_only=True)
        self.schemas = tuple(schemas)
        self.sensitive = frozenset(sensitive)
        self._views_text, self._prefix = metric_views, schema_prefix
        self.name = name or Path(str(path)).stem

    def objects(self) -> list[CatalogObject]:
        marks = ", ".join("?" for _ in self.schemas)
        tables = self.con.execute(
            f"SELECT table_schema, table_name, table_type FROM information_schema.tables "
            f"WHERE table_schema IN ({marks}) ORDER BY 1, 2", list(self.schemas)).fetchall()
        comments = {(s, t): c or "" for s, t, c in self.con.execute(
            "SELECT schema_name, table_name, comment FROM duckdb_tables()").fetchall()}
        col_comments = {(s, t, c): k or "" for s, t, c, k in self.con.execute(
            "SELECT schema_name, table_name, column_name, comment FROM duckdb_columns()").fetchall()}
        out = []
        for schema, table, kind in tables:
            cols = self.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position", [schema, table]).fetchall()
            rows = self.con.execute(f'SELECT count(*) FROM "{schema}"."{table}"').fetchall()[0][0]
            out.append(CatalogObject(
                id=f"duckdb:{self.name}/{schema}.{table}",
                schema=schema, name=table, kind="view" if kind == "VIEW" else "table",
                columns=tuple(
                    Column(c, t, col_comments.get((schema, table, c), ""),
                           sensitive=f"{schema}.{table}.{c}" in self.sensitive)
                    for c, t in cols),
                comment=comments.get((schema, table), ""), rows=int(rows),
            ))
        return out

    def declared_keys(self) -> list[DeclaredKey]:
        rows = self.con.execute(
            "SELECT schema_name, table_name, constraint_type, constraint_column_names, "
            "referenced_table, referenced_column_names FROM duckdb_constraints() "
            "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')").fetchall()
        out = []
        for schema, table, kind, cols, ref, ref_cols in rows:
            if schema not in self.schemas:
                continue
            out.append(DeclaredKey(
                f"{schema}.{table}", tuple(cols), "primary" if kind == "PRIMARY KEY" else "foreign",
                f"{schema}.{ref}" if ref else None, tuple(ref_cols or ())))
        return out

    def metric_views(self) -> list[MetricView]:
        return parse_listing(self._views_text, self._prefix) if self._views_text else []

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        return self.con.execute(sql).fetchall()

    def quote(self, obj: CatalogObject) -> str:
        return f"{self.ident(obj.schema)}.{self.ident(obj.name)}"

    def ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def as_text(self, expr: str) -> str:
        return f"CAST({expr} AS VARCHAR)"
