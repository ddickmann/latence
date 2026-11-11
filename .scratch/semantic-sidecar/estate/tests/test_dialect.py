"""Every statement the engine writes goes through the reader's dialect: the same model comes
out of a warehouse that quotes names with backticks and calls text STRING (ticket 33)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from estate.induce import induce
from estate.instances import build_instances
from estate.reader import CatalogObject
from estate.reader_duckdb import DuckDBReader
from estate.structure import broader_edges, build_structure

from .conftest import OPTIONS


class OtherDialect:
    """A reader over the same data that speaks another SQL: names in backticks, text as
    STRING. Like such a warehouse, it reads a double-quoted name as a text value, which would
    make every count wrong without an error; here it refuses the statement instead."""

    def __init__(self, inner: DuckDBReader) -> None:
        self.inner, self.statements = inner, []
        self.objects, self.declared_keys, self.metric_views = inner.objects, inner.declared_keys, inner.metric_views

    def ident(self, name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    def as_text(self, expr: str) -> str:
        return f"CAST({expr} AS STRING)"

    def quote(self, obj: CatalogObject) -> str:
        return f"{self.ident(obj.schema)}.{self.ident(obj.name)}"

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        if '"' in sql or "VARCHAR" in sql.upper():
            raise ValueError(f"not this warehouse's SQL: {sql[:120]}")
        self.statements.append(sql)
        return self.inner.query(sql.replace("`", '"'))


def _all(reader: Any) -> str:
    model = induce(reader, **OPTIONS, without_personal_columns=("shop.dealers",))
    inst = build_instances(reader, model, personal_objects=("shop.owners", "shop.vehicles"), without_personal_columns=("shop.dealers",), resolve=("make",))
    st = build_structure(reader, model, ("shop.owners", "shop.vehicles"))
    edges = broader_edges(reader, st, {n["id"] for n in inst["nodes"]})
    return json.dumps({"model": {k: model[k] for k in ("concepts", "relations", "measures")}, "nodes": inst["nodes"], "edges": inst["edges"],
                       "review": inst["review"], "structure": {k: v for k, v in st.items() if not k.startswith("_")}, "broader": edges}, sort_keys=True, default=str)


def test_the_same_model_comes_out_of_a_warehouse_with_another_dialect(reader: DuckDBReader) -> None:
    other = OtherDialect(reader)
    assert _all(other) == _all(reader)
    assert len(other.statements) > 50  # it was asked, not bypassed


def test_the_duckdb_reader_quotes_a_name_that_holds_a_quote(reader: DuckDBReader) -> None:
    assert reader.ident('odd"name') == '"odd""name"' and reader.as_text("x") == "CAST(x AS VARCHAR)"
    with pytest.raises(ValueError, match="not this warehouse"):
        OtherDialect(reader).query('SELECT "a" FROM t')
