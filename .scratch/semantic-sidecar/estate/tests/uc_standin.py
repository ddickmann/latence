"""A stand-in for Unity Catalog and a SQL warehouse, for the reader's tests.

It answers the calls the reader makes, over the toy estate in DuckDB, and it can disagree the
way the real services do: it refuses a wrong token, lists two tables at a time, answers the
first listing with "too many requests", lets every third statement be pending first, returns a
result in parts of five rows, fails a statement that is not in its dialect, and returns every
value as text. It knows nothing of the reader's code; it is written from the calls alone.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from typing import Any

from estate.reader_duckdb import DuckDBReader

TYPE_NAMES = (("BIGINT", "LONG"), ("HUGEINT", "LONG"), ("INTEGER", "INT"), ("DOUBLE", "DOUBLE"), ("DECIMAL", "DECIMAL"), ("BOOLEAN", "BOOLEAN"), ("DATE", "DATE"))
PAGE, PART = 2, 5


class StandIn:
    def __init__(self, inner: DuckDBReader, *, catalog: str = "demo", prefix: str = "acme_", warehouse: str = "wh1", tags: dict[tuple[str, str, str], list[str]] | None = None,
                 metric_views: list[dict[str, Any]] | None = None, constraints: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.inner, self.catalog, self.prefix, self.warehouse = inner, catalog, prefix, warehouse
        self.tags, self.constraints = tags or {}, constraints or {}
        self.tables = [self._table(o) for o in inner.objects()] + (metric_views or [])
        self.statements: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.kept: dict[str, dict[str, Any]] = {}
        self.cancelled: list[str] = []
        self.busy_once = True
        self.hang = False  # every statement stays pending for ever

    def _table(self, o: Any) -> dict[str, Any]:
        return {"name": o.name, "schema_name": self.prefix + o.schema, "catalog_name": self.catalog, "table_type": "VIEW" if o.kind == "view" else "MANAGED",
                "table_id": hashlib.sha1(o.full_name.encode()).hexdigest()[:12], "comment": o.comment or None,
                "columns": [{"name": c.name, "type_text": c.type.lower(), "type_name": c.type, "comment": c.comment or None} for c in o.columns]}

    # ---- the wire -----------------------------------------------------------------------------
    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None = None, timeout: float = 60) -> tuple[int, dict[str, str], bytes]:
        parsed = urllib.parse.urlsplit(url)
        path, query = urllib.parse.unquote(parsed.path), dict(urllib.parse.parse_qsl(parsed.query))
        self.calls.append((method, path))
        if headers.get("Authorization") == "Bearer no-rights":
            return 403, {}, b'{"error_code": "PERMISSION_DENIED"}'
        if headers.get("Authorization") != "Bearer good":
            return 401, {}, b'{"error_code": "UNAUTHENTICATED"}'
        status, answer = self._route(method, path, query, json.loads(body) if body else None)
        return status, ({"Retry-After": "0"} if status == 429 else {}), json.dumps(answer).encode()

    def _route(self, method: str, path: str, query: dict[str, str], body: Any) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path == "/api/2.1/unity-catalog/tables":
            if self.busy_once:
                self.busy_once = False
                return 429, {"error_code": "RESOURCE_EXHAUSTED"}
            if query.get("catalog_name") != self.catalog:
                return 404, {"error_code": "CATALOG_DOES_NOT_EXIST"}
            mine = [t for t in self.tables if t["schema_name"] == query.get("schema_name")]
            start = int(query.get("page_token") or 0)
            page = {"tables": mine[start:start + PAGE]}
            if start + PAGE < len(mine):
                page["next_page_token"] = str(start + PAGE)
            return 200, page
        if method == "GET" and path.startswith("/api/2.1/unity-catalog/tables/"):
            full = path.rsplit("/", 1)[-1]
            found = next((t for t in self.tables if f"{t['catalog_name']}.{t['schema_name']}.{t['name']}" == full), None)
            return (200, {**found, "table_constraints": self.constraints.get(full, [])}) if found else (404, {"error_code": "TABLE_DOES_NOT_EXIST"})
        if method == "POST" and path == "/api/2.0/sql/statements/":
            return 200, self._submit(body)
        if method == "GET" and (m := re.fullmatch(r"/api/2.0/sql/statements/(\w+)", path)):
            kept = self.kept[m.group(1)]
            return 200, ({"statement_id": m.group(1), "status": {"state": "PENDING"}} if self.hang else kept["first"])
        if method == "GET" and (m := re.fullmatch(r"/api/2.0/sql/statements/(\w+)/result/chunks/(\d+)", path)):
            return 200, self.kept[m.group(1)]["parts"][int(m.group(2))]
        if method == "POST" and (m := re.fullmatch(r"/api/2.0/sql/statements/(\w+)/cancel", path)):
            self.cancelled.append(m.group(1))
            return 200, {}
        return 404, {"error_code": "ENDPOINT_NOT_FOUND"}

    # ---- statements ---------------------------------------------------------------------------
    def _submit(self, body: dict[str, Any]) -> dict[str, Any]:
        wanted = {"warehouse_id": self.warehouse, "disposition": "INLINE", "format": "JSON_ARRAY", "on_wait_timeout": "CONTINUE"}
        if any(body.get(k) != v for k, v in wanted.items()) or not re.fullmatch(r"\d+s", str(body.get("wait_timeout"))):
            return {"statement_id": "s0", "status": {"state": "FAILED", "error": {"message": "the request is not one this warehouse takes"}}}
        sql = body["statement"]
        sid = f"s{len(self.statements) + 1}"
        self.statements.append(sql)
        answer = self._run(sid, sql)
        self.kept[sid] = answer
        if self.hang or (answer["first"]["status"]["state"] == "SUCCEEDED" and len(self.statements) % 3 == 0):
            return {"statement_id": sid, "status": {"state": "PENDING"}}  # not done within the wait: ask again
        return answer["first"]

    def _run(self, sid: str, sql: str) -> dict[str, Any]:
        def failed(why: str) -> dict[str, Any]:
            return {"first": {"statement_id": sid, "status": {"state": "FAILED", "error": {"message": why}}}, "parts": []}

        if '"' in sql or re.search(r"\bVARCHAR\b", sql, re.I):
            return failed("[PARSE_SYNTAX_ERROR] a name is written in backticks and text is STRING here")
        if "information_schema.column_tags" in sql:
            columns, rows = ["schema_name", "table_name", "column_name", "tag_name"], sorted((s, t, c, tag) for (s, t, c), found in self.tags.items() for tag in found)
            kinds = ["STRING"] * 4
        else:
            local = sql.replace(f"`{self.catalog}`.", "").replace("`", '"').replace(f'"{self.prefix}', '"')
            local = re.sub(r"\bAS STRING\b", "AS VARCHAR", local)
            try:
                cursor = self.inner.con.execute(local)
            except Exception as err:  # noqa: BLE001 - whatever the engine says is the warehouse's error here
                return failed(str(err).splitlines()[0])
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            kinds = [next((name for key, name in TYPE_NAMES if key in str(d[1]).upper()), "STRING") for d in cursor.description]
        as_text = [[None if v is None else str(v).lower() if isinstance(v, bool) else str(v) for v in row] for row in rows]
        parts = [as_text[i:i + PART] for i in range(0, len(as_text), PART)] or [[]]
        pieces = [{"chunk_index": i, "data_array": part, **({"next_chunk_index": i + 1} if i + 1 < len(parts) else {})} for i, part in enumerate(parts)]
        first = {"statement_id": sid, "status": {"state": "SUCCEEDED"},
                 "manifest": {"format": "JSON_ARRAY", "schema": {"columns": [{"name": n, "type_name": k} for n, k in zip(columns, kinds)]}, "total_row_count": len(rows)},
                 "result": pieces[0]}
        return {"first": first, "parts": pieces}
