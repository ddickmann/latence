"""The reader over Unity Catalog and a SQL warehouse, by HTTPS.

What it reads and from where:

- **objects** with their columns, comments and ids: the catalog's table listing
  (``GET /api/2.1/unity-catalog/tables``, paged). An object's id is the catalog's own table id.
- **tags**: one statement over ``information_schema.column_tags``. A column is sensitive when
  the estate file names it or when it carries one of the tags the estate calls sensitive. A
  sensitive column is never selected by any statement of the engine.
- **declared keys**: each table's own entry (``GET .../tables/{full_name}``), whose
  constraints the listing leaves out.
- **metric views**: the listing's entries of type ``METRIC_VIEW``. The catalog stores a
  definition without its comments and display names; those sit on the listed columns, and the
  definition is put together again from both.
- **queries**: the statement API (``POST /api/2.0/sql/statements/``), polled until it ends,
  every further part of the result fetched at once. A statement that fails stops the build
  with the statement's id. Values arrive as text and are read by the column types the result
  states.

Credentials come from the environment only (``from_env``): a host, a warehouse id, and either
a token or the id and secret of a service principal. Nothing here writes to the catalog.

**Proven against a stand-in server, not against a workspace.** The endpoints and field names
are those the host project's own adapter uses against a live workspace; whether this reader
meets a real catalog as it meets the stand-in is for a run inside one to show.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

import yaml

from .metric_views import parse_view
from .reader import CatalogObject, Column, DeclaredKey, MetricView

TABLES = "/api/2.1/unity-catalog/tables"
STATEMENTS = "/api/2.0/sql/statements/"
RUNNING = ("PENDING", "RUNNING")
WHOLE = ("BYTE", "SHORT", "INT", "LONG")
FRACTION = ("FLOAT", "DOUBLE", "DECIMAL")
VIEW_TYPES = ("VIEW", "MATERIALIZED_VIEW")


class WarehouseError(RuntimeError):
    """The catalog or the warehouse did not give what was asked for."""


def _http(method: str, url: str, headers: dict[str, str], body: bytes | None = None, timeout: float = 60) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310 - the scheme is checked where the host is given
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
            return answer.status, dict(answer.headers), answer.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


class DatabricksReader:
    """``schemas`` are the layers to read, by the names the engine uses; in the catalog each
    carries ``schema_prefix`` (the layer ``clean`` is the schema ``nhtsa_clean``)."""

    def __init__(self, host: str, token: str, catalog: str, schemas: Iterable[str], warehouse_id: str, *, schema_prefix: str = "",
                 sensitive: Iterable[str] = (), sensitive_tags: Iterable[str] = (), page: int = 100, wait: str = "30s",
                 poll_seconds: float = 1.0, give_up_after: float = 900.0, http: Any = _http) -> None:
        if not host.startswith("https://") and not host.startswith("http://127.0.0.1"):
            raise ValueError("the host is given as https://<workspace>")
        self.host, self.token, self.catalog, self.schemas = host.rstrip("/"), token, catalog, tuple(schemas)
        self.warehouse_id, self.prefix = warehouse_id, schema_prefix
        self.sensitive, self.sensitive_tags = frozenset(sensitive), frozenset(t.lower() for t in sensitive_tags)
        self.page, self.wait, self.poll_seconds, self.give_up_after, self._http = page, wait, poll_seconds, give_up_after, http
        self.name = catalog
        self._listed: list[dict[str, Any]] | None = None
        self.statements: list[str] = []

    @classmethod
    def from_env(cls, catalog: str, schemas: Iterable[str], **more: Any) -> "DatabricksReader":
        """``DATABRICKS_HOST`` and ``DATABRICKS_WAREHOUSE_ID``, and either ``DATABRICKS_TOKEN``
        or a service principal's ``DATABRICKS_CLIENT_ID`` and ``DATABRICKS_CLIENT_SECRET``."""
        missing = [k for k in ("DATABRICKS_HOST", "DATABRICKS_WAREHOUSE_ID") if not os.environ.get(k)]
        if missing:
            raise WarehouseError(f"not set in the environment: {', '.join(missing)}")
        host = os.environ["DATABRICKS_HOST"]
        token = os.environ.get("DATABRICKS_TOKEN") or cls._service_token(host, more.get("http", _http))
        return cls(host, token, catalog, schemas, os.environ["DATABRICKS_WAREHOUSE_ID"], **more)

    @staticmethod
    def _service_token(host: str, http: Any) -> str:
        cid, secret = os.environ.get("DATABRICKS_CLIENT_ID"), os.environ.get("DATABRICKS_CLIENT_SECRET")
        if not cid or not secret:
            raise WarehouseError("not set in the environment: DATABRICKS_TOKEN, or DATABRICKS_CLIENT_ID with DATABRICKS_CLIENT_SECRET")
        basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
        status, _headers, body = http("POST", f"{host.rstrip('/')}/oidc/v1/token", {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
                                      b"grant_type=client_credentials&scope=all-apis")
        if status != 200:
            raise WarehouseError(f"the workspace gave no token to the service principal (HTTP {status})")
        return str(json.loads(body)["access_token"])

    # ---- the wire -----------------------------------------------------------------------------
    def _call(self, method: str, path: str, query: dict[str, Any] | None = None, body: dict[str, Any] | None = None, what: str = "") -> dict[str, Any]:
        url = self.host + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json", **({"Content-Type": "application/json"} if body is not None else {})}
        payload = json.dumps(body).encode() if body is not None else None
        for attempt in (1, 2):
            status, answered, raw = self._http(method, url, headers, payload)
            if status == 429 and attempt == 1:  # asked too fast: once more, after the pause it names
                time.sleep(min(float(answered.get("Retry-After", 1) or 1), 30.0))
                continue
            break
        if status in (401, 403):
            raise WarehouseError(f"{what or path}: not allowed (HTTP {status}); the token is wrong, expired or lacks the right")
        if status != 200:
            raise WarehouseError(f"{what or path}: HTTP {status}: {raw[:300].decode(errors='replace')}")
        return dict(json.loads(raw or b"{}"))

    def _schema(self, layer: str) -> str:
        return f"{self.prefix}{layer}"

    def _list(self) -> list[dict[str, Any]]:
        if self._listed is None:
            found: list[dict[str, Any]] = []
            for layer in self.schemas:
                token: str | None = None
                while True:
                    query = {"catalog_name": self.catalog, "schema_name": self._schema(layer), "max_results": self.page, **({"page_token": token} if token else {})}
                    page = self._call("GET", TABLES, query, what=f"the tables of {self.catalog}.{self._schema(layer)}")
                    found += [{**t, "_layer": layer} for t in page.get("tables") or []]
                    token = page.get("next_page_token") or None
                    if not token:
                        break
            self._listed = sorted(found, key=lambda t: (t["_layer"], t["name"]))
        return self._listed

    # ---- the reader ---------------------------------------------------------------------------
    def ident(self, name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    def as_text(self, expr: str) -> str:
        return f"CAST({expr} AS STRING)"

    def quote(self, obj: CatalogObject) -> str:
        return f"{self.ident(self.catalog)}.{self.ident(self._schema(obj.schema))}.{self.ident(obj.name)}"

    def _tags(self) -> dict[tuple[str, str, str], set[str]]:
        schemas = ", ".join("'" + self._schema(layer).replace("'", "''") + "'" for layer in self.schemas)
        rows = self.query(f"SELECT schema_name, table_name, column_name, tag_name FROM {self.ident(self.catalog)}.information_schema.column_tags "
                          f"WHERE schema_name IN ({schemas}) ORDER BY 1, 2, 3, 4")
        out: dict[tuple[str, str, str], set[str]] = {}
        for schema, table, column, tag in rows:
            out.setdefault((str(schema), str(table), str(column)), set()).add(str(tag).lower())
        return out

    def objects(self) -> list[CatalogObject]:
        tags = self._tags()
        out = []
        for t in self._list():
            if t.get("table_type") == "METRIC_VIEW":
                continue
            layer, name = t["_layer"], t["name"]
            columns = []
            for c in t.get("columns") or []:
                carried = tags.get((self._schema(layer), name, c["name"]), set())
                sensitive = f"{layer}.{name}.{c['name']}" in self.sensitive or bool(carried & self.sensitive_tags)
                columns.append(Column(c["name"], str(c.get("type_text") or c.get("type_name") or ""), str(c.get("comment") or ""), tuple(sorted(carried)), sensitive))
            obj = CatalogObject(id=f"uc:{t['table_id']}", schema=layer, name=name, kind="view" if t.get("table_type") in VIEW_TYPES else "table",
                                columns=tuple(columns), comment=str(t.get("comment") or ""))
            rows = self.query(f"SELECT count(*) FROM {self.quote(obj)}")[0][0]
            out.append(CatalogObject(obj.id, obj.schema, obj.name, obj.kind, obj.columns, obj.comment, int(rows)))
        return out

    def declared_keys(self) -> list[DeclaredKey]:
        out = []
        for t in self._list():
            if t.get("table_type") == "METRIC_VIEW":
                continue
            full = f"{self.catalog}.{self._schema(t['_layer'])}.{t['name']}"
            one = self._call("GET", f"{TABLES}/{urllib.parse.quote(full, safe='')}", what=f"the table {full}")
            for constraint in one.get("table_constraints") or []:
                primary, foreign = constraint.get("primary_key_constraint"), constraint.get("foreign_key_constraint")
                if primary:
                    out.append(DeclaredKey(f"{t['_layer']}.{t['name']}", tuple(primary.get("child_columns") or ()), "primary"))
                if foreign:
                    parent = str(foreign.get("parent_table") or "").split(".")
                    layer = parent[-2][len(self.prefix):] if len(parent) >= 2 and parent[-2].startswith(self.prefix) else (parent[-2] if len(parent) >= 2 else "")
                    out.append(DeclaredKey(f"{t['_layer']}.{t['name']}", tuple(foreign.get("child_columns") or ()), "foreign",
                                           f"{layer}.{parent[-1]}", tuple(foreign.get("parent_columns") or ())))
        return out

    def metric_views(self) -> list[MetricView]:
        out = []
        for t in self._list():
            if t.get("table_type") != "METRIC_VIEW":
                continue
            doc = yaml.safe_load(t.get("view_definition") or "") or {}
            listed = {c["name"]: c for c in t.get("columns") or []}
            for group in ("dimensions", "measures"):
                for item in doc.get(group) or []:
                    column = listed.get(item.get("name"), {})
                    meta = column.get("metadata") or {}
                    meta = json.loads(meta) if isinstance(meta, str) and meta.strip().startswith("{") else meta if isinstance(meta, dict) else {}
                    item.setdefault("comment", column.get("comment") or "")
                    item.setdefault("display_name", meta.get("display_name") or "")
                    item.setdefault("synonyms", meta.get("synonyms") or [])
            doc.setdefault("comment", t.get("comment") or "")
            out.append(parse_view(t["name"], yaml.safe_dump(doc, sort_keys=False), self.prefix))
        return sorted(out, key=lambda v: v.name)

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        self.statements.append(sql)
        answer = self._call("POST", STATEMENTS, body={"warehouse_id": self.warehouse_id, "statement": sql, "disposition": "INLINE", "format": "JSON_ARRAY",
                                                         "wait_timeout": self.wait, "on_wait_timeout": "CONTINUE"}, what="a statement")
        sid = str(answer.get("statement_id") or "")
        started = time.monotonic()
        while (answer.get("status") or {}).get("state") in RUNNING:
            if time.monotonic() - started > self.give_up_after:
                self._call("POST", f"{STATEMENTS}{sid}/cancel", body={}, what=f"cancelling statement {sid}")
                raise WarehouseError(f"statement {sid} was still running after {self.give_up_after:.0f} s and was cancelled")
            time.sleep(self.poll_seconds)
            answer = self._call("GET", f"{STATEMENTS}{sid}", what=f"statement {sid}")
        status = answer.get("status") or {}
        if status.get("state") != "SUCCEEDED":
            message = (status.get("error") or {}).get("message") or "no reason given"
            raise WarehouseError(f"statement {sid} ended as {status.get('state')}: {message}")
        manifest = answer.get("manifest") or {}
        kinds = [str(c.get("type_name") or "").upper() for c in (manifest.get("schema") or {}).get("columns") or []]
        part = answer.get("result") or {}
        rows = list(part.get("data_array") or [])
        while part.get("next_chunk_index") is not None:
            part = self._call("GET", f"{STATEMENTS}{sid}/result/chunks/{int(part['next_chunk_index'])}", what=f"statement {sid}, the next part of its result")
            rows += part.get("data_array") or []
        return [tuple(_value(v, kinds[k] if k < len(kinds) else "") for k, v in enumerate(row)) for row in rows]


def _value(text: Any, kind: str) -> Any:
    """A value of the result, which arrives as text, in the type the result states."""
    if text is None:
        return None
    if kind in WHOLE:
        return int(text)
    if kind in FRACTION:
        return float(text)
    if kind == "BOOLEAN":
        return str(text).lower() == "true"
    return text
