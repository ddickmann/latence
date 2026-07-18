"""A REAL, local, offline Iceberg REST catalog for the #184 end-to-end tests.

This is deliberately **not** a fake of the thing under test. The client side is the real,
unmodified ``pyiceberg.catalog.rest.RestCatalog`` (that is what the Export uses in production), it
speaks real HTTP over a loopback socket, and every request is answered by a real
``pyiceberg.catalog.sql.SqlCatalog`` (the OSS reference catalog implementation, SQLite-backed) that
writes real Iceberg metadata and real Avro manifests to a local warehouse directory. The only thing
written here is the HTTP routing layer that the Apache ``iceberg-rest-fixture`` container would
otherwise provide — so the tests exercise the genuine protocol and the genuine metadata, with no
network, no container and no cloud.

That matters for false-green: because the catalog behind the routes is a real implementation, it
REJECTS what a real catalog rejects — an unknown namespace, a duplicate table, a stale commit
requirement, a data file whose Parquet schema does not match the table. A hand-written stub that
merely recorded calls would accept all of those and prove nothing.

Route coverage is the subset of the Iceberg REST spec the Export's client actually calls
(``GET /v1/config``, namespace create/list/load/exists, table create/load/list/exists, and the
``POST`` commit endpoint). Anything else answers 404 with a spec-shaped error body, so an untested
code path fails loudly instead of silently appearing to work.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pyiceberg.catalog.rest import CreateTableRequest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.table import CommitTableRequest

# pyiceberg's RestCatalog joins multi-level namespaces with the unit-separator character
# (the Iceberg REST spec's default ``namespace_separator``), so a path segment splits on it.
_NAMESPACE_SEPARATOR = "\x1f"


def _namespace(segment: str) -> tuple[str, ...]:
    return tuple(segment.split(_NAMESPACE_SEPARATOR))


class _IcebergRestHandler(BaseHTTPRequestHandler):
    """Maps Iceberg REST routes onto a real ``SqlCatalog``."""

    catalog: SqlCatalog

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the stdlib access log so a test run stays readable."""

    # -- plumbing -------------------------------------------------------------------------

    def _send(self, code: int, body: object | None = None) -> None:
        payload = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"error": {"message": message, "type": "Error", "code": code}})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length)

    def _segments(self) -> list[str]:
        path = urlparse(self.path).path
        return [unquote(part) for part in path.strip("/").split("/") if part]

    @staticmethod
    def _table_response(table: Any) -> dict[str, Any]:
        # The table metadata is serialised by pyiceberg's OWN model, so the wire bytes are exactly
        # the spec-shaped JSON a real REST catalog returns (aliases, nested structures and all) —
        # and the client parses it back with its own ``TableResponse``.
        return {
            "metadata-location": table.metadata_location,
            "metadata": json.loads(table.metadata.model_dump_json(by_alias=True)),
            "config": {},
        }

    # -- routes ---------------------------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib dispatch name
        parts = self._segments()
        try:
            if len(parts) == 3 and parts[1] == "namespaces":
                exists = self.catalog.namespace_exists(_namespace(parts[2]))
                self._send(204 if exists else 404)
                return
            if len(parts) == 5 and parts[3] == "tables":
                exists = self.catalog.table_exists((*_namespace(parts[2]), parts[4]))
                self._send(204 if exists else 404)
                return
        except Exception as exc:  # noqa: BLE001 - surface catalog failures as 500, like a server
            self._error(500, repr(exc))
            return
        self._send(404)

    def do_GET(self) -> None:  # noqa: N802 - stdlib dispatch name
        parts = self._segments()
        try:
            if parts == ["v1", "config"]:
                self._send(200, {"defaults": {}, "overrides": {}})
                return
            if parts == ["v1", "namespaces"]:
                namespaces = [list(ns) for ns in self.catalog.list_namespaces()]
                self._send(200, {"namespaces": namespaces})
                return
            if len(parts) == 3 and parts[1] == "namespaces":
                namespace = _namespace(parts[2])
                properties = self.catalog.load_namespace_properties(namespace)
                self._send(200, {"namespace": list(namespace), "properties": properties})
                return
            if len(parts) == 4 and parts[3] == "tables":
                tables = self.catalog.list_tables(_namespace(parts[2]))
                identifiers = [{"namespace": list(t[:-1]), "name": t[-1]} for t in tables]
                self._send(200, {"identifiers": identifiers})
                return
            if len(parts) == 5 and parts[3] == "tables":
                table = self.catalog.load_table((*_namespace(parts[2]), parts[4]))
                self._send(200, self._table_response(table))
                return
        except NoSuchTableError as exc:
            self._error(404, str(exc))
            return
        except NoSuchNamespaceError as exc:
            self._error(404, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface catalog failures as 500, like a server
            self._error(500, repr(exc))
            return
        self._error(404, f"no route for GET {self.path}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib dispatch name
        parts = self._segments()
        raw = self._read_body()
        try:
            if parts == ["v1", "namespaces"]:
                self._create_namespace(raw)
                return
            if len(parts) == 5 and parts[3] == "tables":
                self._commit_table(parts, raw)
                return
            if len(parts) == 4 and parts[3] == "tables":
                self._create_table(parts, raw)
                return
        except (NamespaceAlreadyExistsError, TableAlreadyExistsError) as exc:
            self._error(409, str(exc))
            return
        except NoSuchNamespaceError as exc:
            self._error(404, str(exc))
            return
        except NoSuchTableError as exc:
            self._error(404, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface catalog failures as 500, like a server
            self._error(500, repr(exc))
            return
        self._error(404, f"no route for POST {self.path}")

    def _create_namespace(self, raw: bytes) -> None:
        payload = json.loads(raw)
        properties = payload.get("properties") or {}
        self.catalog.create_namespace(tuple(payload["namespace"]), properties)
        self._send(200, {"namespace": payload["namespace"], "properties": properties})

    def _create_table(self, parts: list[str], raw: bytes) -> None:
        payload = json.loads(raw)
        # The client omits null-valued optional fields; the spec model declares them required-but-
        # nullable, so defaulting them here is the same normalisation a real server performs.
        for key in ("location", "partition-spec", "write-order"):
            payload.setdefault(key, None)
        request = CreateTableRequest.model_validate(payload)
        table = self.catalog.create_table(
            identifier=(*_namespace(parts[2]), request.name),
            schema=request.table_schema,
            location=request.location,
            properties=request.properties,
        )
        self._send(200, self._table_response(table))

    def _commit_table(self, parts: list[str], raw: bytes) -> None:
        request = CommitTableRequest.model_validate_json(raw)
        table = self.catalog.load_table((*_namespace(parts[2]), parts[4]))
        # The real catalog validates the commit REQUIREMENTS here (assert-ref-snapshot-id and
        # friends), so a stale or conflicting commit is rejected exactly as in production.
        response = self.catalog.commit_table(table, tuple(request.requirements), tuple(request.updates))
        self._send(200, json.loads(response.model_dump_json(by_alias=True)))


@contextmanager
def local_rest_catalog(root: Path) -> Iterator[str]:
    """Run a local Iceberg REST catalog over ``root``; yield its base URL.

    ``root`` holds both the catalog's SQLite metastore and its warehouse directory, so the whole
    catalog is a throwaway temp tree — offline, deterministic, and gone with the test.
    """
    warehouse = root / "warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "local",
        uri=f"sqlite:///{root / 'catalog.db'}",
        warehouse=f"file://{warehouse}",
    )
    handler = type("_BoundHandler", (_IcebergRestHandler,), {"catalog": catalog})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    hostname = host.decode() if isinstance(host, bytes) else host
    try:
        yield f"http://{hostname}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
