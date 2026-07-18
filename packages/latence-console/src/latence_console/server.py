"""The console itself: a pure request→response core plus a thin localhost socket shim.

:class:`ConsoleApp` is the whole console. It maps a method+path to a :class:`Response` and
holds **no state between requests** — no session, no cache, no writable field. Every answer
is re-derived from the files on Storage at the moment of the request, which is what makes
"read-only over files, no server state" (ADR-0010/0017) a structural property rather than a
promise: there is nowhere for state to live.

:func:`serve` is the socket half — twenty lines of :mod:`http.server` that hand each request
to the app. It refuses to bind anywhere but the loopback interface, so the console is never
accidentally exposed on a network.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

from latence_console import browse, render
from latence_console.artifacts import RunArtifacts
from latence_console.runs import RunIndex, is_run_id

if TYPE_CHECKING:
    from collections.abc import Callable

    from latence_core.storage import Storage

# The methods a read-only console answers. Everything else — POST, PUT, PATCH, DELETE and
# any invented verb — is refused with 405 before a path is even parsed. This is the console's
# read-only posture expressed where it is enforceable, not as a config flag.
READ_METHODS = frozenset({"GET", "HEAD"})

_RUN_PREFIX = "/run/"

# The page loads nothing: no script, no image, no font, no fetch. Only the inlined stylesheet
# is permitted. A browser therefore CANNOT be talked into making a network request from a
# console page, whatever ends up in a rendered field.
CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'"


class NotLoopbackError(ValueError):
    """Raised when the console is asked to bind somewhere other than this machine."""


@dataclass(frozen=True)
class Response:
    """One rendered answer: status, content type, body."""

    status: int
    content_type: str
    body: bytes

    @classmethod
    def html(cls, status: int, markup: str) -> Response:
        return cls(status, "text/html; charset=utf-8", markup.encode("utf-8"))


class ConsoleApp:
    """The read-only run console. Accepts the :class:`RunIndex` it reads through."""

    def __init__(self, index: RunIndex) -> None:
        self._index = index

    @classmethod
    def over_storage(cls, storage: Storage, storage_uri: str) -> ConsoleApp:
        return cls(RunIndex(storage, storage_uri))

    @property
    def storage_uri(self) -> str:
        return self._index.storage_uri

    def respond(self, method: str, target: str) -> Response:
        """Answer one request. Pure: same files in, same bytes out, nothing written."""
        if method.upper() not in READ_METHODS:
            return Response.html(405, render.render_method_not_allowed())
        split = urlsplit(target)
        path = unquote(split.path)
        if path in ("", "/"):
            return Response.html(
                200,
                render.render_run_list(self._index.storage_uri, self._index.list_runs()),
            )
        if path.startswith(_RUN_PREFIX):
            return self._run_page(path[len(_RUN_PREFIX) :].strip("/"), split.query)
        return Response.html(404, render.render_not_found(f"There is no page at {path!r}."))

    def _run_page(self, tail: str, query: str) -> Response:
        """Route ``/run/<run_id>[/<view>]`` — the overview plus the #188 debugging views."""
        run_id, _, view = tail.partition("/")
        # A run id that is not a plain path segment is refused here — before it can be
        # concatenated into a Storage URI. `/run/../../etc/passwd` therefore reads nothing.
        if not is_run_id(run_id):
            return Response.html(404, render.render_not_found(f"{run_id!r} is not a run id."))
        summary = self._index.summary(run_id)
        if summary is None:
            return Response.html(
                404, render.render_not_found(f"There is no run called {run_id!r} here.")
            )
        view = view.strip("/")
        if not view:
            return Response.html(200, render.render_run(summary, self._index.report(run_id)))
        handler = self._VIEWS.get(view)
        if handler is None:
            return Response.html(
                404, render.render_not_found(f"There is no {view!r} view of a run.")
            )
        # One fresh reader per request: the console holds no index between answers.
        return handler(self, run_id, self._index.artifacts(run_id), _param(query, "id"))

    # -- the #188 views ----------------------------------------------------

    def _quarantine_view(
        self, run_id: str, artifacts: RunArtifacts, _id: str | None
    ) -> Response:
        return Response.html(
            200,
            render.render_quarantine(
                run_id, browse.quarantine_entries(artifacts), browse.flag_entries(artifacts)
            ),
        )

    def _chunks_view(self, run_id: str, artifacts: RunArtifacts, _id: str | None) -> Response:
        return Response.html(200, render.render_chunks(run_id, browse.chunk_rows(artifacts)))

    def _chunk_view(
        self, run_id: str, artifacts: RunArtifacts, record_id: str | None
    ) -> Response:
        # The id is MATCHED against records already read — never concatenated into a Storage
        # URI — so a crafted id can only fail to match, not address a file.
        chunk = artifacts.chunk(record_id) if record_id else None
        if chunk is None:
            return Response.html(
                404,
                render.render_not_found(
                    f"There is no chunk {record_id!r} in run {run_id!r}."
                    if record_id
                    else "That link is missing the chunk it should show."
                ),
            )
        return Response.html(200, render.render_chunk(run_id, browse.inspect_chunk(artifacts, chunk)))

    def _graph_view(self, run_id: str, artifacts: RunArtifacts, _id: str | None) -> Response:
        return Response.html(
            200,
            render.render_graph(
                run_id, browse.entity_views(artifacts), browse.edge_views(artifacts)
            ),
        )

    def _entity_view(
        self, run_id: str, artifacts: RunArtifacts, node_id: str | None
    ) -> Response:
        node = artifacts.node(node_id) if node_id else None
        if node is None:
            return Response.html(
                404,
                render.render_not_found(
                    f"There is no graph entity {node_id!r} in run {run_id!r}."
                    if node_id
                    else "That link is missing the entity it should show."
                ),
            )
        return Response.html(
            200,
            render.render_entity(
                run_id,
                browse.entity_view(artifacts, node),
                browse.edge_views(artifacts, node.node_id),
            ),
        )

    _VIEWS: ClassVar[dict[str, Callable[[ConsoleApp, str, RunArtifacts, str | None], Response]]] = {
        "quarantine": _quarantine_view,
        "chunks": _chunks_view,
        "chunk": _chunk_view,
        "graph": _graph_view,
        "entity": _entity_view,
    }


def _param(query: str, name: str) -> str | None:
    """One query parameter, percent-decoded — ``None`` when absent or empty.

    Record ids (``sha256:…#chunk-0``) ride the query rather than the path precisely so the
    path space stays plain segments; ``parse_qs`` does the decoding.
    """
    values = parse_qs(query, keep_blank_values=False).get(name)
    return values[0] if values else None


def _require_loopback(host: str) -> str:
    """Reject any bind address that is not this machine (the console is never a service)."""
    if host in ("localhost", "127.0.0.1", "::1"):
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    msg = (
        f"The run console binds to the loopback interface only; refusing to bind {host!r}. "
        "It is a local, read-only viewer, not a shared service."
    )
    raise NotLoopbackError(msg)


def build_server(
    app: ConsoleApp, host: str = "127.0.0.1", port: int = 8787
) -> ThreadingHTTPServer:
    """Build (but do not start) the loopback HTTP server that serves ``app``."""
    _require_loopback(host)

    class Handler(BaseHTTPRequestHandler):
        server_version = "latence-console"
        sys_version = ""

        def _answer(self, *, with_body: bool) -> None:
            response = app.respond(self.command or "GET", self.path)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            # A local viewer over local files: no caching, no referrer leakage, and a CSP
            # that forbids the page from loading anything at all from the network.
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if with_body:
                self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=False)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=True)

        def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=True)

        def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=True)

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler dispatch name
            self._answer(with_body=True)

        def log_message(self, format: str, *args: object) -> None:
            """Silence the per-request stderr chatter; the CLI prints what matters."""

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    app: ConsoleApp,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve ``app`` on loopback until interrupted. Blocks."""
    httpd = build_server(app, host, port)
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    shown = bound_host.decode() if isinstance(bound_host, bytes) else str(bound_host)
    url = f"http://{shown}:{bound_port}/"
    if on_ready is not None:
        on_ready(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
