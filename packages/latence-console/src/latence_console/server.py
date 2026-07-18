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
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from latence_console import render
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
        path = unquote(urlsplit(target).path)
        if path in ("", "/"):
            return Response.html(
                200,
                render.render_run_list(self._index.storage_uri, self._index.list_runs()),
            )
        if path.startswith(_RUN_PREFIX):
            return self._run_page(path[len(_RUN_PREFIX) :].strip("/"))
        return Response.html(404, render.render_not_found(f"There is no page at {path!r}."))

    def _run_page(self, run_id: str) -> Response:
        # A run id that is not a plain path segment is refused here — before it can be
        # concatenated into a Storage URI. `/run/../../etc/passwd` therefore reads nothing.
        if not is_run_id(run_id):
            return Response.html(404, render.render_not_found(f"{run_id!r} is not a run id."))
        summary = self._index.summary(run_id)
        if summary is None:
            return Response.html(
                404, render.render_not_found(f"There is no run called {run_id!r} here.")
            )
        return Response.html(200, render.render_run(summary, self._index.report(run_id)))


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
