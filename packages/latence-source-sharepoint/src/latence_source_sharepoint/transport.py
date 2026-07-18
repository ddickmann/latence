"""The HTTP transport seam the SharePoint connector talks to Microsoft Graph through.

Two things live here and nothing else: the tiny request/response shape the Graph client
needs (:class:`HttpResponse`, :class:`HttpTransport`) and the real, dependency-free
implementation over the Python standard library (:class:`UrllibTransport`).

The seam exists because the behaviour genuinely varies (the codebase-design bar: a seam only
where two REAL implementations exist, never speculatively):

* :class:`UrllibTransport` — production. Talks to ``https://graph.microsoft.com``.
* :class:`~latence_source_sharepoint.testing.FakeGraphTransport` — the shipped faithful
  offline fake of the Graph API, used by every offline test, by the conformance suite, and
  by an adopter who wants a dry run with no tenant.

An adopter with a corporate proxy / mTLS / a different HTTP client supplies a third one by
naming a factory in config (``transport: "my_pkg.module:build_transport"``) — no fork.

**Two security properties are enforced here, at the only place every byte passes through.**

1. **TLS only.** A request to a non-``https`` URL is refused before a single header is
   written, so a bearer token can never leave the process in the clear. (``http://`` is
   allowed only for a loopback host, so an adopter can point the connector at a local Graph
   emulator without disabling the guard globally.)
2. **Redirects are never followed automatically.** Graph answers
   ``GET /drives/{id}/items/{id}/content`` with a ``302`` to a short-lived pre-authenticated
   download URL on a *different host* (an Azure blob endpoint). ``urllib``'s default opener
   would follow it and re-send the ``Authorization`` header to that host — a token leak. The
   opener installed here returns the ``3xx`` to the caller instead, and
   :class:`~latence_source_sharepoint.graph.GraphClient` re-issues the download with **no**
   credentials, exactly as the real API expects. The faithful fake rejects a credentialed
   request to the download host with the same ``403`` the real blob endpoint returns, so a
   regression that re-introduces the leak fails the offline suite.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from latence_core.errors import ConfigError, ProviderError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from http.client import HTTPMessage
    from typing import IO

# Graph's own timeouts are generous; a stuck socket must not hang a pipeline forever.
DEFAULT_TIMEOUT_SECONDS = 60.0

# Hosts for which plain ``http://`` is tolerated (a local Graph emulator / test double).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response: status, headers, body. Deliberately smaller than any HTTP client's."""

    status: int
    headers: Mapping[str, str]
    body: bytes

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


@runtime_checkable
class HttpTransport(Protocol):
    """Send one HTTP request, return the response. No retries, no auth, no JSON — those are
    :class:`~latence_source_sharepoint.graph.GraphClient`'s job."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HttpResponse:
        """Perform the request and return the response, including for ``4xx``/``5xx``/``3xx``.

        Only a *transport* failure (DNS, connection refused, timeout, a refused non-TLS URL)
        raises; every status the server actually returned comes back as an
        :class:`HttpResponse` so the client can apply the Graph retry/refresh policy.
        """
        ...


def require_tls(url: str) -> None:
    """Refuse any URL that would put a bearer token on the wire in the clear.

    Raises :class:`~latence_core.errors.ConfigError` for a non-``https`` URL, except for a
    loopback host (a local Graph emulator). The URL is echoed in the message; a token never
    appears in a URL (Graph rejects query-string tokens, and so does the shipped fake), so
    the message is safe to log.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return
    msg = (
        f"Refusing to send a SharePoint request over {parsed.scheme or 'a schemeless URL'!r} "
        f"({url!r}): credentials may only travel over https (loopback http is allowed for a "
        "local Graph emulator). Fix 'graph_base_url' / 'login_base_url'."
    )
    raise ConfigError(msg)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into a returned response instead of a followed request.

    Returning ``None`` from ``redirect_request`` makes ``urllib`` raise ``HTTPError`` for the
    ``3xx``, which :class:`UrllibTransport` converts back into an :class:`HttpResponse`. The
    point is that the ``Authorization`` header is never replayed to the redirect target.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class UrllibTransport:
    """The production transport: Python's standard library, zero third-party dependencies.

    Zero-dep is a deliberate choice, not an omission (ADR-0016 keeps heavy deps out of core;
    this connector keeps them out of itself too). Graph is a plain REST/JSON API — an HTTP
    client library would buy nothing a Source connector needs and would add a supply-chain
    surface to a package whose entire job is to hold credentials.
    """

    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HttpResponse:
        require_tls(url)
        request = urllib.request.Request(  # noqa: S310 - scheme is enforced by require_tls
            url, data=body, method=method.upper()
        )
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=bytes(response.read()),
                )
        except urllib.error.HTTPError as exc:
            # A status the server really returned (including the 302 the no-redirect handler
            # surfaces): the client decides what it means, not the transport.
            return HttpResponse(
                status=int(exc.code),
                headers=dict(exc.headers.items()),
                body=bytes(exc.read()),
            )
        except urllib.error.URLError as exc:
            # A genuine transport failure. The reason is included; headers (and therefore the
            # bearer token) never are.
            msg = f"SharePoint transport failure for {method.upper()} {url!r}: {exc.reason}"
            raise ProviderError(msg) from exc
