"""The faithful offline fake of Microsoft Graph — shipped, not hidden in a tests folder.

Every offline test in this package, the framework's Provider conformance suite, and any
adopter who wants a dry run without a tenant drive the connector through
:class:`FakeGraphTransport`. Shipping it in ``src`` (like
:class:`latence_retrieval.testing.FakeGraphSource`) is deliberate: the fake and the real
client must stay in step, and a fake that lives next to the code it fakes does.

**A fake that accepts what the real API rejects manufactures false green, so this one
rejects.** Faithfully reproduced refusals, each mirroring documented Graph behaviour:

* no ``Authorization`` header → ``401 InvalidAuthenticationToken`` ("Access token is empty.")
* an unknown/expired bearer token → ``401 InvalidAuthenticationToken``
* a token passed in the **query string** (``?access_token=``) → ``401`` — Graph accepts a
  token only in the header, so the "just put it in the URL" shortcut cannot pass offline
* ``$skip`` on a ``driveItem`` children collection → ``400 invalidRequest`` — Graph does not
  support ``$skip`` there, only ``@odata.nextLink`` paging
* an unknown drive id, folder path, or item id → ``404 itemNotFound``
* a ``GET`` of a pre-authenticated download URL that carries an ``Authorization`` header →
  ``403 AuthenticationFailed``, exactly as the storage endpoint the ``302`` points at does.
  This is the check that fails loudly if the connector ever starts replaying the bearer token
  across the redirect.
* the client-credentials exchange with a wrong secret → ``401 invalid_client``; with a missing
  ``grant_type``/``scope`` → ``400 invalid_request``

Throttling is scriptable (``throttle_after`` / ``retry_after_seconds``) so the real
``Retry-After`` path is exercised rather than asserted about — on the Graph host *and*,
via ``download_throttle_burst``, on the pre-authenticated download host, so the retry
behaviour of the leg that actually moves the bytes is proven too. Every request is recorded on
:attr:`FakeGraphTransport.calls` so a test can assert *behaviour* — e.g. "the excluded document
was never downloaded" — instead of a flag.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_source_sharepoint.transport import HttpResponse, require_tls

if TYPE_CHECKING:
    from collections.abc import Mapping

# The host the fake redirects ``/content`` to, mirroring Graph's pre-authenticated download
# URL living on a *different* origin from graph.microsoft.com.
DOWNLOAD_HOST = "https://fake-sharepoint-download.invalid"

GRAPH_HOST = "https://graph.microsoft.com/v1.0"
LOGIN_HOST = "https://login.microsoftonline.com"


@dataclass(frozen=True)
class FakeFile:
    """One file in the fake drive.

    ``hash_algorithm`` models which ``file.hashes`` facet the backend reports: ``"sha256Hash"``
    (OneDrive Personal, and the default here because it is the one algorithm the connector can
    reproduce from the bytes), ``"quickXorHash"`` (SharePoint Online), or ``None`` — a drive
    state that really occurs, because Graph computes the facet asynchronously after upload.
    """

    path: str
    content: bytes
    hash_algorithm: str | None = "sha256Hash"


@dataclass(frozen=True)
class FakeFolder:
    """A folder that exists in the fake drive even when it holds no accepted file."""

    path: str


@dataclass
class FakeGraphApi:
    """An in-memory SharePoint drive that answers Graph requests the way Graph does.

    ``files`` are keyed by drive-relative path (``/Contracts/nda.md``); intermediate folders
    are synthesised, so a test declares only the leaves.
    """

    drive_id: str = "drive-1"
    files: list[FakeFile] = field(default_factory=list)
    valid_tokens: frozenset[str] = frozenset({"valid-token"})
    page_size_override: int | None = None
    # Client-credentials: the fake app registration. ``client_secret`` is a *fake* value that
    # exists only inside this in-memory double; no real credential is ever committed.
    tenant_id: str = "tenant-1"
    client_id: str = "client-1"
    client_secret: str = "fake-secret-not-a-real-credential"
    issued_token: str = "issued-token"
    token_lifetime_seconds: int = 3600
    # Throttling: once this many Graph requests have been served, the next
    # ``throttle_burst`` requests answer 429 with ``Retry-After``.
    throttle_after: int | None = None
    throttle_burst: int = 1
    retry_after_seconds: int = 7
    # The pre-authenticated download host is a *different* service from Graph and throttles
    # (or blips) independently, so it has its own script. The first
    # ``download_throttle_burst`` requests to it answer ``download_throttle_status`` with
    # ``Retry-After: download_retry_after_seconds``; the default of 0 leaves the download host
    # answering exactly as before.
    # Report a ``sha256Hash`` that does NOT match the bytes for this drive-relative path, to
    # exercise the connector's post-download verification of a listing-derived identity.
    corrupt_listing_hash_for: str | None = None
    download_throttle_burst: int = 0
    download_throttle_status: int = 503
    download_retry_after_seconds: int = 1

    _served: int = field(default=0, init=False)
    _throttled: int = field(default=0, init=False)
    _download_throttled: int = field(default=0, init=False)

    # -- item helpers -------------------------------------------------------

    def item_id(self, path: str) -> str:
        """The stable fake item id for a path (Graph ids are opaque and stable per item)."""
        return "item-" + path.strip("/").replace("/", "-")

    def _folders(self) -> set[str]:
        folders = {""}
        for entry in self.files:
            parts = entry.path.strip("/").split("/")[:-1]
            for index in range(len(parts)):
                folders.add("/" + "/".join(parts[: index + 1]))
        return folders

    def _children_of(self, folder: str) -> list[dict[str, Any]]:
        normalized = "" if folder in {"", "/"} else "/" + folder.strip("/")
        children: list[dict[str, Any]] = []
        for sub in sorted(self._folders()):
            if not sub:
                continue
            parent = sub.rsplit("/", 1)[0]
            if parent == normalized:
                children.append(
                    {
                        "id": self.item_id(sub),
                        "name": sub.rsplit("/", 1)[-1],
                        "folder": {"childCount": 0},
                        "webUrl": f"https://contoso.sharepoint.com{sub}",
                    }
                )
        for entry in sorted(self.files, key=lambda f: f.path, reverse=True):
            parent = entry.path.rsplit("/", 1)[0]
            if parent == normalized:
                children.append(self._file_entry(entry))
        # Deliberately NOT in ascending order: Graph specifies no ordering for a children
        # collection absent ``$orderby``, which the connector does not send. A fake that handed
        # its children back pre-sorted would prove a property the real dependency does not
        # deliver, and every determinism test above it would pass on a client that never sorts.
        return children

    def _file_entry(self, entry: FakeFile) -> dict[str, Any]:
        file_facet: dict[str, Any] = {"mimeType": "text/plain"}
        if entry.hash_algorithm is not None:
            if entry.hash_algorithm == "sha256Hash":
                # Graph's sha256Hash IS the SHA-256 of the file content, uppercase hex — the
                # one facet the connector can reproduce from the bytes, so the fake reports it
                # faithfully rather than as a stand-in.
                digest = hashlib.sha256(entry.content).hexdigest().upper()
                if self.corrupt_listing_hash_for == entry.path:
                    digest = hashlib.sha256(b"not the bytes").hexdigest().upper()
            else:
                # A deterministic stand-in for quickXorHash/sha1Hash, whose real values the
                # connector never reproduces; it only requires that it changes iff the
                # content changes, which this does.
                digest = hashlib.sha256(entry.content).hexdigest()[:32].upper()
            file_facet["hashes"] = {entry.hash_algorithm: digest}
        return {
            "id": self.item_id(entry.path),
            "name": entry.path.rsplit("/", 1)[-1],
            "size": len(entry.content),
            "file": file_facet,
            "webUrl": f"https://contoso.sharepoint.com{entry.path}",
            "lastModifiedDateTime": "2026-07-18T00:00:00Z",
        }

    def _file_by_item_id(self, item_id: str) -> FakeFile | None:
        for entry in self.files:
            if self.item_id(entry.path) == item_id:
                return entry
        return None

    # -- request handling ---------------------------------------------------

    def handle(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResponse:
        """Answer one request. The single entry point :class:`FakeGraphTransport` delegates to."""
        parsed = urllib.parse.urlsplit(url)
        if url.startswith(DOWNLOAD_HOST):
            return self._handle_download(parsed, headers)
        if url.startswith(LOGIN_HOST):
            return self._handle_token(body)
        return self._handle_graph(method, parsed, headers)

    def _handle_token(self, body: bytes | None) -> HttpResponse:
        form = urllib.parse.parse_qs((body or b"").decode("utf-8"))

        def single(key: str) -> str:
            values = form.get(key, [])
            return values[0] if values else ""

        if single("grant_type") != "client_credentials" or not single("scope"):
            return _oauth_error(400, "invalid_request")
        if single("client_id") != self.client_id or single("client_secret") != self.client_secret:
            return _oauth_error(401, "invalid_client")
        payload = {
            "token_type": "Bearer",
            "expires_in": self.token_lifetime_seconds,
            "access_token": self.issued_token,
        }
        return _json_response(200, payload)

    def _handle_download(
        self, parsed: urllib.parse.SplitResult, headers: Mapping[str, str]
    ) -> HttpResponse:
        # The real pre-authenticated download endpoint is a storage host that knows nothing
        # about Entra ID bearer tokens: an unexpected Authorization header is rejected. A
        # connector that replays the token across the redirect leaks it — and fails here.
        if _header(headers, "Authorization") is not None:
            return _json_response(
                403,
                {
                    "error": {
                        "code": "AuthenticationFailed",
                        "message": (
                            "Server failed to authenticate the request. A pre-authenticated "
                            "download URL must be requested without an Authorization header."
                        ),
                    }
                },
            )
        throttled = self._maybe_throttle_download()
        if throttled is not None:
            return throttled
        item_id = urllib.parse.parse_qs(parsed.query).get("itemId", [""])[0]
        entry = self._file_by_item_id(item_id)
        if entry is None:
            return _graph_error(404, "itemNotFound")
        return HttpResponse(
            status=200,
            headers={"Content-Type": "application/octet-stream"},
            body=entry.content,
        )

    def _handle_graph(
        self, method: str, parsed: urllib.parse.SplitResult, headers: Mapping[str, str]
    ) -> HttpResponse:
        query = urllib.parse.parse_qs(parsed.query)
        if "access_token" in query:
            # Graph accepts a bearer token in the Authorization header only.
            return _graph_error(
                401, "InvalidAuthenticationToken", "Access token is not supported in the query."
            )
        authorization = _header(headers, "Authorization")
        if not authorization:
            return _graph_error(401, "InvalidAuthenticationToken", "Access token is empty.")
        if not authorization.startswith("Bearer "):
            return _graph_error(
                401, "InvalidAuthenticationToken", "Access token validation failure."
            )
        token = authorization[len("Bearer ") :]
        if token not in self.valid_tokens and token != self.issued_token:
            return _graph_error(
                401,
                "InvalidAuthenticationToken",
                "Lifetime validation failed, the token is expired.",
            )
        if "$skip" in query:
            # Graph: "$skip is not supported on this API." Only nextLink paging works here.
            return _graph_error(400, "invalidRequest", "$skip is not supported on this API.")

        throttled = self._maybe_throttle()
        if throttled is not None:
            return throttled

        path = parsed.path
        marker = f"/drives/{self.drive_id}"
        index = path.find("/drives/")
        if index == -1 or not path.startswith(marker, index):
            # An unknown drive id (or a URL that is not a drive request at all): Graph answers
            # 404 itemNotFound, which the connector must surface as a configuration error.
            return _graph_error(404, "itemNotFound", "The resource could not be found.")
        remainder = path[index + len(marker) :]

        if remainder.endswith("/content"):
            item_id = remainder[len("/items/") : -len("/content")]
            entry = self._file_by_item_id(urllib.parse.unquote(item_id))
            if entry is None:
                return _graph_error(404, "itemNotFound", "The resource could not be found.")
            quoted = urllib.parse.quote(self.item_id(entry.path))
            location = f"{DOWNLOAD_HOST}/download?itemId={quoted}"
            return HttpResponse(status=302, headers={"Location": location}, body=b"")

        folder = self._folder_for(remainder)
        if folder is None:
            return _graph_error(404, "itemNotFound", "The resource could not be found.")
        return self._children_page(folder, query, parsed)

    def _folder_for(self, remainder: str) -> str | None:
        """Map the drive-relative Graph path onto a folder in the fake drive, or ``None``."""
        if remainder == "/root/children":
            return ""
        if remainder.startswith("/root:/") and remainder.endswith(":/children"):
            raw = remainder[len("/root:/") : -len(":/children")]
            candidate = "/" + urllib.parse.unquote(raw).strip("/")
            return candidate if candidate in self._folders() else None
        if remainder.startswith("/items/") and remainder.endswith("/children"):
            item_id = urllib.parse.unquote(remainder[len("/items/") : -len("/children")])
            for folder in self._folders():
                if folder and self.item_id(folder) == item_id:
                    return folder
            return None
        return None

    def _children_page(
        self,
        folder: str,
        query: dict[str, list[str]],
        parsed: urllib.parse.SplitResult,
    ) -> HttpResponse:
        children = self._children_of(folder)
        requested = self.page_size_override or int(query.get("$top", ["200"])[0])
        offset = int(query.get("__page", ["0"])[0])
        page = children[offset : offset + requested]
        payload: dict[str, Any] = {"value": page}
        if offset + requested < len(children):
            base_query = {k: v for k, v in query.items() if k != "__page"}
            base_query["__page"] = [str(offset + requested)]
            payload["@odata.nextLink"] = urllib.parse.urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    urllib.parse.urlencode(base_query, doseq=True),
                    "",
                )
            )
        return _json_response(200, payload)

    def _maybe_throttle(self) -> HttpResponse | None:
        self._served += 1
        if self.throttle_after is None or self._served <= self.throttle_after:
            return None
        if self._throttled >= self.throttle_burst:
            return None
        self._throttled += 1
        return HttpResponse(
            status=429,
            headers={"Retry-After": str(self.retry_after_seconds)},
            body=json.dumps(
                {"error": {"code": "activityLimitReached", "message": "Too many requests."}}
            ).encode("utf-8"),
        )

    def _maybe_throttle_download(self) -> HttpResponse | None:
        """The download host's own back-off script — the leg that actually moves the bytes.

        Kept separate from :meth:`_maybe_throttle` because the pre-authenticated URL lives on a
        storage host, not on graph.microsoft.com: it throttles on its own budget and answers
        ``503 serviceNotAvailable`` as readily as ``429``.
        """
        if self._download_throttled >= self.download_throttle_burst:
            return None
        self._download_throttled += 1
        code = (
            "activityLimitReached"
            if self.download_throttle_status == 429
            else "serviceNotAvailable"
        )
        return HttpResponse(
            status=self.download_throttle_status,
            headers={"Retry-After": str(self.download_retry_after_seconds)},
            body=json.dumps(
                {"error": {"code": code, "message": "The service is temporarily unavailable."}}
            ).encode("utf-8"),
        )


@dataclass(frozen=True)
class RecordedCall:
    """One request the connector made, for behavioural assertions."""

    method: str
    url: str
    had_authorization: bool

    @property
    def is_download(self) -> bool:
        """True for the byte-transferring leg (the ``/content`` GET or its redirect target)."""
        return "/content" in self.url or self.url.startswith(DOWNLOAD_HOST)


class FakeGraphTransport:
    """An :class:`~latence_source_sharepoint.transport.HttpTransport` over a :class:`FakeGraphApi`.

    Applies the same TLS guard the real transport applies, so a test cannot accidentally
    exercise a code path the production transport would refuse.
    """

    def __init__(self, api: FakeGraphApi) -> None:
        self.api = api
        self.calls: list[RecordedCall] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HttpResponse:
        require_tls(url)
        self.calls.append(
            RecordedCall(
                method=method.upper(),
                url=url,
                had_authorization=_header(headers, "Authorization") is not None,
            )
        )
        return self.api.handle(method.upper(), url, headers, body)

    @property
    def download_urls(self) -> list[str]:
        """The URLs of the byte-transferring requests — the cost an incremental run avoids."""
        return [call.url for call in self.calls if call.is_download]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _json_response(status: int, payload: dict[str, Any]) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _graph_error(status: int, code: str, message: str = "") -> HttpResponse:
    return _json_response(status, {"error": {"code": code, "message": message or code}})


def _oauth_error(status: int, code: str) -> HttpResponse:
    return _json_response(status, {"error": code, "error_description": f"{code} (fake)."})


# ---------------------------------------------------------------------------
# The offline dry-run path
# ---------------------------------------------------------------------------

# A tiny canned document library, so `transport: "latence_source_sharepoint.testing:demo_transport"`
# in a stack file runs the WHOLE pipeline against a SharePoint-shaped source with no tenant, no
# credential beyond a token the operator exports themselves, and no network. This is the offline
# half of the live smoke path documented in the package README.
DEMO_DRIVE_ID = "demo-drive"
DEMO_TOKEN = "demo-token"

_DEMO_FILES = (
    FakeFile(path="/Handbook/onboarding.md", content=b"# Onboarding\n\nWelcome to Contoso.\n"),
    FakeFile(path="/Handbook/security.md", content=b"# Security\n\nReport incidents promptly.\n"),
    FakeFile(path="/Contracts/nda.md", content=b"# NDA\n\nContoso and Fabrikam agree.\n"),
)


def demo_api() -> FakeGraphApi:
    """A fresh canned drive for the offline dry run."""
    return FakeGraphApi(
        drive_id=DEMO_DRIVE_ID, files=list(_DEMO_FILES), valid_tokens=frozenset({DEMO_TOKEN})
    )


def demo_transport() -> FakeGraphTransport:
    """Factory for ``transport: "latence_source_sharepoint.testing:demo_transport"``."""
    return FakeGraphTransport(demo_api())
