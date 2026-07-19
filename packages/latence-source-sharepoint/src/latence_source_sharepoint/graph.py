"""``GraphClient`` — everything about talking to Microsoft Graph, behind four methods.

This is the deep module of the connector: auth attachment, pagination, throttling, retry,
token refresh, redirect-safe download, and Graph's error taxonomy all live here, and the
Provider above it sees only :meth:`GraphClient.list_files` and :meth:`GraphClient.download`.

What it handles, and what the shipped faithful fake therefore also enforces:

* **Auth** — the bearer token is attached as an ``Authorization`` header on every Graph
  request. Graph does **not** accept a token in the query string; the fake rejects one with
  the same ``401`` so a "just put it in the URL" regression cannot pass offline.
* **Token expiry** — a ``401 InvalidAuthenticationToken`` triggers exactly ONE forced token
  refresh and one retry. A second ``401`` is a real authorisation failure and is raised.
* **Pagination** — a drive listing is followed via ``@odata.nextLink`` until it is absent.
  Graph does **not** support ``$skip`` on ``driveItem`` children; the client never sends it
  and the fake rejects it with the real ``400 invalidRequest``.
* **Throttling** — ``429`` (and ``503``) carry ``Retry-After`` in seconds. The client waits
  exactly that long and retries, up to ``max_retries``, then raises. It never busy-loops and
  never ignores ``Retry-After``, which is what gets a tenant blocked. This holds on **every**
  leg, including the redirected download: the storage host throttles on its own budget, and
  that is the leg that moves the bytes, so leaving it out would let one transient blip abort a
  multi-hour library ingest.
* **Download** — ``/content`` answers ``302`` to a pre-authenticated URL on a different host.
  The client re-issues that GET with **no** ``Authorization`` header (see
  :mod:`~latence_source_sharepoint.transport`) but with the same retry/back-off plumbing, and
  verifies the byte count against the size Graph reported, so a truncated body is a typed
  error rather than a silently short document.

Determinism: children of a folder are sorted by item id before being walked, so two runs over
an unchanged drive produce identical documents in an identical order.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError, ProviderError

from latence_source_sharepoint.transport import redact_url, require_tls

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from latence_source_sharepoint.auth import TokenProvider
    from latence_source_sharepoint.transport import HttpResponse, HttpTransport

DEFAULT_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

# Graph's documented ceiling for ``$top`` on a children collection.
MAX_PAGE_SIZE = 999

# Statuses Graph uses for "back off and try again": throttled, and the transient service
# faults its own SDKs retry.
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

# A cap on how long a single ``Retry-After`` may park the pipeline. Graph occasionally returns
# minutes-long values under heavy tenant throttling; without a cap one document could stall a
# run for an hour with no signal. Beyond the cap the client gives up with a typed error naming
# the requested wait, so an operator can schedule the run differently.
DEFAULT_MAX_RETRY_AFTER_SECONDS = 300.0

# Content-hash algorithms Graph reports on a ``driveItem``'s ``file.hashes``, most trustworthy
# first. SharePoint Online reports ``quickXorHash``; OneDrive Personal reports ``sha256Hash``
# (and legacy ``sha1Hash``). Whichever is present is the *content address* the connector uses
# to decide whether a document changed — which is what lets a Delta run skip an unchanged
# document without downloading a byte of it.
_HASH_FIELDS: tuple[tuple[str, str], ...] = (
    ("sha256Hash", "sha256"),
    ("quickXorHash", "quickxor"),
    ("sha1Hash", "sha1"),
)


@dataclass(frozen=True)
class GraphItem:
    """One SharePoint file, as Graph describes it — before any byte is downloaded.

    ``content_hash`` is ``"<algorithm>:<value>"`` when Graph reported one, else ``None`` (a
    drive that returns no hashes; the connector then falls back to hashing the bytes it
    fetches). ``path`` is the drive-relative folder path with a leading ``/``, e.g.
    ``/Contracts``, so ``f"{path}/{name}"`` is the human location inside the drive.
    """

    item_id: str
    name: str
    path: str
    size: int
    content_hash: str | None
    web_url: str
    last_modified: str

    @property
    def drive_path(self) -> str:
        """The drive-relative path of the file itself, e.g. ``/Contracts/nda.md``."""
        return f"{self.path.rstrip('/')}/{self.name}"


class GraphClient:
    """List and fetch ``driveItem``s from one SharePoint/OneDrive drive.

    Accepts its transport, token provider, and sleep function rather than constructing them
    (the offline suite drives the real retry/refresh logic with a fake clock, so the tested
    behaviour is the shipped behaviour, not a parallel implementation).
    """

    def __init__(
        self,
        *,
        transport: HttpTransport,
        token_provider: TokenProvider,
        drive_id: str,
        base_url: str = DEFAULT_GRAPH_BASE_URL,
        page_size: int = 200,
        max_retries: int = 5,
        max_retry_after_seconds: float = DEFAULT_MAX_RETRY_AFTER_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not drive_id:
            msg = (
                "SharePointSource requires a non-empty 'drive_id' — the Graph id of the "
                "document library to read (GET /sites/{site-id}/drives lists them)."
            )
            raise ConfigError(msg)
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            msg = (
                f"SharePointSource 'page_size' must be between 1 and {MAX_PAGE_SIZE} "
                f"(Graph's documented $top ceiling for a children collection); got {page_size}."
            )
            raise ConfigError(msg)
        if max_retries < 0:
            msg = f"SharePointSource 'max_retries' must be >= 0; got {max_retries}."
            raise ConfigError(msg)
        self._transport = transport
        self._tokens = token_provider
        self._drive_id = drive_id
        self._base_url = base_url.rstrip("/")
        self._page_size = page_size
        self._max_retries = max_retries
        self._max_retry_after = max_retry_after_seconds
        self._sleep = sleep

    # -- public API ---------------------------------------------------------

    def list_files(self, root_path: str = "") -> Iterator[GraphItem]:
        """Yield every file under ``root_path``, recursing into folders, deterministically.

        ``root_path`` is drive-relative (``""``/``"/"`` = the drive root, ``"/Contracts"`` a
        subfolder). Folders are walked depth-first in sorted item-id order and each folder's
        children are fully paginated, so the yielded sequence is stable across runs.
        """
        normalized = "/" + root_path.strip("/") if root_path.strip("/") else ""
        pending: list[tuple[str, str]] = [(self._children_url(normalized), normalized)]
        while pending:
            url, folder_path = pending.pop()
            folders: list[tuple[str, str]] = []
            for entry in self._paginate(url):
                item_path = folder_path or ""
                if _is_folder(entry):
                    child_id = str(entry.get("id", ""))
                    child_name = str(entry.get("name", ""))
                    if child_id and child_name:
                        folders.append(
                            (
                                f"{self._base_url}/drives/{self._drive_id}/items/"
                                f"{urllib.parse.quote(child_id)}/children"
                                f"?$top={self._page_size}",
                                f"{item_path}/{child_name}",
                            )
                        )
                elif _is_file(entry):
                    yield _to_item(entry, item_path)
            # Sorted, then reversed onto a LIFO stack, so the walk visits folders in
            # ascending name order — a stable, reproducible traversal.
            for folder in sorted(folders, key=lambda pair: pair[1], reverse=True):
                pending.append(folder)

    def download(self, item: GraphItem) -> bytes:
        """Fetch a file's bytes, following Graph's ``302`` to the pre-authenticated URL safely.

        The redirected request carries **no** credentials (the download host is a different
        origin; replaying the bearer token there would leak it, and the real endpoint rejects
        it), but it goes through the same retry/``Retry-After`` plumbing as every other
        request. The returned byte count is checked against the size Graph reported.
        """
        url = (
            f"{self._base_url}/drives/{self._drive_id}/items/"
            f"{urllib.parse.quote(item.item_id)}/content"
        )
        response = self._send(url, method="GET")
        if response.status in {301, 302, 303, 307, 308}:
            location = response.header("Location")
            if not location:
                msg = (
                    f"SharePointSource: Graph redirected the download of {item.drive_path!r} "
                    "with no 'Location' header."
                )
                raise ProviderError(msg)
            require_tls(location)
            # Deliberately uncredentialed: a pre-authenticated download URL needs no token,
            # and the storage host rejects an unexpected Authorization header outright. It is
            # still sent through _send, because the download host throttles and blips like any
            # other service — and this is the leg that moves the bytes, so dropping it out of
            # the retry plumbing turns one transient 503 into a failed library ingest.
            response = self._send(location, method="GET", credentialed=False)
        if response.status != 200:
            raise ProviderError(
                self._failure_message(response, what=f"download {item.drive_path!r}")
            )
        data = response.body
        if len(data) != item.size:
            msg = (
                f"SharePointSource: truncated download of {item.drive_path!r} — Graph reported "
                f"{item.size} bytes, {len(data)} arrived. Refusing to ingest a partial document."
            )
            raise ProviderError(msg)
        return data

    # -- request plumbing ---------------------------------------------------

    def _children_url(self, folder_path: str) -> str:
        drive = f"{self._base_url}/drives/{self._drive_id}"
        if not folder_path:
            return f"{drive}/root/children?$top={self._page_size}"
        quoted = urllib.parse.quote(folder_path.lstrip("/"))
        return f"{drive}/root:/{quoted}:/children?$top={self._page_size}"

    def _paginate(self, url: str) -> Iterator[dict[str, Any]]:
        """Follow ``@odata.nextLink`` until exhausted, yielding each page's ``value`` entries.

        The ``nextLink`` is a **server-supplied** URL, so like the download redirect it only ever
        reaches an error message through :func:`~latence_source_sharepoint.transport.redact_url`
        — its query string is not ours to classify, and every message raised from here is copied
        verbatim into the runner's span (ADR-0034).
        """
        next_url: str | None = url
        seen: set[str] = set()
        while next_url:
            if next_url in seen:
                # A server that echoed its own page as the nextLink would loop forever.
                msg = (
                    "SharePointSource: Graph returned a repeating '@odata.nextLink' "
                    f"({redact_url(next_url)!r}); refusing to paginate in a cycle."
                )
                raise ProviderError(msg)
            seen.add(next_url)
            payload = self._get_json(next_url)
            value = payload.get("value")
            if not isinstance(value, list):
                msg = (
                    "SharePointSource: Graph returned a children page without a 'value' array "
                    f"for {redact_url(next_url)!r}."
                )
                raise ProviderError(msg)
            for entry in value:
                if isinstance(entry, dict):
                    yield entry
            raw_next = payload.get("@odata.nextLink")
            next_url = raw_next if isinstance(raw_next, str) and raw_next else None

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self._send(url, method="GET")
        if response.status == 404:
            msg = (
                f"SharePointSource: Graph returned 404 for {redact_url(url)!r}. The drive id "
                f"{self._drive_id!r} or the configured 'root_path' does not exist, or the "
                "application has no permission to read it."
            )
            raise ConfigError(msg)
        if response.status != 200:
            raise ProviderError(
                self._failure_message(response, what=f"list {redact_url(url)!r}")
            )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            msg = f"SharePointSource: Graph returned a non-JSON body for {redact_url(url)!r}."
            raise ProviderError(msg) from exc
        if not isinstance(payload, dict):
            msg = (
                "SharePointSource: Graph returned a non-object JSON body for "
                f"{redact_url(url)!r}."
            )
            raise ProviderError(msg)
        return payload

    def _send(self, url: str, *, method: str, credentialed: bool = True) -> HttpResponse:
        """One request, with throttling backoff and a single 401-driven token refresh.

        ``credentialed=False`` is the pre-authenticated download leg: it sends **no** headers
        (replaying the bearer token on the storage host would leak it, and that host rejects
        it outright), and skips the token-refresh and Graph-specific 401/403 handling — there
        is no token to refresh on a pre-authenticated URL, and its refusals are the storage
        host's, not Graph's, so the caller reports them. The retry/``Retry-After``/
        ``max_retries`` behaviour is identical on both legs, deliberately.
        """
        refreshed = False
        attempt = 0
        while True:
            headers = (
                {
                    "Authorization": f"Bearer {self._tokens.token(force_refresh=refreshed)}",
                    "Accept": "application/json",
                }
                if credentialed
                else {}
            )
            response = self._transport.request(method, url, headers=headers)
            if response.status in _RETRYABLE:
                if attempt >= self._max_retries:
                    raise ProviderError(
                        self._retry_exhausted_message(
                            response, method=method, url=url, credentialed=credentialed
                        )
                    )
                self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue
            if not credentialed:
                # The storage host's own refusals (401/403/404) are not Graph's, and the
                # caller renders them; only the retryable statuses are handled here.
                return response
            if response.status == 401 and not refreshed:
                # The token expired mid-run. Force exactly one refresh and retry; a second
                # 401 is a genuine authorisation failure and falls through below.
                refreshed = True
                continue
            if response.status == 401:
                msg = (
                    "SharePointSource: Graph rejected the credentials for "
                    f"{method} {redact_url(url)!r} "
                    f"(HTTP 401, {_graph_error_code(response.body)!r}) even after refreshing "
                    "the token. Check the app registration's Sites.Read.All / Files.Read.All "
                    "application permission and its admin consent."
                )
                raise ProviderError(msg)
            if response.status == 403:
                msg = (
                    f"SharePointSource: Graph denied access for {method} {redact_url(url)!r} "
                    f"(HTTP 403, {_graph_error_code(response.body)!r}). The token is valid but the "
                    "application is not permitted to read this drive."
                )
                raise ProviderError(msg)
            return response

    def _retry_exhausted_message(
        self, response: HttpResponse, *, method: str, url: str, credentialed: bool
    ) -> str:
        if credentialed:
            who, target = "Graph", f"{method} {redact_url(url)!r}"
            advice = (
                "The tenant is throttling this application; lower 'page_size' or run less often."
            )
        else:
            # NEVER interpolate the redirect URL: a pre-authenticated download URL carries its
            # own access token in the query string, so echoing it into an error message (and
            # from there into a log) leaks a live credential. ADR-0034: no secrets in surfaced
            # output. The Graph-side item is already named by the caller's message.
            who, target = "The pre-authenticated download host", "the redirected content GET"
            advice = (
                "The tenant is throttling this application, or the storage host is degraded; "
                "raise 'max_retries' or run again later."
            )
        return (
            f"SharePointSource: {who} kept returning HTTP {response.status} "
            f"({_graph_error_code(response.body)!r}) for {target} after "
            f"{self._max_retries} retries. {advice}"
        )

    def _retry_delay(self, response: HttpResponse, attempt: int) -> float:
        """Honour ``Retry-After`` exactly; fall back to bounded exponential backoff.

        Graph's ``Retry-After`` is authoritative — ignoring it is what escalates a throttle
        into a tenant-wide block — so it is used verbatim, subject only to the configured
        ceiling.
        """
        raw = response.header("Retry-After")
        if raw is not None:
            try:
                requested = float(raw.strip())
            except ValueError:
                requested = -1.0
            if requested >= 0:
                if requested > self._max_retry_after:
                    msg = (
                        f"SharePointSource: Graph asked for a {requested:g}s back-off, beyond "
                        f"the configured 'max_retry_after_seconds' ({self._max_retry_after:g}). "
                        "The tenant is heavily throttling this application; reschedule the run "
                        "or raise the ceiling deliberately."
                    )
                    raise ProviderError(msg)
                return requested
        return float(min(2**attempt, self._max_retry_after))

    def _failure_message(self, response: HttpResponse, *, what: str) -> str:
        return (
            f"SharePointSource: Graph failed to {what} (HTTP {response.status}, "
            f"{_graph_error_code(response.body)!r})."
        )


def _graph_error_code(body: bytes) -> str:
    """The machine-readable ``error.code`` from a Graph error body, or a safe placeholder.

    Only the code is ever surfaced: a Graph error body echoes the request, and an echoed
    request must not end up in a log line.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "unparseable_error_body"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return str(error["code"])
    return "unknown_error"


def _is_folder(entry: dict[str, Any]) -> bool:
    return isinstance(entry.get("folder"), dict)


def _is_file(entry: dict[str, Any]) -> bool:
    return isinstance(entry.get("file"), dict)


def _to_item(entry: dict[str, Any], folder_path: str) -> GraphItem:
    file_facet = entry.get("file")
    hashes = file_facet.get("hashes") if isinstance(file_facet, dict) else None
    content_hash: str | None = None
    if isinstance(hashes, dict):
        for field, algorithm in _HASH_FIELDS:
            value = hashes.get(field)
            if isinstance(value, str) and value:
                content_hash = f"{algorithm}:{value.lower()}"
                break
    raw_size = entry.get("size", 0)
    return GraphItem(
        item_id=str(entry.get("id", "")),
        name=str(entry.get("name", "")),
        path=folder_path,
        size=int(raw_size) if isinstance(raw_size, int) else 0,
        content_hash=content_hash,
        web_url=str(entry.get("webUrl", "")),
        last_modified=str(entry.get("lastModifiedDateTime", "")),
    )
