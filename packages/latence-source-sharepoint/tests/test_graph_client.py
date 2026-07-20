"""GraphClient behaviour: pagination, throttling, token refresh, and the transport's TLS guard.

These drive the REAL client against the shipped faithful fake, so the retry/refresh/pagination
logic under test is the logic that ships — not a parallel implementation living in a test.
"""

from __future__ import annotations

import urllib.error
from typing import TYPE_CHECKING

import pytest
from latence_core.errors import ConfigError, ProviderError
from latence_source_sharepoint.graph import MAX_PAGE_SIZE, GraphClient
from latence_source_sharepoint.testing import (
    DOWNLOAD_HOST,
    GRAPH_HOST,
    FakeFile,
    FakeGraphApi,
    FakeGraphTransport,
)
from latence_source_sharepoint.transport import (
    HttpResponse,
    UrllibTransport,
    redact_url,
    require_tls,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from latence_source_sharepoint.auth import TokenProvider


class _ScriptedTokenProvider:
    """Hands out tokens from a list, advancing only when a refresh is forced."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.index = 0
        self.refreshes = 0

    def token(self, *, force_refresh: bool = False) -> str:
        # ``refreshes`` counts EVERY forced exchange, not only the ones that had a further
        # token left to hand out. Gating the counter on the script's length would cap it at
        # ``len(tokens) - 1`` by construction, and an assertion on a counter that cannot
        # exceed its expected value proves nothing about the client.
        if force_refresh:
            self.refreshes += 1
            if self.index + 1 < len(self._tokens):
                self.index += 1
        return self._tokens[self.index]


def _client(
    api: FakeGraphApi,
    *,
    token_provider: TokenProvider | None = None,
    sleeps: list[float] | None = None,
    page_size: int = 200,
    max_retries: int = 5,
    max_retry_after_seconds: float = 300.0,
) -> tuple[GraphClient, FakeGraphTransport]:
    transport = FakeGraphTransport(api)
    provider = token_provider or _ScriptedTokenProvider(["valid-token"])
    return (
        GraphClient(
            transport=transport,
            token_provider=provider,
            drive_id=api.drive_id,
            base_url=GRAPH_HOST,
            page_size=page_size,
            max_retries=max_retries,
            max_retry_after_seconds=max_retry_after_seconds,
            sleep=(sleeps.append if sleeps is not None else lambda _seconds: None),
        ),
        transport,
    )


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_follows_odata_next_link_across_every_page() -> None:
    files = [FakeFile(path=f"/doc-{n:02d}.md", content=f"body {n}".encode()) for n in range(25)]
    client, transport = _client(FakeGraphApi(files=files), page_size=4)
    listed = list(client.list_files())
    assert [item.name for item in listed] == sorted(f.path.lstrip("/") for f in files)
    # 25 items at 4 per page = 7 pages; a client that ignored nextLink would make one request.
    assert sum(1 for call in transport.calls if "children" in call.url) == 7


class _UnorderedGraphApi(FakeGraphApi):
    """A Graph double that hands every folder's children back in **descending id** order.

    Entirely legal: Graph documents no ordering for a ``driveItem`` children collection unless
    the request carries ``$orderby``, and the connector sends none. Any offline fake that hands
    back its children pre-sorted proves a property the real dependency does not deliver, so the
    ordering the connector promises has to be demonstrated against a fake that does not help.

    The key is stated absolutely (descending id) rather than relatively ("reverse whatever the
    base fake did"): a relative rule silently cancels out whenever the base fake happens to
    emit the opposite order, which is exactly how the first version of this double handed the
    client a perfectly ascending fixture and let the sort under test be deleted with the suite
    still green.
    """

    def _children_of(self, folder: str) -> list[dict[str, object]]:
        return sorted(
            super()._children_of(folder), key=lambda child: str(child["id"]), reverse=True
        )

    def children_ids(self, folder: str) -> list[str]:
        """The ids this double serves for ``folder``, so a test can prove it does not help."""
        return [str(child["id"]) for child in self._children_of(folder)]


def test_files_are_sorted_by_item_id_even_when_graph_returns_them_unordered() -> None:
    """The listing order is the **item id** order, not the name order Graph happened to send.

    The ids here are opaque and disagree with the names on purpose (real Graph ids are), so
    this fails both on a client that does not sort at all and on one that sorts by name — the
    rename-unstable key the connector deliberately rejects.
    """
    files = [
        FakeFile(path="/alpha.md", content=b"alpha", item_id="01QQ"),
        FakeFile(path="/bravo.md", content=b"bravo", item_id="01AA"),
        FakeFile(path="/charlie.md", content=b"charlie", item_id="01ZZ"),
        FakeFile(path="/delta.md", content=b"delta", item_id="01MM"),
    ]
    api = _UnorderedGraphApi(files=files)
    # Guard the guard: the double must hand the client an order that is NOT the answer.
    assert api.children_ids("") == ["01ZZ", "01QQ", "01MM", "01AA"]

    listed = list(_client(api)[0].list_files())

    # Ascending by opaque id — which for this fixture is NOT ascending by name.
    assert [item.item_id for item in listed] == ["01AA", "01MM", "01QQ", "01ZZ"]
    assert [item.name for item in listed] == ["bravo.md", "delta.md", "alpha.md", "charlie.md"]
    assert [item.name for item in listed] != sorted(item.name for item in listed)


def test_an_unordered_drive_lists_identically_to_an_ordered_one() -> None:
    """Two Graph servers disagreeing only on child order must produce one identical listing."""
    files = [
        FakeFile(path="/Contracts/nda.md", content=b"nda"),
        FakeFile(path="/Contracts/msa.md", content=b"msa"),
        FakeFile(path="/Reports/q1.md", content=b"q1"),
        FakeFile(path="/root.md", content=b"root"),
    ]
    ordered, _ = _client(FakeGraphApi(files=files))
    unordered, _ = _client(_UnorderedGraphApi(files=files))
    assert [item.drive_path for item in ordered.list_files()] == [
        item.drive_path for item in unordered.list_files()
    ]


def test_the_fake_rejects_skip_paging_the_way_graph_does() -> None:
    """Guards the fake's faithfulness: Graph does not support $skip on driveItem children."""
    api = FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")])
    transport = FakeGraphTransport(api)
    response = transport.request(
        "GET",
        f"{GRAPH_HOST}/drives/{api.drive_id}/root/children?$top=10&$skip=10",
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status == 400
    assert b"invalidRequest" in response.body


def test_page_size_beyond_the_graph_ceiling_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=str(MAX_PAGE_SIZE)):
        _client(FakeGraphApi(), page_size=MAX_PAGE_SIZE + 1)


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def test_a_429_is_retried_after_exactly_the_retry_after_interval() -> None:
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        throttle_after=0,
        throttle_burst=1,
        retry_after_seconds=11,
    )
    sleeps: list[float] = []
    client, _ = _client(api, sleeps=sleeps)
    assert [item.name for item in client.list_files()] == ["a.md"]
    assert sleeps == [11.0]


def test_persistent_throttling_raises_a_typed_error_instead_of_looping_forever() -> None:
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        throttle_after=0,
        throttle_burst=99,
        retry_after_seconds=1,
    )
    sleeps: list[float] = []
    client, _ = _client(api, sleeps=sleeps, max_retries=3)
    with pytest.raises(ProviderError, match="throttling"):
        list(client.list_files())
    assert sleeps == [1.0, 1.0, 1.0]


def test_an_absurd_retry_after_is_refused_rather_than_parking_the_pipeline() -> None:
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        throttle_after=0,
        throttle_burst=1,
        retry_after_seconds=3600,
    )
    sleeps: list[float] = []
    client, _ = _client(api, sleeps=sleeps, max_retry_after_seconds=60.0)
    with pytest.raises(ProviderError, match="back-off"):
        list(client.list_files())
    assert sleeps == []


# ---------------------------------------------------------------------------
# Auth failure and refresh
# ---------------------------------------------------------------------------


def test_an_expired_token_triggers_exactly_one_refresh_and_the_run_continues() -> None:
    tokens = _ScriptedTokenProvider(["expired-token", "valid-token"])
    client, _ = _client(
        FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")]), token_provider=tokens
    )
    assert [item.name for item in client.list_files()] == ["a.md"]
    assert tokens.refreshes == 1


def test_throttling_after_an_expired_token_does_not_re_force_a_refresh_per_retry() -> None:
    """The 401 forces ONE token exchange; the throttled retries reuse the fresh token.

    Re-forcing on every retry defeats the token cache and hammers the Entra ID token
    endpoint — which is itself rate-limited — precisely while the tenant is already
    throttling this application, escalating a transient throttle into a token-endpoint
    block.
    """
    tokens = _ScriptedTokenProvider(["expired-token", "valid-token"])
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        # The expired-token 401 is answered before the throttle counter is consulted, so
        # the burst lands on the retries that follow the refresh.
        throttle_after=0,
        throttle_burst=3,
        retry_after_seconds=1,
    )
    sleeps: list[float] = []
    client, _ = _client(api, token_provider=tokens, sleeps=sleeps)
    assert [item.name for item in client.list_files()] == ["a.md"]
    assert sleeps == [1.0, 1.0, 1.0]
    assert tokens.refreshes == 1


def test_a_permanently_bad_token_fails_with_the_graph_error_code() -> None:
    tokens = _ScriptedTokenProvider(["nope"])
    client, _ = _client(
        FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")]), token_provider=tokens
    )
    with pytest.raises(ProviderError, match="InvalidAuthenticationToken"):
        list(client.list_files())


def test_an_unknown_drive_is_a_config_error_not_a_provider_error() -> None:
    api = FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")])
    transport = FakeGraphTransport(api)
    client = GraphClient(
        transport=transport,
        token_provider=_ScriptedTokenProvider(["valid-token"]),
        drive_id="no-such-drive",
        base_url=GRAPH_HOST,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(ConfigError, match="does not exist"):
        list(client.list_files())


def test_an_unknown_root_path_is_a_config_error() -> None:
    client, _ = _client(FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")]))
    with pytest.raises(ConfigError, match="root_path"):
        list(client.list_files("/no-such-folder"))


@pytest.mark.parametrize(
    ("label", "headers"),
    [
        ("no Authorization header at all", {}),
        ("a non-bearer scheme", {"Authorization": "Basic dXNlcjpwYXNz"}),
        ("an unknown bearer token", {"Authorization": "Bearer not-a-real-token"}),
    ],
)
def test_the_fake_refuses_every_unauthenticated_shape(
    label: str, headers: dict[str, str]
) -> None:
    """Guards the fake's faithfulness: Graph 401s all three, so an offline pass means something.

    Without this, a fake that quietly served documents to an unauthenticated request would let a
    connector that forgot to attach credentials sail through the whole offline suite.
    """
    api = FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")])
    response = FakeGraphTransport(api).request(
        "GET", f"{GRAPH_HOST}/drives/{api.drive_id}/root/children", headers=headers
    )
    assert response.status == 401, label
    assert b"InvalidAuthenticationToken" in response.body


def test_a_query_string_token_is_rejected_the_way_graph_rejects_it() -> None:
    """Guards the fake's faithfulness: Graph accepts a bearer token in the header only."""
    api = FakeGraphApi(files=[FakeFile(path="/a.md", content=b"a")])
    transport = FakeGraphTransport(api)
    response = transport.request(
        "GET",
        f"{GRAPH_HOST}/drives/{api.drive_id}/root/children?access_token=valid-token",
        headers={},
    )
    assert response.status == 401
    assert b"InvalidAuthenticationToken" in response.body


# ---------------------------------------------------------------------------
# Download integrity
# ---------------------------------------------------------------------------


def test_a_truncated_download_is_refused_rather_than_ingested() -> None:
    class _TruncatingApi(FakeGraphApi):
        def _file_entry(self, entry: FakeFile) -> dict[str, object]:
            listed = super()._file_entry(entry)
            listed["size"] = len(entry.content) + 10  # Graph says 10 bytes more than arrive
            return listed

    client, _ = _client(_TruncatingApi(files=[FakeFile(path="/a.md", content=b"short")]))
    item = next(iter(client.list_files()))
    with pytest.raises(ProviderError, match="truncated download"):
        client.download(item)


def test_a_throttled_download_host_is_retried_honouring_retry_after() -> None:
    """The byte-transferring leg gets the same backoff as every other request.

    The redirect target is a *different* host from Graph, and it throttles on its own budget.
    A client that re-issues the pre-authenticated GET outside the retry plumbing aborts the
    whole ingest on one transient 503 — even with ``max_retries=5`` and an explicit
    ``Retry-After``.
    """
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        download_throttle_burst=1,
        download_throttle_status=503,
        download_retry_after_seconds=1,
    )
    sleeps: list[float] = []
    client, transport = _client(api, sleeps=sleeps, max_retries=5)
    item = next(iter(client.list_files()))

    assert client.download(item) == b"a"
    assert sleeps == [1.0]
    # Two attempts against the download host, and NEITHER may carry the bearer token.
    redirected = [call for call in transport.calls if call.url.startswith(DOWNLOAD_HOST)]
    assert len(redirected) == 2
    assert not any(call.had_authorization for call in redirected)


def test_a_permanently_failing_download_host_still_raises_a_typed_error() -> None:
    api = FakeGraphApi(
        files=[FakeFile(path="/a.md", content=b"a")],
        download_throttle_burst=99,
        download_throttle_status=429,
        download_retry_after_seconds=2,
    )
    sleeps: list[float] = []
    client, _ = _client(api, sleeps=sleeps, max_retries=3)
    item = next(iter(client.list_files()))
    with pytest.raises(ProviderError, match="429") as raised:
        client.download(item)
    assert sleeps == [2.0, 2.0, 2.0]
    # A pre-authenticated download URL carries its own access token, so it must never be
    # echoed into an error message and from there into a log (ADR-0034).
    assert DOWNLOAD_HOST not in str(raised.value)


# ---------------------------------------------------------------------------
# The transport's TLS guard (the production transport, not the fake)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://graph.microsoft.com/v1.0/drives", "ftp://example.invalid/x", "graph/v1.0"]
)
def test_a_non_tls_url_is_refused_before_any_header_is_written(url: str) -> None:
    with pytest.raises(ConfigError, match="https"):
        require_tls(url)


@pytest.mark.parametrize("url", ["https://graph.microsoft.com/v1.0", "http://localhost:8080/v1.0"])
def test_tls_and_loopback_urls_are_allowed(url: str) -> None:
    require_tls(url)  # does not raise


def test_the_production_transport_refuses_a_cleartext_request() -> None:
    headers: Mapping[str, str] = {"Authorization": "Bearer secret"}
    with pytest.raises(ConfigError, match="https"):
        UrllibTransport().request("GET", "http://graph.microsoft.com/v1.0/me", headers=headers)


# ---------------------------------------------------------------------------
# The pre-authenticated download URL is a credential (ADR-0034 / ADR-0044)
# ---------------------------------------------------------------------------

# Shaped exactly like the URL Graph puts in the 302 'Location' for a content download: the
# 'tempauth' query parameter is a live access token for the storage host.
_PREAUTH_TOKEN = "eyJhbGciOiJub25lIn0.LIVE_TEMPAUTH_TOKEN_VALUE.sig"  # noqa: S105 - fake
_PREAUTH_PATH = "/_layouts/15/download.aspx"
_PREAUTH_QUERY = f"UniqueId=abc-123&tempauth={_PREAUTH_TOKEN}&ApiVersion=2.0"


class _ExplodingOpener:
    """Stands in for the urllib opener when the storage host is unreachable."""

    def open(self, request: object, timeout: float | None = None) -> object:
        raise urllib.error.URLError("[Errno 8] nodename nor servname provided")


def test_a_transport_failure_never_echoes_the_preauthenticated_download_token() -> None:
    url = f"https://contoso.sharepoint.com{_PREAUTH_PATH}?{_PREAUTH_QUERY}"
    transport = UrllibTransport()
    transport._opener = _ExplodingOpener()  # type: ignore[assignment]  # noqa: SLF001 - the socket is the thing being faked

    with pytest.raises(ProviderError) as raised:
        transport.request("GET", url, headers={})

    message = str(raised.value)
    assert _PREAUTH_TOKEN not in message
    assert "tempauth" not in message
    assert "UniqueId=abc-123" not in message
    # Still diagnosable: the operator learns which host and which path failed, and why.
    assert "contoso.sharepoint.com" in message
    assert _PREAUTH_PATH in message
    assert "nodename nor servname" in message


def test_a_non_tls_download_redirect_never_echoes_its_token() -> None:
    url = f"http://contoso.sharepoint.com{_PREAUTH_PATH}?{_PREAUTH_QUERY}"

    with pytest.raises(ConfigError) as raised:
        require_tls(url)

    message = str(raised.value)
    assert _PREAUTH_TOKEN not in message
    assert "tempauth" not in message
    assert "https" in message
    assert "contoso.sharepoint.com" in message


def test_a_server_supplied_next_link_is_never_echoed_into_an_error() -> None:
    """The `@odata.nextLink` is server-controlled, so its query string is not ours to classify.

    `redact_url`'s rule is unconditional for exactly this reason (transport.py). `GraphClient`
    handles the one *other* server-supplied URL in the connector — the pagination link — and must
    apply the same rule, because a `ProviderError` message is copied verbatim into the runner's
    `stage_end` span and from there into the `latence.trace` log pipeline (ADR-0034).
    """
    secret = "OPAQUE_SERVER_STATE_VALUE"  # pragma: allowlist secret
    next_link = f"{GRAPH_HOST}/drives/d1/root/children?$skiptoken={secret}"

    class _PaginatingThenFailingTransport:
        """Page 1 hands back a nextLink; page 2 fails the way a degraded Graph does."""

        def request(
            self,
            method: str,
            url: str,
            *,
            headers: Mapping[str, str],
            body: bytes | None = None,
        ) -> HttpResponse:
            if url == next_link:
                return HttpResponse(status=500, headers={}, body=b'{"error":{"code":"boom"}}')
            page = f'{{"value": [], "@odata.nextLink": "{next_link}"}}'
            return HttpResponse(status=200, headers={}, body=page.encode())

    client = GraphClient(
        transport=_PaginatingThenFailingTransport(),
        token_provider=_ScriptedTokenProvider(["valid-token"]),
        drive_id="d1",
        base_url=GRAPH_HOST,
        page_size=200,
        max_retries=0,
        max_retry_after_seconds=300.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ProviderError) as raised:
        list(client.list_files())

    message = str(raised.value)
    assert secret not in message
    assert "$skiptoken" not in message
    # Still diagnosable: which host and which listing failed survive the redaction.
    assert "graph.microsoft.com" in message
    assert "/drives/d1/root/children" in message


class _RecordingTransport:
    """Records every (url, Authorization) pair, and serves a scripted body per URL prefix."""

    def __init__(self, first_page_body: bytes) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._first_page_body = first_page_body

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
    ) -> HttpResponse:
        self.calls.append((url, headers.get("Authorization")))
        if not url.startswith(GRAPH_HOST):
            # The foreign host happily answers — a real harvesting endpoint would, and would
            # keep the token. If the client ever reaches here, the token has already left.
            return HttpResponse(status=200, headers={}, body=b'{"value": []}')
        return HttpResponse(status=200, headers={}, body=self._first_page_body)


@pytest.mark.parametrize(
    "foreign_next_link",
    [
        "https://attacker.example.com/harvest?p=1",
        # Same path, different host: the resemblance to a Graph URL buys it nothing.
        "https://graph.microsoft.com.attacker.example.com/v1.0/drives/d1/root/children",
        # A different *scheme* on the right host is still a different origin (and would also
        # be caught by require_tls — belt and braces, in either order).
        "http://graph.microsoft.com/v1.0/drives/d1/root/children",
    ],
)
def test_a_foreign_next_link_never_receives_the_bearer_token(foreign_next_link: str) -> None:
    """A server-supplied `@odata.nextLink` off the configured origin is refused, uncalled.

    This is the pagination twin of the download leg's cross-host protection. `_paginate` follows
    the nextLink *credentialed*, so an unpinned link turns any hostile / compromised / misdirected
    Graph-compatible endpoint into a token exfiltration primitive: one listing body containing
    `{"@odata.nextLink": "https://attacker..."}` and the tenant's access token is `GET`-ed to the
    attacker with a full `Authorization: Bearer` header. `require_tls` does not catch it (an
    attacker's URL is happily https), and the shipped fake never emits a foreign-host nextLink.

    The assertion is on the RECORDED WIRE TRAFFIC, not on the raised error: what matters is that
    the request was never made, not that a message was produced afterwards.
    """
    token = "SUPER-SECRET-BEARER"  # pragma: allowlist secret
    page = f'{{"value": [], "@odata.nextLink": "{foreign_next_link}"}}'.encode()
    transport = _RecordingTransport(page)

    client = GraphClient(
        transport=transport,
        token_provider=_ScriptedTokenProvider([token]),
        drive_id="d1",
        base_url=GRAPH_HOST,
        page_size=200,
        max_retries=0,
        max_retry_after_seconds=300.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises((ProviderError, ConfigError)) as raised:
        list(client.list_files())

    # Non-vacuity: the first, legitimate page WAS fetched credentialed, so the client really did
    # reach the pagination decision rather than failing before it.
    assert len(transport.calls) >= 1
    assert transport.calls[0][0].startswith(GRAPH_HOST)
    assert transport.calls[0][1] == f"Bearer {token}"
    # The defect: any call to a host other than the configured Graph origin.
    off_origin = [url for url, _auth in transport.calls if not url.startswith(GRAPH_HOST)]
    assert off_origin == [], (
        f"the bearer token was replayed off-origin to {off_origin!r} "
        f"(headers: {[auth for url, auth in transport.calls if url in off_origin]!r})"
    )
    # And the token never appears on any recorded request bar the legitimate Graph one.
    for url, auth in transport.calls[1:]:
        assert auth is None or url.startswith(GRAPH_HOST)

    message = str(raised.value)
    assert token not in message
    assert "p=1" not in message  # the link's query string is redacted like every other


def test_a_same_origin_next_link_spelled_differently_still_paginates() -> None:
    """The pin is an ORIGIN comparison, not a string prefix — it must not break real paging.

    Graph is free to spell its own nextLink with an explicit default port or a differently-cased
    host. Rejecting those would turn a security guard into an outage, so this pins the normalising
    behaviour that keeps a legitimate multi-page listing working.
    """
    next_link = "https://GRAPH.microsoft.com:443/v1.0/drives/d1/root/children?__page=1"
    second_page = (
        '{"value": [{"id": "i2", "name": "b.md", "size": 1, "file": {"mimeType": "text/plain"}, '
        '"webUrl": "https://w/2", "lastModifiedDateTime": "2024-01-01T00:00:00Z"}]}'
    )
    first_page = (
        '{"value": [{"id": "i1", "name": "a.md", "size": 1, "file": {"mimeType": "text/plain"}, '
        '"webUrl": "https://w/1", "lastModifiedDateTime": "2024-01-01T00:00:00Z"}], '
        f'"@odata.nextLink": "{next_link}"}}'
    )

    class _TwoPageTransport:
        def request(
            self,
            method: str,
            url: str,
            *,
            headers: Mapping[str, str],
            body: bytes | None = None,
        ) -> HttpResponse:
            payload = second_page if url == next_link else first_page
            return HttpResponse(status=200, headers={}, body=payload.encode())

    client = GraphClient(
        transport=_TwoPageTransport(),
        token_provider=_ScriptedTokenProvider(["valid-token"]),
        drive_id="d1",
        base_url=GRAPH_HOST,
        page_size=200,
        max_retries=0,
        max_retry_after_seconds=300.0,
        sleep=lambda _seconds: None,
    )

    assert [item.item_id for item in client.list_files()] == ["i1", "i2"]


@pytest.mark.parametrize(
    "extra_facets",
    [
        pytest.param("", id="folder-facet-only"),
        # `_is_folder` is tested before `_is_file`, so an entry carrying BOTH facets is walked
        # as a folder and reaches the identical loop.
        pytest.param(
            ', "file": {"mimeType": "text/plain"}, "size": 1', id="both-file-and-folder-facet"
        ),
    ],
)
def test_a_folder_listing_that_names_itself_is_refused_rather_than_walked_forever(
    extra_facets: str,
) -> None:
    """A children collection is a SERVER-SUPPLIED body, so it can name an already-walked folder.

    `_paginate` already refuses a repeating `@odata.nextLink` ("A server that echoed its own page
    as the nextLink would loop forever"); the folder recursion is the same threat one level up. A
    hostile, compromised or misdirected Graph-compatible endpoint (or a `graph_base_url` pointed
    at a non-Graph proxy) that answers a folder's children with that same folder makes `list_files`
    push the identical children URL back onto `pending` on every iteration: the client requests
    forever and `pending` grows without bound, so the run both hangs and leaks memory.

    The assertion is on the RECORDED WIRE TRAFFIC as well as the error: what matters is that the
    walk stopped on its own, not that some outer limit eventually stopped it. The transport's cap
    is the *failure detector* here — on the pre-fix client it is what trips.
    """
    cap = 40

    class _SelfReferencingFolderTransport:
        """Every children page hands back the very folder that was asked for."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def request(
            self,
            method: str,
            url: str,
            *,
            headers: Mapping[str, str],
            body: bytes | None = None,
        ) -> HttpResponse:
            self.calls.append(url)
            if len(self.calls) > cap:
                msg = (
                    f"list_files did not terminate: {len(self.calls)} requests issued against a "
                    "drive containing exactly one folder. The folder recursion has no cycle guard."
                )
                raise AssertionError(msg)
            page = (
                '{"value": [{"id": "FOLDER-1", "name": "loop", '
                '"folder": {"childCount": 1}' + extra_facets + "}]}"
            )
            return HttpResponse(status=200, headers={}, body=page.encode())

    transport = _SelfReferencingFolderTransport()
    client = GraphClient(
        transport=transport,
        token_provider=_ScriptedTokenProvider(["valid-token"]),
        drive_id="d1",
        base_url=GRAPH_HOST,
        page_size=200,
        max_retries=0,
        max_retry_after_seconds=300.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ProviderError) as raised:
        list(client.list_files())

    message = str(raised.value)
    assert "FOLDER-1" in message
    assert "cycle" in message
    # Non-vacuity: the root listing AND the folder's own listing were really fetched, so the
    # client reached the recursion decision rather than failing before it...
    assert len(transport.calls) >= 2
    # ...and then stopped there, far below the transport's cap.
    assert len(transport.calls) <= 3, transport.calls


def test_a_nested_drive_still_lists_every_file_with_the_cycle_guard_in_place() -> None:
    """The guard is inert on legitimate input — distinct folder ids walk exactly as before."""
    files = [
        FakeFile(path="/Contracts/nda.md", content=b"nda"),
        FakeFile(path="/Contracts/Archive/old.md", content=b"old"),
        FakeFile(path="/Reports/q1.md", content=b"q1"),
        FakeFile(path="/root.md", content=b"root"),
    ]
    client, _ = _client(FakeGraphApi(files=files))
    assert [item.drive_path for item in client.list_files()] == [
        "/root.md",
        "/Contracts/nda.md",
        "/Contracts/Archive/old.md",
        "/Reports/q1.md",
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://host/path?a=b", "https://host/path?<redacted>"),
        ("https://host/path", "https://host/path"),
        ("https://user:pw@host/path", "https://host/path"),  # pragma: allowlist secret
        ("https://host:8443/p#frag", "https://host:8443/p#<redacted>"),
        ("http://[::1]:8080/p?t=1", "http://[::1]:8080/p?<redacted>"),
        ("graph/v1.0", "graph/v1.0"),
    ],
)
def test_redact_url_keeps_the_diagnosable_parts_and_nothing_else(url: str, expected: str) -> None:
    assert redact_url(url) == expected
