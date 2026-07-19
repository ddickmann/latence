# R2-security_secrets-3 — SharePoint transport leaks the pre-authenticated download token into typed errors

**Severity:** major (credential disclosure)
**Where:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/transport.py`
(`require_tls`, `UrllibTransport.request`), reached from
`packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py:202-207`.

## What to build

Every URL that this connector interpolates into an error message must be **redacted** first:
scheme + host + path are kept (that is the diagnostic value), the query string, the fragment and
any userinfo are dropped, because a Graph pre-authenticated download URL carries a live access
token in its query string (`tempauth=<JWT>`).

Add a `redact_url()` helper in `transport.py` — the single place every byte passes through — and
use it at both leak sites (`require_tls`'s `ConfigError`, `UrllibTransport.request`'s
`ProviderError` on `URLError`). Correct the `require_tls` docstring, which currently asserts the
false premise ("a token never appears in a URL … so the message is safe to log") that caused the
defect.

## Failure scenario

A real run against Microsoft Graph: `GET /drives/{id}/items/{id}/content` answers `302` with a
`Location` on a different host, e.g.
`https://contoso.sharepoint.com/_layouts/15/download.aspx?UniqueId=...&tempauth=<live JWT>&ApiVersion=2.0`.

`GraphClient.download` calls `require_tls(location)` and then `self._send(location,
credentialed=False)` → `UrllibTransport.request`. Any DNS failure, connection reset or socket
timeout on the storage host raises `urllib.error.URLError`, and the transport built
`ProviderError(f"SharePoint transport failure for GET {url!r}: {exc.reason}")` — the **full** URL
including `tempauth`. That `ProviderError` propagates out of the Provider into the CLI and the
logs. `require_tls` did the same for an `http://` redirect target.

This is the exact rule the same connector enforces elsewhere: `graph.py:427` comments that
"a pre-authenticated download URL carries its own access token in the query string, so echoing it
into an error message (and from there into a log) leaks a live credential", and
`tests/test_graph_client.py:287` asserts `DOWNLOAD_HOST not in str(raised.value)` for the sibling
HTTP-status path. The offline suite stayed green because `FakeGraphTransport`'s redirect target
(`testing.py:275`) carries no credential in its query string and the fake never raises a
transport-level `URLError`.

Violates **ADR-0034** (no secrets in any surfaced error / telemetry) and **ADR-0044** (redaction
floor).

## Acceptance criteria

1. `redact_url()` exists in `transport.py`, keeps scheme/host/port/path, and replaces a non-empty
   query or fragment with a `<redacted>` marker; userinfo (`user:pass@`) is dropped.
2. `UrllibTransport.request`'s `ProviderError` on a transport failure contains no query-string
   value from the requested URL.
3. `require_tls`'s `ConfigError` contains no query-string value from the refused URL, and still
   names the offending scheme so the config fix is obvious.
4. A regression test drives both paths with a URL carrying a live-looking `tempauth=<JWT>` and
   asserts the token is absent from the message while the host/path remain present. It fails on
   the pre-fix code.
5. No behaviour change for credential-free URLs beyond the redaction; the existing TLS-guard tests
   stay green.
6. `make test` stays at or above 2985 passed / 28 skipped.
