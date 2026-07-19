# R2-connectors_ingestion-9 — SharePoint URL-in-error-message leaks (assigned finding ALREADY FIXED; residual closed)

**Severity:** critical as assigned → **the assigned leak was already fixed**; the residual
closed here is **minor/consistency** (server-supplied URL echoed verbatim).
**Where:**
`packages/latence-source-sharepoint/src/latence_source_sharepoint/transport.py`
(`UrllibTransport.request`, `require_tls`) — **already fixed**;
`packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py`
(`_paginate`, `_get_json`, `_send`) — fixed here.
Sink: `latence-core/.../runner.py:1194` → `tracing.py:83` → `LoggingSpanSink`.

## Status of the assigned finding: ALREADY FIXED (duplicate)

The assigned defect — `UrllibTransport.request`'s `URLError` branch interpolating the full
pre-authenticated redirect URL (`...&tempauth=<JWT>`) into a `ProviderError`, and
`require_tls(location)` doing the same in a `ConfigError` — was fixed earlier in this same
audit round by commit `c5bb45c` (ticket `issues/audit/R2-security_secrets-3.md`), which added
`redact_url()` in `transport.py` and applied it at both sites.

Verified empirically on the current tree rather than taken from the diff: driving the real
`UrllibTransport` at an unresolvable pre-auth-shaped URL now raises

    SharePoint transport failure for GET
    'https://tenant-my.sharepoint.invalid-nxdomain-zzz/personal/x/_layouts/15/download.aspx?<redacted>':
    [Errno 8] nodename nor servname provided, or not known

with the JWT absent, and `require_tls` on the `http://` form likewise. Regression coverage
already exists (`test_a_transport_failure_never_echoes_the_preauthenticated_download_token`,
`test_a_non_tls_download_redirect_never_echoes_its_token`,
`test_redact_url_keeps_the_diagnosable_parts_and_nothing_else`). No re-fix was made — doing so
would have been a no-op change.

## Residual actually fixed here

`redact_url`'s stated rule is unconditional: *"the query string of a URL the **server** handed
us is not ours to classify."* `graph.py` handles a second server-supplied URL — the
`@odata.nextLink` returned in a Graph listing page — and echoed it **verbatim** into six error
messages, so the transport's rule held at the transport boundary but not one layer above it:

* `_paginate` — the pagination-cycle message and the missing-`value` message (`{next_url!r}`)
* `_get_json` — the 404 `ConfigError`, the non-200 `_failure_message`, the non-JSON and
  non-object-JSON `ProviderError`s (`{url!r}`)
* `_send` — the 401, 403 and retry-exhausted (credentialed) messages (`{url!r}`)

`_get_json` and `_send` receive both the client's own constructed Graph URLs and a fully
server-controlled `@odata.nextLink`, and cannot distinguish the two at the point where the
message is built — the same argument that made the redaction unconditional in `transport.py`.

Note honestly: a Graph `$skiptoken` is opaque pagination state, not a demonstrated live
credential, so this is a consistency/defence-in-depth fix, not a proven credential
disclosure. It is claimed as nothing more than that.

## Failure scenario

A drive listing paginates. The endpoint (`graph_base_url` is adopter-configurable — a corporate
proxy or a Graph emulator is a supported deployment) returns an `@odata.nextLink` whose query
string carries opaque server state. The next page 500s, or returns a non-JSON body. `_get_json`
raises `ProviderError(f"... for {url!r}")` with the whole server-supplied URL, and
`runner.py:1194` copies that message into the `stage_end` span's `error_message`, writing it to
the `latence.trace` structured-log pipeline — the identical sink and identical mechanism as the
assigned finding, one layer up.

## Acceptance criteria

1. No URL is interpolated into any error message in `graph.py` without passing through
   `redact_url()` first. (AC1 — met: all six sites.)
2. Host and path survive in every such message, so the operator can still tell which listing or
   which drive failed. (AC2 — met.)
3. A regression test drives a fake Graph whose `@odata.nextLink` carries a secret-shaped query
   parameter, forces a failing next page, and asserts the secret is absent from the raised
   error while host and path remain. It fails on the pre-fix code. (AC3 — met:
   `test_a_server_supplied_next_link_is_never_echoed_into_an_error`.)
4. No behaviour change beyond the redaction; existing SharePoint tests stay green. (AC4 — met.)
5. `make test` stays at or above 2985 passed / 28 skipped. (AC5 — met: 3142 passed, 28 skipped, 1 warning in 308s.)

## Invariants

ADR-0034 (no secrets in any telemetry/lineage facet; failures stay typed) — upheld and
extended. No other invariant is touched: no dependency, config, packaging or output-format
change, so the off-by-default byte-identity and thin-core rules are unaffected.
