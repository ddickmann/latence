# R4-connectors_ingestion-7 — GraphClient forces a token refresh on *every* retry after a 401

**Severity:** major
**Area:** connectors / ingestion — `latence-source-sharepoint`
**Location:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py:322` (`GraphClient._send`)
**Guarding test (hollow):** `packages/latence-source-sharepoint/tests/test_graph_client.py:35-47,200-206`

## What is wrong

`_send` documents "a single 401-driven token refresh" and its own comment says
"Force exactly one refresh and retry". It implements that with a single `refreshed`
flag that is set to `True` on the 401 and **never cleared**. The bearer header is
built at the top of the retry loop as
`self._tokens.token(force_refresh=refreshed)` — so once a 401 has occurred, *every
subsequent iteration of that request's retry loop* passes `force_refresh=True` and
forces a brand-new token exchange.

## Failure scenario

Auth mode `client_credentials`. Graph returns `401 InvalidAuthenticationToken`
(token expired mid-run), then three `429`s carrying `Retry-After`. Five HTTP
attempts produce **four** forced client-credentials POSTs to
`login.microsoftonline.com` — one per retry — because the token cache is bypassed
on each pass. The Entra ID token endpoint is itself aggressively rate-limited, so
the connector hammers it precisely while the tenant is already throttling the app,
escalating a transient throttle into a token-endpoint block.

## Why the existing test cannot see it

`_ScriptedTokenProvider.token` increments `refreshes` only while
`self.index + 1 < len(self._tokens)`. With the two scripted tokens the counter is
capped at 1 **by construction**, so `assert tokens.refreshes == 1` holds no matter
how many refreshes the client actually forces. The test is structurally incapable
of failing.

## Acceptance criteria

1. Within one `_send` call, `TokenProvider.token(force_refresh=True)` is invoked at
   most once, regardless of how many retryable statuses follow the 401.
2. After the forced refresh, later retries of the same request reuse the cached
   (now fresh) token — `force_refresh=False`.
3. Existing behaviour is unchanged: a second 401 after the refresh still raises the
   typed `ProviderError` about credentials; the `credentialed=False` download leg
   still never touches the token provider.
4. A regression test counts **every** `force_refresh=True` call (not a capped
   counter) across a 401-then-throttle sequence and fails on the old code.
5. `make test` stays green at or above 2985 passed, 28 skipped.
