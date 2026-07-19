# R2-security_secrets-4 — `HttpLineageTransport` leaks the collector API key on a redirect

**Severity:** major
**Area:** `packages/latence-core/src/latence_core/observability.py` (`HttpLineageTransport`, #185)
**Status:** FIXED

## What was wrong

`HttpLineageTransport.send` POSTed the OpenLineage event with a bare
`urllib.request.urlopen`. urllib's *default* opener installs `HTTPRedirectHandler`, which follows
a `3xx` automatically and **re-sends the original request headers — including
`Authorization: Bearer <api key>` — to the host named in `Location`**, even a different host.

`__init__` additionally accepted plain `http` for a URL carrying an `api_key`, so the bearer token
was also exposed to anything on the wire.

Two things made the leak silent:

* the transport's own `status >= 300` guard never fires, because it only ever observes the final
  `200` after urllib has already followed the redirect; and
* `RunLineage._emit` wraps every send in a blanket `except Exception` (lineage must never fail a
  run), so even a raised error would surface only as a `WARNING`.

The sibling SharePoint connector in this same stream already builds `_NoRedirectHandler`
(`packages/latence-source-sharepoint/src/latence_source_sharepoint/transport.py:140`) for exactly
this reason, documented as "a token leak" — the lineage transport reproduced the bug its sibling
module documents as unacceptable. No test in `packages/latence-core/tests/test_openlineage.py`
exercised a redirect.

## Failure scenario

An operator follows the documented hosted-collector path (`cli.py:263-288`) and sets
`LATENCE_OPENLINEAGE_URL` + `LATENCE_OPENLINEAGE_API_KEY`. The collector — or anything on the
wire, since plain `http` was accepted and the CLI docstring itself recommends
`http://localhost:5000` — answers the POST with `302 Location: http://evil.example/steal`.
urllib follows it and replays `Authorization: Bearer <key>` to `evil.example`. The operator sees
nothing: the run completes, at most one WARNING is logged.

Observed with a loopback probe driven through the real transport:
`attacker host received Authorization: Bearer SUPER-SECRET-LINEAGE-KEY`.

## What was built

1. `_NoRedirectLineageHandler` — a `urllib.request.HTTPRedirectHandler` whose `redirect_request`
   returns `None`, so a `3xx` raises instead of being followed. The transport opens through
   `urllib.request.build_opener(_NoRedirectLineageHandler)`, mirroring the SharePoint connector.
2. `__init__` refuses an `api_key` over plaintext `http` to a non-loopback host (`ConfigError`).
   Loopback `http` stays usable for a local Marquez/emulator, and keyless plain `http` is
   unchanged, so no existing off-the-shelf local setup breaks.
3. `send` converts `urllib.error.HTTPError` (including the surfaced `3xx`) into the typed
   `ProviderLineageError` rather than letting a raw urllib traceback escape (ADR-0034).
4. The CLI docstring for `LATENCE_OPENLINEAGE_API_KEY` now states the https requirement.

## Acceptance criteria

* [x] A collector answering `302` to a second host receives the POST; the redirect target receives
      **no** `Authorization` header — asserted against a real loopback HTTP server pair driven
      through the real `HttpLineageTransport`.
* [x] That `3xx` surfaces as a typed `ProviderLineageError`, not a raw `urllib.error.HTTPError`.
* [x] `HttpLineageTransport("http://collector.example", api_key=...)` raises `ConfigError`;
      keyless plain http, loopback http + key, and https + key all still construct.
* [x] Regression tests fail on the pre-fix code and pass on the fix.
* [x] `make test` stays green at or above the 2985 passed / 28 skipped baseline.

## Regression tests

`packages/latence-core/tests/test_openlineage.py`

* `test_http_transport_never_replays_the_api_key_to_a_redirect_target`
* `test_http_transport_refuses_an_api_key_over_plaintext_to_a_remote_host`
