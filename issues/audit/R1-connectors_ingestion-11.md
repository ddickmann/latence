# R1-connectors_ingestion-11 — SharePoint download redirect leg bypasses retry/backoff entirely

**Severity:** major
**Area:** connectors / ingestion (`latence-source-sharepoint`)
**Stream:** `feat/release-fleet-integration` (tickets #178–#199)
**Sites:**
- `packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py:200` (uncovered redirect request)
- `packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py:201-204` (immediate `ProviderError`)
- fake gap: `packages/latence-source-sharepoint/src/latence_source_sharepoint/testing.py:186-213` vs `:310-323`

## What is wrong

`GraphClient.download` follows Graph's `302` to the pre-authenticated download host by
re-issuing the GET through `self._transport.request("GET", location, headers={})` instead of
through `self._send(...)`. `_send` is the only place that holds the retry/backoff plumbing
(`_RETRYABLE`, `Retry-After` honouring, `max_retries`, `max_retry_after_seconds`), so on the
redirect leg:

- a `429` or `503` from the download host is **never** retried,
- `Retry-After` is **ignored**,
- `max_retries` has **no effect** on the leg that actually moves the bytes.

The shipped faithful fake cannot detect this, because `_maybe_throttle` is reachable only from
`_handle_graph` and never from `_handle_download`. The connector's "throttling / rate-limit
behaviour is mirrored honestly" acceptance criterion is therefore **unproven for the byte
leg** — the offline suite exercises retry only on metadata/listing requests.

## Failure scenario

A transport where Graph answers `/content` with the normal `302`, and the pre-authenticated
download host returns ONE `503 serviceNotAvailable` with `Retry-After: 1` before succeeding.

With `GraphClient(max_retries=5)`:

```
ProviderError: SharePointSource: Graph failed to download '/a.md' (HTTP 503, 'serviceNotAvailable').
```

raised on the first response; the recorded sleeps list is empty and only two requests are made.

In production a single transient throttle or service blip on any one document's content fetch
aborts an entire multi-hour library ingest — even though the operator configured five retries
and the server explicitly asked for a one-second wait.

## What to build

1. Route the redirect leg through the same retry plumbing as every other request. Extend
   `_send` with an uncredentialed mode (`credentialed=False`) that sends **no** headers — the
   token-leak guard must stay intact — and skips the 401-driven token refresh (there is no
   token to refresh on a pre-authenticated URL) and the Graph-specific 401/403 messages, while
   keeping `_RETRYABLE` + `Retry-After` + `max_retries` + `max_retry_after_seconds`.
2. Close the fake's blind spot: make the download host scriptable so it can answer a burst of
   retryable responses with `Retry-After` before succeeding. Off by default, so existing fake
   behaviour is byte-identical.

## Acceptance criteria

- A `503`/`429` from the pre-authenticated download host is retried, waiting exactly the
  `Retry-After` the host asked for, up to `max_retries`.
- Persistent failure on the download leg still raises a typed `ProviderError` (ADR-0034) rather
  than looping forever.
- The redirect request still carries **no** `Authorization` header on every attempt, including
  retries (the fake's `403 AuthenticationFailed` guard must keep passing).
- The truncated-download byte-count check is unchanged.
- The fake's download throttling is opt-in; with it unset, every existing test and output is
  bit-for-bit unchanged.
- A regression test drives the REAL client against the shipped fake, fails on the old code, and
  passes on the fix.
- `make test` stays green at or above 2985 passed, 28 skipped.
