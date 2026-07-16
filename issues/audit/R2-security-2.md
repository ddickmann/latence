# R2 · security · 2 — non-ASCII presented bearer token crashes the MCP auth path with an uncaught `TypeError` (unauthenticated, un-throttled DoS)

**Severity:** major
**Status:** fixed (already resolved on branch `feat/retrieval-tooling` by commit `2d7f728`; this ticket records it for traceability)
**Package:** `latence-retrieval`
**File:** `packages/latence-retrieval/src/latence_retrieval/mcp/security.py` — `BearerAuth.authenticate`
**Reached via:** `packages/latence-retrieval/src/latence_retrieval/mcp/server.py` — `RetrievalServer._dispatch` (calls `authenticate` before the rate limiter and before the `RetrievalError` try/except)

## What was wrong

`BearerAuth.authenticate` compared the presented token against the configured
token with `hmac.compare_digest(presented, self._token)` on the raw `str`
values. `hmac.compare_digest` accepts `str` operands **only when both are pure
ASCII**; on any non-ASCII character it raises
`TypeError: comparing strings with non-ASCII characters is not supported`.

Because `_dispatch` runs `authenticate` **first** — before `RateLimiter.allow()`
and before the `try/except` that maps typed retrieval faults to leak-free
envelopes — a non-ASCII presented token produced an **uncaught** `TypeError`
that propagated out of the tool-call handler. This broke the server's central
contract that every `call` returns "success or typed, leak-free failure": an
un-authenticated caller could crash the handler, the crash was **not**
rate-limited (it fires before the limiter), and it was fully
unauthenticated-reachable — an un-throttled denial-of-service that also defeated
the uniform leak-free-envelope invariant. A legitimately non-ASCII **configured**
token was equally fatal: `compare_digest(correct, correct)` raised on every call,
bricking the server.

## Failure scenario

An agent/client calls a tool presenting `token="tökén-attack"` (non-empty `str`
with any non-ASCII char). `server.call("search", {"query": "hi"}, token="tökén-attack")`
reaches `authenticate`, which calls `hmac.compare_digest(presented, self._token)`
and raises `TypeError: comparing strings with non-ASCII characters is not
supported`, uncaught. An ASCII-but-wrong token correctly returns
`ok=False, code=unauthorized`; only the non-ASCII path crashes. A client can spam
non-ASCII tokens to repeatedly crash the handler without ever being authenticated
or throttled.

## Fix

Encode **both** operands to UTF-8 bytes before the constant-time compare:
`hmac.compare_digest(presented.encode("utf-8"), self._token.encode("utf-8"))`.
`compare_digest` is built for bytes and preserves the constant-time guarantee, so
the timing side-channel protection is unchanged while the comparison now **fails
closed**: a non-ASCII wrong token is a clean `False` → `UNAUTHORIZED` envelope,
and a legitimately non-ASCII configured token authenticates its own value instead
of bricking every call. The `authenticate` docstring states this bytes-comparison
rationale explicitly.

## Acceptance criteria

- A non-ASCII presented token yields `authenticate(...) is False` (and thus a
  leak-free `UNAUTHORIZED` envelope through `_dispatch`), never a raised
  `TypeError`.
- A legitimately non-ASCII **configured** token authenticates its own exact value
  and rejects a near-miss.
- The ASCII accept/reject/empty/`None` behaviour is unchanged; constant-time
  comparison is preserved (bytes operands to `compare_digest`).
- Feature tests + `ruff` + `mypy --strict` stay green; change is a localized,
  additive normalisation, no contract or pipeline regression.

## Regression tests

`packages/latence-retrieval/tests/test_mcp_security.py`:
- `test_authenticate_rejects_non_ascii_presented_token_without_raising` — fails on
  the old raw-`str` compare (raises `TypeError`), passes on the fix (clean
  `False`).
- `test_authenticate_matches_a_non_ascii_configured_token` — fails on the old code
  (`compare_digest(correct, correct)` raises for a non-ASCII token), passes on the
  fix (authenticates its own value, rejects a near-miss).

Both were verified non-vacuous by reverting `authenticate` to the raw-`str`
compare: each fails with `TypeError: comparing strings with non-ASCII characters
is not supported`, and passes once the bytes-encoding fix is restored.
