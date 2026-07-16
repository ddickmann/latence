# R2 — BearerAuth constant-time compare passes raw `str` to `hmac.compare_digest`: TypeError on any non-ASCII token instead of failing closed

- **Severity:** major
- **Dimension:** spec/AC/ADR compliance / security (auth boundary, fail-open)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #174 (MCP server core, AC "auth required", "typed, leak-free failure"), ADR-0052, SPEC §31
- **Status:** fixed

## What is wrong

`BearerAuth.authenticate` (`packages/latence-retrieval/src/latence_retrieval/mcp/security.py:98`)
compares the presented token against the configured token with
`hmac.compare_digest(presented, self._token)` — passing two raw `str` objects.
`hmac.compare_digest` accepts `str` **only when both are ASCII**; on any non-ASCII character it
raises `TypeError: comparing strings with non-ASCII characters is not supported`.

The auth guard in `RetrievalServer._dispatch`
(`packages/latence-retrieval/src/latence_retrieval/mcp/server.py:276`) runs the `authenticate`
call **before** the tool-dispatch `try/except`, so the `TypeError` is not converted into a
`ResponseEnvelope`. It escapes `call()` uncaught — the exact opposite of the ticket #174 / ADR-0052
contract that every call returns a typed, leak-free envelope (success or a clean `UNAUTHORIZED`).

This is the auth boundary. Its whole premise (constant-time compare) is that the presented token is
attacker-controlled, so an attacker who presents a non-ASCII token gets an internal error / stack
trace instead of a leak-free `UNAUTHORIZED` — the boundary fails open into an internal fault.
Symmetrically, `BearerAuth.from_file` only `.strip()`s the file and imposes no ASCII restriction, so
an operator whose token file contains any non-ASCII character makes
`compare_digest(correct, correct)` raise on **every** legitimate call — the server is fully
bricked. ADR-0052 states this same auth core is reused unchanged by the deferred remote
(wire-supplied `Authorization` header) transport, where non-ASCII presented tokens are entirely
routine.

## Failure scenario

Proven live. Build `RetrievalServer` with `BearerAuth("secret-token")` and call
`server.call("search", {"query": "hi"}, token="töken")`:

- an ASCII wrong token (`"wrong"`) correctly returns `ok=False`, `code="unauthorized"`;
- the non-ASCII presented token `"töken"` raises an uncaught
  `TypeError: comparing strings with non-ASCII characters is not supported`.

Reproduced directly against `BearerAuth("secret-token").authenticate("töken")`. The existing
MCP-security tests only exercise ASCII tokens, so this rejection path is untested — a partial
coverage gap of the "auth required" AC.

## What to build

Compare the tokens as UTF-8-encoded **bytes**, which `hmac.compare_digest` supports for any string
and which is constant-time over the byte sequences:

```python
return hmac.compare_digest(presented.encode("utf-8"), self._token.encode("utf-8"))
```

The fix stays inside the one `authenticate` method behind the `BearerAuth` interface (deep-module
standard). The constant-time guarantee is preserved (bytes comparison is what `compare_digest` is
built for); the `None`/empty/non-`str` rejection above it is unchanged. No contract changes —
additive hardening only; statelessness and the envelope contract are untouched.

## Acceptance criteria

- [x] `BearerAuth.authenticate` compares UTF-8 bytes via `hmac.compare_digest`, so a non-ASCII
      presented token returns `False` (clean reject) instead of raising `TypeError`.
- [x] A legitimate non-ASCII configured token authenticates its own value (server not bricked).
- [x] Regression test at the `RetrievalServer.call` boundary: a wrong non-ASCII presented token
      yields a leak-free `UNAUTHORIZED` envelope (never an uncaught exception) — fails on the
      pre-fix raw-`str` compare (raises `TypeError`), passes on the fix.
- [x] Existing ASCII accept/reject behaviour is unchanged (equal tokens still match, unequal still
      reject).
- [x] Feature tests green + ruff + mypy --strict clean.
