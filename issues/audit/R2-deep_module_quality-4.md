# R2 · deep_module_quality · 4 — non-ASCII presented bearer token crashes the MCP auth path with an uncaught `TypeError` (uniform leak-free-envelope contract broken on the auth boundary)

**Severity:** major
**Status:** wontfix — already resolved on branch `feat/retrieval-tooling` (fix commit `2d7f728`, boundary-test commit `1ee7d21`). This is a **duplicate** of `issues/audit/R2-security-2.md`; the R2 auditor re-surfaced the same finding under a second dimension. Recorded here for traceability. **No code change made** — forcing a redundant edit would be pointless churn and a regression risk against a green, tested surface.
**Package:** `latence-retrieval`
**File:** `packages/latence-retrieval/src/latence_retrieval/mcp/security.py` — `BearerAuth.authenticate` (line 94)
**Reached via:** `packages/latence-retrieval/src/latence_retrieval/mcp/server.py` — `RetrievalServer._dispatch` (line 277, auth check runs *before* the rate limiter and *outside* the `OrchestrationError`/`RetrievalError` try/except) → `RetrievalServer.call` (line 238) → `packages/latence-retrieval/src/latence_retrieval/mcp/protocol.py` — `dispatch_text` (lines 68-69, docstring: "the MCP handler never raises across the protocol").

## The failure scenario (as reported)

A caller invokes `server.call('search', {'query':'hi'}, token='wröng-token')` — any presented bearer
token containing a non-ASCII character. On the pre-fix code, `authenticate()` passed the non-empty
`str` straight to `hmac.compare_digest(presented, self._token)`, which raises
`TypeError: comparing strings with non-ASCII characters is not supported`. Because the auth check in
`_dispatch` is *not* wrapped by the dispatch try/except, the `TypeError` propagated uncaught out of
`call()` (and out of `dispatch_text`, whose docstring claims it never raises) instead of returning
the documented `ErrorCode.UNAUTHORIZED` envelope — a crash/DoS on the *unauthenticated, un-throttled*
boundary and a potential traceback leak. It also fully bricks the server if the operator's own
configured token is non-ASCII (every legitimate call raises). ADR-0052 makes this latent: a future
remote (wire-supplied token) transport reuses this core unchanged, so an attacker directly controls
the presented bytes.

Two contracts are broken on the security/auth boundary:
- *"every call returns a typed, leak-free `ResponseEnvelope`"* (`RetrievalServer.call` / ADR-0052);
- *"the MCP handler never raises across the protocol"* (`dispatch_text`).

## What was required (acceptance criteria) — and how it is already met

1. **`authenticate` fails closed on a non-ASCII presented token** — returns a clean `False`, never
   raises. ✅ Met: `security.py:106` now compares **UTF-8-encoded bytes**
   (`hmac.compare_digest(presented.encode("utf-8"), self._token.encode("utf-8"))`), preserving the
   constant-time guarantee (`compare_digest` is built for bytes) while failing closed. Docstring at
   lines 95-102 documents the byte-comparison contract.
2. **A legitimately non-ASCII configured token authenticates its own value** — does not brick the
   server. ✅ Met (same byte-encoding fix; symmetric on both operands).
3. **The auth boundary returns a leak-free `UNAUTHORIZED` envelope, never an uncaught error**,
   through `call` and `dispatch_text`. ✅ Met: the guard now yields `ErrorCode.UNAUTHORIZED` for a
   non-ASCII wrong token instead of raising.
4. **Non-vacuous regression tests that fail on the pre-fix raw-`str` compare** — asserting observable
   behaviour, not a flag (anti-false-green). ✅ Met, at two layers:
   - unit (through the interface): `packages/latence-retrieval/tests/test_mcp_security.py`
     `test_authenticate_rejects_non_ascii_presented_token_without_raising` (line 39) and
     `test_authenticate_matches_a_non_ascii_configured_token` (line 47);
   - tool-call boundary (the whole point — the guard sits outside the try/except):
     `packages/latence-retrieval/tests/test_mcp_server.py`
     `test_non_ascii_wrong_token_is_a_clean_unauthorized_not_an_uncaught_error` (line 129, asserts
     the `UNAUTHORIZED` envelope) and `test_non_ascii_configured_token_still_authenticates_its_own_value`
     (line 138, asserts a successful search). All four raise `TypeError` on the pre-fix code and pass
     on the fix.

## Verification (this pass)

- Fix live: `security.py:106` encodes both operands to UTF-8 bytes.
- `PYTHONHASHSEED=0 LATENCE_CUDA=0 uv run pytest -q packages/latence-retrieval/tests/test_mcp_security.py packages/latence-retrieval/tests/test_mcp_server.py -k "ascii or token or auth"` → 11 passed.
- Deep-module standard honoured: the mechanism stays hidden behind the tiny `authenticate(presented) -> bool` interface and is tested *through* it (both the unit face and the `RetrievalServer.call` boundary), with no interface change and no leak into the envelope.

## Disposition

No further action. The reported defect is genuinely fixed, tested at both the unit and boundary
layers, and already ticketed as `R2-security-2.md`. This ticket exists only so the R2
`deep_module_quality` dimension has its own traceable record of the duplicate.
