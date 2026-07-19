# R3-console_readonly-10 — A malformed `Origin` or absolute-form request target crashes the console with a raw traceback and zero bytes

**Severity:** major
**Where:** `packages/latence-console/src/latence_console/server.py` — `is_local_origin_header` (host guard) and `ConsoleApp.respond` (router)
**Invariant:** ADR-0034 — no silent errors; failures surface as TYPED errors, never a raw native traceback.

## What to build

Both the host guard and the router call `urllib.parse.urlsplit` on attacker-controlled text
without guarding it. `urlsplit` raises `ValueError("Invalid IPv6 URL")` on an unbalanced `[`.
Neither call site catches it, so the exception escapes into `BaseHTTPRequestHandler`, which
closes the connection **without writing a status line** and prints a 15-frame traceback with
absolute filesystem paths to stderr — the exact opposite of the deliberate `log_message`
silencing that keeps the console quiet.

The fix is to make URL parsing total at both seams:

* a malformed `Origin` is **not** a recognisable local origin → the guard returns `False` →
  the existing 403 "wrong host" page is served;
* a malformed request target has no page behind it → `respond` returns the existing typed
  404 `Response`, exactly as every other unparseable path already does.

No new status codes, no new rendering — the malformed input joins the refusals the console
already expresses.

## Failure scenario (pre-fix, observed)

```
$ uv run python -c "from latence_console.server import is_local_origin_header, ConsoleApp; ..."
ORIGIN RAISED: ValueError Invalid IPv6 URL
RESPOND RAISED: ValueError Invalid IPv6 URL
```

Over a live socket against the real `build_server`:

1. `GET / HTTP/1.1` + `Host: 127.0.0.1:<port>` + `Origin: http://[::1` → the client reads
   `b''` (0 bytes, connection closed, no status line); the traceback ending
   `ValueError: Invalid IPv6 URL` at `server.py:270` is dumped to stderr.
2. The absolute-form target `GET http://[::1 HTTP/1.1` passes through `http.server`
   unchanged and raises the same `ValueError` in `ConsoleApp.respond` → again zero bytes.
3. `ConsoleApp.respond("GET", "//[")` — the documented pure request→response core — raises
   instead of returning a 404 `Response`.

Every *other* malformed input on the same paths (`Host: [::1`, `Origin: null`, `id=%ff`,
`/run/../../etc/passwd`) is already handled and returns 403/404, so this is an unguarded
gap, not a designed refusal. No existing console test sends a malformed `Origin` or an
absolute-form target, which is why the green gate cannot see it.

## Acceptance criteria

- [x] `is_local_origin_header("http://[::1", port)` returns `False` (no exception).
- [x] `ConsoleApp.respond("GET", "//[")` returns a 404 `Response` (no exception).
- [x] `ConsoleApp.respond("GET", "http://[::1")` returns a `Response` (no exception).
- [x] Over a live `build_server` socket, a request carrying `Origin: http://[::1` receives a
      complete HTTP response with status 403 — not zero bytes — and no traceback reaches stderr.
- [x] A regression test that FAILS on the pre-fix code and PASSES after.
- [x] `make test` stays green at or above 2985 passed, 28 skipped.
