# R2 — Injected query-signal-gen / bad-signal-key faults escape the dispatch guard, leaking raw text across the tool boundary

- **Severity:** major
- **Dimension:** security / no-leak-into-responses invariant (ADR-0052)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #174 (mcp-server, ADR-0052)
- **Status:** fixed

## What is wrong

`RetrievalServer.call` promises exactly one leak-free `ResponseEnvelope` per tool call: a success or
a **typed, text-stripped** failure whose `message` carries no raw store/model/driver text (ADR-0052,
mirroring `BackendSearchError`). The dispatch guard at `mcp/server.py:300-310` enforces this by
mapping faults to envelopes — but it only catches `RetrievalError` subclasses
(`OrchestrationError` / `BackendCapabilityError` / `BackendDimensionError` / `RetrievalError`).

`_build_bundle` (`mcp/server.py:391-402`) runs *inside* that guarded region (via `_search` /
`_more_like_this` → `_run_tool`) and does two things that raise **non-`RetrievalError`** exceptions
on agent-controllable query text:

1. `self._query_signal_gen(text)` — the operator-injected dense/sparse query encoder, run on the
   raw agent query. A plain `Callable`, not wrapped by a fault-mapping adapter. Real
   tokenizers/encoders raise arbitrary runtime exceptions (`RuntimeError`, `OSError`, CUDA/OOM,
   tokenizer faults) whose messages embed model paths, payloads, and driver stacks.
2. `QueryBundle(text=text, filter=filter_, **signals)` — construction with `extra="forbid"`. A
   generator returning an unexpected key (`{"unknown_signal": 1}`) or a wrong-typed value makes
   pydantic raise `ValidationError`.

Neither `RuntimeError`/`OSError` nor `pydantic.ValidationError` is a `RetrievalError` subclass, so
both **escape the try/except uncaught**, cross the tool-call boundary, and surface as a raw
exception on the response path — leaking internal text and breaking the no-leak-into-responses
invariant (and, since `call` never returns, breaking the one-envelope contract).

## Failure scenario

The operator wires a `query_signal_gen` (a dense/sparse query encoder), as intended. On some
agent-controllable query text it faults, as real encoders do:

- A generator raising `RuntimeError("encoder OOM at /opt/models/enc.bin ...")` makes
  `server.call("search", {"query": "anything"}, token="tok")` raise that `RuntimeError`
  **uncaught** at the server boundary — the internal path/message leaks into the response path
  rather than a leak-free `BACKEND_ERROR` envelope.
- A generator returning `{"unknown_signal": 1}` makes `QueryBundle` (extra=forbid) raise a
  `ValidationError`, also **uncaught**.

Both are reachable via agent query/text input on `search` and `more_like_this`.

## Fix

Contain the untrusted seam inside `_build_bundle` (the point the server crosses into caller-supplied
code), following the module's own adapter discipline (`rerank.py` / `pack.py`): catch the fault and
raise a typed, leak-free `QuerySignalError(RetrievalError)` whose message carries only
`type(exc).__name__` — never `str(exc)`. It then flows through the existing
`except RetrievalError → BACKEND_ERROR` mapping, so every tool call again returns exactly one
leak-free envelope.

- New typed error `QuerySignalError(RetrievalError)` in `errors.py` (a query-side encoder-fault
  analogue of `RerankModelError`).
- `_build_bundle` wraps (a) the generator invocation and (b) bundle construction, mapping any fault
  to `QuerySignalError`.

## Acceptance criteria

1. A `query_signal_gen` raising `RuntimeError("... /opt/models/enc.bin ...")` makes `search` return
   `ok=False`, `error.code == BACKEND_ERROR`, and `error.message` contains **neither** the raw
   message **nor** the path substring `/opt/models/enc.bin` (leak-free); `call` never raises.
2. A `query_signal_gen` returning `{"unknown_signal": 1}` makes `search` return a leak-free
   `BACKEND_ERROR` envelope (not an uncaught `ValidationError`).
3. The same holds for `more_like_this` (its `text` input is equally agent-controllable).
4. No regression: the feature test subset, ruff, and mypy --strict stay green; the change is
   additive (new error type + a leak-free mapping) and touches no existing contract.
