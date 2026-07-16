# R3 — Candidate-processor faults escape the MCP dispatch guard, leaking raw store metadata across the tool boundary

- **Severity:** major
- **Dimension:** security / no-leak-into-responses invariant (ADR-0052)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #174 (mcp-server, ADR-0052); #169 (expand/multihop); #172 (orchestrator)
- **Status:** fixed

## What is wrong

`RetrievalServer.call` promises exactly one leak-free `ResponseEnvelope` per tool call — a success
or a **typed, text-stripped** failure whose `message` carries no raw store/model/driver text
(ADR-0052) — and `dispatch_text` (the MCP-protocol face) promises the handler **never raises across
the protocol**. The dispatch guard (`mcp/server.py:_dispatch`) enforces this by mapping faults to
envelopes — but it only catches `RetrievalError` subclasses (`OrchestrationError` /
`BackendCapabilityError` / `BackendDimensionError` / `RetrievalError`).

`_search` / `_more_like_this` run the orchestrator (`retrieve`) *inside* that guarded region, and
`retrieve` runs the operator-configured **candidate-processor chain** over the fetched candidates'
metadata (`orchestrator.py:268`, `candidates = processor.process(...)`). The default
(non-experimental) graph-augmenting processors deliberately fail **loudly** on a malformed KG column
— their anti-false-green contract — by raising a plain `TypeError` / `ValueError`:

- `Expander._id_set` (`expand.py:269,277`) — a scalar/JSON-string where a list of node-id strings is
  expected, or a non-string element.
- `MultiHopExpander._node_ids` (`multihop.py:665,673`) — the same, one hop out.

Neither a plain `TypeError` nor `ValueError` is a `RetrievalError`, so both **escape the dispatch
try/except uncaught**, cross the tool-call boundary (out of both `server.call()` and
`dispatch_text()`), and the escaping exception message embeds the **raw store metadata value and the
document id** — the identical no-leak / never-raise invariant the R2 commits (307e4ab / 1ee7d21)
closed for the `query_signal_gen` path, still open for the processor path.

## Failure scenario

The operator wires the default `expand` processor into the MCP server pipeline. Their store holds a
document whose `kg_node_ids` metadata is a scalar / JSON-string instead of a list (heterogeneous
ingestion). An agent sends an authenticated `search` (or `more_like_this`) whose query surfaces that
document in the fetch results. `Expander._id_set` (or `MultiHopExpander._node_ids`) raises
`TypeError: expand metadata 'kg_node_ids' must be a list of node-id strings, got 'not-a-list' for
candidate 'd1'.` — not a `RetrievalError`, so `_dispatch` does not catch it: `server.call()` raises
(violating "every call returns one `ResponseEnvelope`") and `dispatch_text()` raises across the MCP
protocol handler (violating "the MCP handler never raises"), with raw store text in the traceback.

Reproduced directly: `FakeBackend([Candidate(id='d1', metadata={'kg_node_ids':'not-a-list'})])` +
`OrchestrationPipeline(processors=(Expander({'hop_weight':0.5}),))` →
`server.call('search', {'query':'alpha'}, token='tok')` raised `TypeError` instead of returning an
envelope; the multi-hop variant with metadata `{'kg_node_ids':['ok',123]}` raised `ValueError`
through `server.call()`.

## Fix

Contain the untrusted seam at the point the server runs caller-supplied code over caller-supplied
data — a new `RetrievalServer._retrieve` helper wrapping `retrieve(...)`, following the module's own
adapter discipline (mirroring the R2 `_build_bundle` containment):

- **Re-raise `RetrievalError` unchanged** — the orchestrator's typed store/planning faults
  (`OrchestrationError` → `UNSUPPORTED`, `Backend*` → `UNSUPPORTED`, `BackendSearchError` →
  `BACKEND_ERROR`) are already leak-free and carry their own dispatch code, so they keep it.
- **Map any other `Exception`** (the processors' plain `TypeError` / `ValueError`) to a new typed,
  leak-free `CandidateProcessorError(RetrievalError)` whose message carries only
  `type(exc).__name__` — never `str(exc)`. It then flows through the existing
  `except RetrievalError → BACKEND_ERROR` mapping, so every tool call again returns exactly one
  leak-free envelope.

The processors' loud `TypeError` / `ValueError` for **direct Level-1 callers** is intentional and
unchanged — the containment lives only at the MCP boundary (ADR-0052).

- New typed error `CandidateProcessorError(RetrievalError)` in `errors.py` (the processor-chain
  analogue of `QuerySignalError`), exported from the package root.
- `_search` / `_more_like_this` route through `_retrieve`.

## Acceptance criteria

1. An `expand` pipeline over a candidate whose `kg_node_ids` is `'not-a-list'` makes `search` return
   `ok=False`, `error.code == BACKEND_ERROR`, and `error.message` contains **neither** the raw
   metadata value `not-a-list` **nor** the document id `d1`; `call` never raises. — MET
2. The same holds for `more_like_this` (equally agent-surfaced candidates). — MET
3. A `multihop_expand` pipeline over `{'kg_node_ids':['ok',123]}` returns a leak-free
   `BACKEND_ERROR` envelope (not an uncaught `ValueError`); the offending `123` and the id do not
   leak. — MET
4. A typed `OrchestrationError` planning refusal (a filter on a store lacking `FILTER`) still
   surfaces as `UNSUPPORTED`, NOT collapsed into `BACKEND_ERROR` — the containment re-raises
   `RetrievalError`. — MET
5. No regression: the feature test subset, ruff, and mypy --strict stay green; the change is
   additive (new error type + a leak-free mapping at the MCP seam) and touches no existing contract
   or the loud Level-1 processor behaviour. — MET

## Regression tests

- `test_expand_processor_malformed_metadata_is_a_leak_free_backend_error_not_uncaught`
- `test_multihop_processor_malformed_metadata_is_a_leak_free_backend_error`
- `test_orchestration_error_keeps_its_unsupported_code_through_the_processor_guard`

(All three fail on the pre-fix `server.py` — the exception escapes `server.call` uncaught — and pass
on the fix.)
