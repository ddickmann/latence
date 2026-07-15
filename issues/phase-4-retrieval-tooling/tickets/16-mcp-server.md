## Parent
Spec: retrieval-quality tooling (#158)

## What to build
An optional, hardened, stateless MCP server that wraps a `RetrievalBackend` + the orchestrator and exposes retrieval as MCP tools, with all agent-facing copy loaded from operator config and the ported memory_engine security/observability.

## Acceptance criteria
- [ ] The server exposes the escalating-ladder tool surface (search, expand/more-like-this, get-context with provenance, list-sources, index-status) over a supplied backend.
- [ ] All tool names/descriptions/parameter docs/operating-policy/hierarchy load from operator config at runtime, with strong defaults auto-tailored by backend capabilities; copy config is operator-trusted, sandboxed, schema-validated, and not agent-mutable.
- [ ] Ported hardening: bearer-0600 + constant-time auth, loopback + UDS, rate limiting, structured rotating JSON logs, a uniform response envelope; local/loopback v1 (ADR-0052).
- [ ] Holds no corpus/state; tested at the tool-call boundary with the faithful fake backend (envelope contract, auth required, rate-limit engages, copy renders, agent input cannot mutate copy).

## Blocked by
- #159 — query-time seam
- #172 — orchestrator retrieve()
