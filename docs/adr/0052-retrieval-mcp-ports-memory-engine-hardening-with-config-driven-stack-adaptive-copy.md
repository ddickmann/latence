# The Retrieval MCP server ports memory_engine's hardening, wraps a bring-your-own backend, and pulls all agent-facing copy from operator config at runtime

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

The optional MCP server (`latence-retrieval-mcp`) is a **stack-agnostic, stateless orchestrator**: it
wraps the *customer's* retrieval engine (via the `RetrievalBackend` adapter) with the query-time tooling
and exposes it as MCP tools for agentic search. It **ports the security / reliability / observability
core of the maintainer's `memory_engine`** (their own IP) rather than re-implementing it, and it makes
**all agent-facing copy — tool names, descriptions, parameter docs, the operating-policy guidance, and
the escalating-ladder hierarchy — operator config loaded at runtime**, so one binary auto-adapts to any
production stack.

## Context

`memory_engine` is the maintainer's project (relicensed Apache-2.0 for this use). Its retrieval guts
are ColPali MaxSim over a bespoke shard engine — Apple/MLX-bound and stack-specific — but its **hardening
is excellent and hard-won**: bearer-token auth (0600 token file, constant-time compare, rotation),
loopback-only + Unix-domain-socket transport with an origin allowlist, per-minute rate limiting, WAL /
crash recovery, structured rotating JSON logs, a uniform `ResponseEnvelope` / `Result` / `ToolError`
contract, pydantic input validation, and a documented threat model. That posture is exactly what an
enterprise agentic-search tool needs; the retrieval mechanism is what must become stack-agnostic.

MCP tool descriptions are how an agent decides what to call, so their quality *is* the product — and the
right description depends on the customer's stack (hybrid over Qdrant reads differently from a Databricks
index). Hardcoding the copy would freeze the server to one stack.

## Decision

- **Port the hardening as-is** (it is the maintainer's IP): bearer-0600 + constant-time auth,
  loopback + UDS transport, rate limiting, WAL, structured rotating JSON logs, the envelope/error
  contract, per-tool session records, the threat model. **Strip** what is stack- or product-specific:
  the vendored (formerly CC-BY-NC) colsearch engine, the subscription/entitlement gating, and the
  Electron sidecar. The ColPali/MaxSim retrieval mechanism is replaced by the backend seam.
- **Bring-your-own backend via a capability-declaring adapter.** `RetrievalBackend` declares its
  capabilities (`{"dense","sparse","bm25","filter","multivector"}`) and runs first-stage search; the
  orchestrator composes the best pipeline the declared capabilities + emitted signals allow
  (fetch-per-modality → fuse → graph-expand → rerank → pack) and **gracefully drops any leg the store
  lacks**. Opinionated default, fully config-overridable. Reference adapters ship in order: **Qdrant**
  first, then a lakehouse/Databricks and Azure adapter. The protocol is the contract; the server depends
  on no specific engine.
- **All agent-facing copy is operator config, loaded at runtime.** Tool names, descriptions, parameter
  docs, the server operating-policy, and the tool hierarchy (L1 entry → L2 expand → L3 full-context) live
  in config files. The framework ships **strong defaults** modelled on the `memory_engine` escalating
  ladder; the capability declaration **auto-tailors** those defaults (sparse+dense declared → copy
  mentions hybrid); the operator **overrides** to speak their stack + domain. The tool *surface* keeps
  the ladder shape retargeted to stack-agnostic text + graph (`search`, `expand_neighbors` /
  `more_like_this`, `get_context` returning grounded chunks + provenance, `list_sources`,
  `index_status`); ColPali page-image return becomes a later option, not v1.
- **The copy config is operator-trusted, not agent-mutable.** It is loaded from a sandboxed path and
  schema-validated; the agent only ever reads the rendered tools. Otherwise the server's own tool
  descriptions become a prompt-injection surface — the instruction-source-boundary principle applied to
  our own configuration.
- **v1 is local/loopback + bearer only.** Remote agent access (the OAuth 2.1 tunnel from `memory_engine`,
  or mTLS) is deferred; the transport is seamed so a remote/authenticated adapter drops in later without
  a refactor.

## Consequences

- The maintainer keeps the hardening already earned; only the Apple/colsearch-specific parts are
  replaced, and v1 carries no NC or third-party-encumbered code (ADR-0049).
- One MCP binary serves any stack: capability-adaptive orchestration + config-driven copy means no code
  change to retarget it — the "extremely easy to plug into agentic search" requirement.
- Deferring remote transport keeps the v1 threat model small; co-located agent + tool is the v1
  deployment shape.
