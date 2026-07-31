# 07 — Grilling: licence placement of the new modules

Type: grilling
Status: resolved
Blocked by: 02

## Question

The repo is PolyForm-NC with a deliberately pinned 7-module Apache-2.0 retained set for query-time retrieval primitives (LICENSE + test_publish_readiness.py). The CoE selector/propagation are exactly that kind of primitive; the sidecar builder is pipeline. Which new modules (if any) join the Apache set vs default NC? Dennis-owned business decision. Outcome: LICENSE scope update + publish-readiness pin update plan.

## Answer

**Dennis (2026-07-31): ALL new hypergraph/CoE modules stay PolyForm-Noncommercial-1.0.0.**

Nothing joins the Apache-2.0 retained set — it stays exactly the seven generic query-time
primitives; the LICENSE scope note and the `test_publish_readiness.py` pin are untouched
(they would have failed the build on an eighth grant, which is exactly why this decision had
to be explicit rather than drift). The CoE stack — `hypergraph.py`, `coe.py`, the chains
module to come, and everything hyperedge-shaped in core — is the frontier differentiator
behind the commercial-by-agreement model (ADR-0055): free for research, commercial by
agreement.

The middle option (Apache-ing only the KV *format reader* as interop grease) was presented
and declined — the format spreads via the spec'd layout in the docs if needed; the reader
implementations stay NC. Revisitable after the M6 eval proves the SOTA claim, as a fresh
decision, not a drift.

Zero code changes required: every new module already carries the NC SPDX header (the M4
handoff instructed the same), so this resolution is a confirmation that the default was
correct, now on the record.
