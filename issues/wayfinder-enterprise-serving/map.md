# Wayfinder map — Enterprise Serving (the final push to production readiness)

Label: `wayfinder:map`

## Destination (Dennis, 2026-08-12 — binding)

When the enterprise-readiness question is raised, the answer is **"yes, measured"** — with
no caveat list. Enterprises can build, test, AND SERVE end-to-end data + retrieval pipelines
in their own environment/cloud, with local models, almost no LLM involvement, and a retrieval
stack that is elite for enterprise requirements. The three named caveats from the 2026-08-12
assessment are each closed by a ticket, verified with real dependencies, and brought into the
enterprise production lane. Then one final `/improve-codebase-architecture` pass, with
**byte-identical outputs** as the hard gate.

## Operating contract (carried from wayfinder-e2e-sota-run, still binding)

1. **No fake green**: every claim verified against real dependencies (real Qdrant, real
   exports, real scale). Fakes are offline-CI-only and must reject what the real dep rejects.
2. **Sacrifices only with data**; shipped default = strong configuration.
3. **No quality degradation**: latency work must produce identical (or measurably equal)
   rankings — "orders of magnitude faster" is only claimed with a before/after measurement
   AND a parity proof on real data.
4. Subagents execute the heavy work; long runs get an on-host supervisor.
5. **The running S5 rescue sweep on the pod is untouchable** — ticket work runs on the Mac
   (local exports, local Docker) until the sweep completes. The sweep re-runs AFTER ticket 01
   lands to measure the latency gain at full scale.
6. License discipline: any new dependency gets weights-vs-code license verification before
   adoption; UNVERIFIED is a blocking state.

## Tickets

| # | Ticket | Closes caveat | Gate |
|---|---|---|---|
| 01 | Hypergraph + graph lane latency at SOTA — orders of magnitude, zero quality loss | "seconds per query at 677k hyperedges" | parity proof (identical rankings) + measured speedup on real exports; re-run of the S5 rescue sweep confirms at full scale |
| 02 | Qdrant serving-path parity — the benched stack on a real ANN engine | "the serving path isn't the benchmarked path" | live Qdrant, real export indexed, measured recall parity vs brute-force + serving latency |
| 03 | Incremental ingest — document churn without full rebuild | "batch, not incremental" | incremental == full-rebuild equivalence proven on a real export; upsert/delete/re-extract paths tested |
| 04 | Enterprise readiness pass — docs, deployment, release, CI | the meta-caveat ("here are a few caveats…") | a releasable repo: README, licenses, deployment guides for the stacks, publishing, CI green |

Order: 01 ∥ 02 ∥ 03 (distinct surfaces: retrieval internals / serving adapter / core ingest),
then 04 (documents what 01–03 built), then the architecture pass (byte-identical gate),
then the S5 rescue sweep re-run at scale (the ticket-01 full-scale number).

## Decisions

1. (2026-08-12, Dennis) The push happens NOW, in parallel with the running uda/wiki bench
   cells; the sweep is re-run afterwards to measure ticket-01's improvement at scale.
2. (2026-08-12, Dennis) Architecture pass at the end: everything in terms of outputs must be
   byte-identical afterwards while the architecture and codebase are massively improved.
