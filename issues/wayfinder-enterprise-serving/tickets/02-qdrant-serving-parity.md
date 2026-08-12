# 02 — Qdrant serving-path parity: the benched stack on a real ANN engine

Type: task
Status: open
Blocked by: —

## Why (Dennis, 2026-08-12)

The bench runs brute-force dense scans; production needs the ANN engine. The Qdrant sink
exists and `GraphRescueRetrieval` deliberately takes engines through the seam — but an
end-to-end "deployed with Qdrant, same recall" validation has never been run. Per the
no-fake-green rule, that's a real gap until measured.

## Scope

1. **Stand up a real Qdrant** (Docker, local Mac — the pod is off-limits while the sweep
   runs). No in-memory fakes for any claimed number.
2. **Index a real S3 export** (wiki2multihop preferred: largest local corpus, 57,812 chunks,
   768-d granite vectors; more if locally available) through the existing sink/adapter path,
   payload columns included (the KG metadata the Expander/Packer read must survive the trip).
3. **Run the graph-rescue retrieval** with Qdrant as the dense engine vs the brute-force
   reference, same queries (the S5 gold query sets, fixed-seed sample is acceptable for
   iteration but the final number is the full set):
   - recall parity: R@10 / docR@5 / docR@10 deltas vs brute-force (target: within ANN noise,
     report exactly; HNSW exactness params tuned and recorded),
   - serving latency: per-query median/p95 end-to-end through the production composition.
4. **Close whatever the trip breaks**: if payloads, filters, or the metadata contract don't
   survive Qdrant round-trip, fix the shipped adapter (that's the point of the validation).
5. **Document the deployment**: a reproducible compose file + guide section (hand to ticket
   04 for final placement).

## Non-negotiables

- Real Qdrant, real export, real queries — every number from the live engine.
- BM25 stays local (it's already fast and exact); this ticket is the dense lane + payload
  contract. Hybrid-in-Qdrant is out of scope unless the parity check forces it.
- Shipped-code changes carry tests (against the fake for CI, validated against live Qdrant
  in this ticket's run — the fake must reject what the real dep rejects).
- Tests, ruff, mypy --strict clean.

## Gate artifact

Parity report: index build stats, HNSW params, recall deltas table, latency table, list of
contract fixes made. Dennis inspects.
