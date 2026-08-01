# 16 — Task: M5 — chain assembly + selection-trace logging

Type: task
Status: resolved
Blocked by: 15

## Question

Implement spec 3.4–3.5: connected-component chains, BFS ordering from V0 with rho tie-break, structured per-chain context emission, peripheral-evidence handling; selection-trace logging (query, V0, E_c features, rho, per-step accept/reject, final A) from day one as the future classifier's weak-supervision dataset.

## Answer

**BUILT + green** (subagent; independently verified — the agent stalled before its full-suite
gate, so the supervising session re-ran every gate: package tests, ruff, `mypy --strict`, and
`make test` = **4730 passed / 0 failed**). Details in [M5-RESULT.md](../M5-RESULT.md). Chains:
sparse component assembly, BFS-from-anchor ordering (ρ/id ties), spec-6.3 directed preference
behind `relations_lookup`, peripheral evidence last, byte-deterministic emission, τ text
redaction-safe by construction. Traces: optional `trace_sink` on the selector (default path
byte-identical), ids-and-counts only — no raw text, a deliberate privacy choice. The e2e
renders the planted 2-hop chain as one chain with the bridge after the direct evidence.
