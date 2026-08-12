# 01 — Hypergraph + graph lane latency at SOTA: orders of magnitude, zero quality loss

Type: task
Status: open
Blocked by: —

## Why (Dennis, 2026-08-12)

uda's hypergraph lane costs **5.24s median per query at 677k hyperedges** (measured, equal-weight
sweep). Fine for a benchmark that pays full price honestly; not fine for interactive serving.
"Make it SOTA without quality degradation but with enormous latency gain. Orders of magnitude.
Test it. Verify it. Bring it in the enterprise production lane."

## The measured cost structure (start here, profile before optimizing)

- Hypergraph signals lane: gated ρ scoring over the matched hyperedge neighbourhood — the
  dominant term at scale (5.24s median on uda vs ~0.6s whole-pipeline on multihop_rag).
- GraphRetriever traversal: alias seeding + 1-hop expansion over 516k nodes.
- Dense first stage: brute-force scan (ticket 02's territory — do not solve it here, but do
  not let it pollute the profile either; measure lanes separately).
- Redundancy precompute: O(pool²) Jaccard per query — corpus-independent but nonzero.

## Known mitigations (documented, NOT yet measured — this ticket measures them)

1. **Precomputed hyperedge indices**: entity→hyperedge inverted index, hyperedge→chunk maps,
   built once at export/load time instead of scanned per query.
2. **Community-scoped traversal**: restrict expansion to the seed entities' communities.
3. The policy gate already skips the graph on queries that don't need it (landed) — measure
   its contribution but don't count it as this ticket's speedup.
4. Anything else the profile reveals (e.g. alias-map hashing, ρ memoization, numpy-vectorized
   scoring) — profile first, optimize what the data says.

## Non-negotiables

- **Parity proof**: identical rankings (byte-identical result lists) on real exports before/
  after, at minimum on wiki2multihop (137,986 hyperedges) locally. If uda's export is not
  available locally, uda-scale verification rides the post-sweep re-run — say so explicitly,
  never extrapolate.
- **Measured speedup**: per-lane latency before/after on the same hardware, medians + p95,
  reported per dataset. "Orders of magnitude" is claimed only if the numbers show it.
- The optimization lives in the SHIPPED packages (`latence_retrieval`), not the benchmark
  harness; the benchmark consumes it through the existing seams. Default behavior stays
  byte-identical for existing users (index building must be transparent or opt-in-by-default
  with identical results).
- **Do not touch the pod while the S5 sweep runs.** All work verifies on Mac-local exports.
- Tests, ruff, mypy --strict clean; boundary tests in the house style.

## Gate artifact

A report (in this ticket's directory or the campaign stage dir): profile table, per-mitigation
speedup, parity proof method + result, and the shipped config surface. Dennis inspects; the
post-sweep S5 re-run provides the full-scale confirmation number.
