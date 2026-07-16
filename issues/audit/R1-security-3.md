# R1 — multi-hop DuckDB/Neo4j traversal has no cycle/visited guard: bounded depth, unbounded work

- **Severity:** major
- **Dimension:** security / availability (DoS) / STATELESS zero-ops (ADR-0048/0051)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #169 (multi-hop expand / GraphSource seam), ADR-0051
- **Status:** fixed

## What is wrong

ADR-0051's multi-hop `expand` promises a **bounded** traversal of the emitted knowledge graph
(`max_hops`, AC3): the reference `_traverse` (`multihop.py:82-116`) is a visited-set BFS that
expands each node **once** at its minimum hop — polynomial `O(V+E)` — and `NetworkxGraphSource`
uses `single_source_shortest_path_length(cutoff=max_hops)`, likewise BFS.

The two real query-engine adapters bound **depth but not work**:

- `DuckDBGraphSource.neighbors` (`multihop.py:210-231`) builds a recursive CTE whose recursive
  `walk` step is `UNION ALL` with **no visited set** — its only stop is `WHERE w.hop < ?`.
- `Neo4jGraphSource.neighbors` (`multihop.py:406-411`) composes a bare variable-length pattern
  `MATCH p = (s)-[*1..max_hops]-(n)` and collapses it with `min(length(p))`.

Both **enumerate every path**, not every node. Real knowledge graphs are cyclic and high-degree,
so a node on a cycle is re-expanded at every hop and the number of walks grows as
`~degree**max_hops`. The final `min(hop)` is still *correct*, so nothing signals the blowup —
the "bounded traversal" is illusory precisely on the cyclic graphs this feature exists to
traverse.

## Failure scenario

An operator wires `multihop_expand` into the orchestrator/MCP pipeline (the ADR-0051 use case)
over a real cyclic KG with high average degree `d` and sets `max_hops=5` (or more) for recall. The
DuckDB CTE's `UNION ALL` re-expands every node each hop, so the `walk` relation grows as `~d**max_hops`
(e.g. `d=50, hops=5` → `~3e8` rows) and the in-memory DuckDB connection exhausts RAM/CPU per agent
search query — an availability/DoS blowup — while the correct `min(hop)` result masks it. Neo4j's
`(s)-[*1..max_hops]-(n)` enumerates all paths for `min(length(p))` with the same exponential class.
`FakeGraphSource`/`_traverse` and networkx complete the same query in `O(V+E)`.

Reproduced against real DuckDB (complete graph K12, seed `n0`, `max_hops=9`): the pre-fix
`UNION ALL` query does not return within 30s; the fixed `UNION` query returns the identical
neighbourhood in ~0.02s.

## What to build

Make the two engine adapters bounded in **work**, matching the reference BFS, with the same result set:

1. **DuckDB** — change the recursive `walk` step from `UNION ALL` to `UNION` (set/distinct). `UNION`
   deduplicates `(node, hop)` across all iterations, so each distinct `(node, hop)` pair is expanded
   at most once and total work is bounded by `max_hops × |edges|` (polynomial). The final
   `SELECT node, min(hop) … GROUP BY node` is unchanged, so the result is byte-identical to the
   faithful fake / networkx reference.
2. **Neo4j** — wrap the variable-length pattern in `shortestPath((s)-[*1..max_hops]-(n))` so the
   engine runs a single bounded BFS per `(s, n)` pair instead of enumerating all paths. `min(length(p))`
   and the `$seeds` binding are unchanged.
3. `NetworkxGraphSource` is already bounded (BFS shortest-path) — untouched.

The fix stays inside each adapter behind the one `neighbors` method (deep-module standard); the
`GraphSource` contract, the `MultiHopExpander` algebra, and statelessness are untouched (additive,
no regression).

## Acceptance criteria

- [x] DuckDB recursive CTE deduplicates the walk (`UNION`, not `UNION ALL`); Neo4j uses
      `shortestPath`.
- [x] Real-DuckDB regression on a cyclic graph (K12, `max_hops=9`) completes within a wall-clock
      budget **and** returns the exact reference neighbourhood — fails on the pre-fix `UNION ALL`
      engine (does not return in 20s), passes on the fix. Gated on the `[graph-duckdb]` extra
      (skipped offline, the established norm for real-engine tests).
- [x] Offline Neo4j regression (through the faithful fake `neo4j` module) asserts the composed
      Cypher uses `shortestPath` (bounded) and still maps records — fails on the pre-fix bare
      path-match, passes on the fix; runs in the offline gate.
- [x] Existing DuckDB/networkx equivalence-to-the-fake tests stay green (result set unchanged).
- [x] Feature tests green + ruff + mypy --strict clean.
