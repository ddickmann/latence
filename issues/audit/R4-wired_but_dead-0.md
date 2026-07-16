# R4 — ADR-0051 graph features are computed and stamped, but the Export never surfaces them as retrievable columns (wired-but-dead graph-augmented-retrieval seam)

- **Severity:** major
- **Dimension:** wired-but-dead / spec-AC-ADR compliance (ADR-0051 / ADR-0039 / ADR-0017)
- **Branch:** feat/retrieval-tooling
- **Tickets:** #159 (context_columns projection); ADR-0051 (graph features); expand/multihop/pack consumers
- **Status:** fixed

## What is wrong

ADR-0051's graph features (`kg_node_centrality`, `kg_node_community`) and the join keys
(`kg_node_ids`, `neighbor_node_ids`) are computed by the `graph_features` enrichment, projected onto
each chunk's `ContextHeader` by `context_enrichment` (`ContextHeader.kg_node_ids` /
`neighbor_node_ids` / `kg_node_centrality` / `kg_node_community`), and the docstrings of every
query-time consumer promise these keys "travel with the chunk into any vector store as metadata …
present whenever context enrichment ran" (`expand.py` module docstring, ADR-0051's default 1-hop
path).

But the Export (`JsonlParquetExport`, the framework's own corpus materialiser) never surfaces them
as retrievable columns. With `context_columns: true` it emitted exactly two graph-related columns:

- `context_header` — an **opaque** sorted-key JSON string of the whole header. No consumer parses
  it back into per-field keys.
- `neighbor_entity_ids` — the header's `neighbor_node_ids`, but under the **wrong column name**.
  Every consumer's default key is `neighbor_node_ids` (`Expander`, `MultiHopExpander`).

It did **not** emit `kg_node_ids`, `kg_node_community`, or `kg_node_centrality` as columns at all.

A store (e.g. `QdrantBackend._to_candidate`, which copies every non-text payload field into
`Candidate.metadata`) therefore yields candidates whose metadata carries `{context_header,
neighbor_entity_ids, …}` — **none of** the four keys the graph consumers read
(`kg_node_ids` / `neighbor_node_ids` / `kg_node_community` / `kg_node_centrality`). So
`Expander._nodes/_reach/_communities/_peak_centrality` all return empty, `_linked` is always
`False`, and every graph bonus is `0.0`: the augmented list is byte-identical to the first-stage
list. The whole graph-augmented-retrieval seam — and the index-time graph-feature computation
feeding it — is dead weight on the framework's own path.

## Failure scenario

Operator wires `graph_features.degree` at index time, `context_columns: true` at export, and
configures `Expander(hop_weight=0.5, community_weight=1.0, centrality_weight=1.0)` (or
`MultiHopExpander`, or `pack value_signals={'centrality': w}`) at query time. The exported
`records.parquet` contains only `context_header` and `neighbor_entity_ids`. The store copies its
columns into `Candidate.metadata`. Running the Expander over a candidate whose header carried
`kg_node_centrality=[0.9, 0.3]`, `kg_node_community=['c-acme', 'c-acme']` leaves every score
unchanged — the graph signal is silently dead.

## What to build

Make the opt-in `context_columns` projection surface the graph header under the **exact metadata
keys the consumers default to**, so a store round-trips them into `Candidate.metadata` and the
graph seam is alive end-to-end on the framework's own corpus. Emit, in addition to the opaque
`context_header` (kept for full fidelity):

- `kg_node_ids` — native `list<string>`
- `neighbor_node_ids` — native `list<string>` (rename of the mis-named `neighbor_entity_ids`)
- `kg_node_community` — native `list<string>`
- `kg_node_centrality` — native `list<double>`

All aligned with the `ContextHeader` fields; `[]` when the header is absent or the graph-feature
enrichment was OFF (the additive / byte-identical-when-unwired guarantee — absent `context_columns`
⇒ the schema is unchanged).

## Acceptance criteria

1. With `context_columns: true`, `records.parquet` contains columns named exactly `kg_node_ids`,
   `neighbor_node_ids`, `kg_node_community`, `kg_node_centrality` (plus `context_header`), populated
   from the header.
2. Feeding those columns into `Candidate.metadata` as `QdrantBackend._to_candidate` does, then
   running `Expander(hop_weight, community_weight, centrality_weight)`, produces the graph bonuses
   (scores change) — proving the seam is alive off the framework's own export.
3. Without `context_columns` the Parquet schema is byte-identical to today (no graph columns).
4. A header-less / unenriched row writes empty graph columns (no crash).
5. No regression: feature test subset + ruff + mypy --strict clean.
