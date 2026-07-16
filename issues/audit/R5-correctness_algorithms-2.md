# R5 · correctness_algorithms · 2 — pack cannot consume the framework's own `kg_node_centrality` value-signal

**Severity:** major
**Area:** `packages/latence-retrieval/src/latence_retrieval/pack.py` (`_signal_bonus`), against the emitted column in `packages/latence-core/src/latence_core/stages/export.py` (`_context_columns_for_row`, contract at `export.py:286`) and the intent doc `packages/latence-core/src/latence_core/stages/graph_features.py:10-11`.

## What is wrong

The framework emits `kg_node_centrality` as a **per-entity `list<double>`** (one centrality float per entity of a chunk; see `export._context_columns_for_row` → `"kg_node_centrality": [float(v) for v in ...]`, Parquet type `pa.list_(pa.float64())`). `export.py:286` explicitly documents that column as *"`expand`'s `centrality_key` **and** `pack`'s centrality value-signal"*, and `graph_features.py:10-11` names *"the Knapsack packer's `centrality` ... inputs"* as one of the two ADR-0051 consumers.

But `pack`'s `value_signals` path (`Packer._signal_bonus`) requires each present signal to be a **scalar** `int`/`float`. Handed the emitted list it takes the `not isinstance(value, (int, float))` branch and raises:

```
ValueError: pack value_signal 'kg_node_centrality' must be numeric where present, got [0.5, 0.9] for candidate 'c1'.
```

So the documented graph-augmented-pack wiring crashes on the framework's own exported corpus. `expand` consumes the identical column correctly via `_peak_centrality` (reduces the per-entity list to its peak), but `pack` cannot, and no scalar per-chunk centrality column is emitted anywhere — so the documented `pack` consumption is impossible on the framework's own export.

R4's e2e test (`test_export_graph_columns_e2e.py`) only exercises `expand`, so the gap is uncaught.

## Failure scenario

A store copies the exported `kg_node_centrality` column (e.g. `[0.5, 0.9]`) into `candidate.metadata` and wires pack per the docs:

```python
Packer({"budget": 100, "value_signals": {"kg_node_centrality": 1.0}}).process(q, [c1, ...])
```

Every centrality-bearing candidate raises `ValueError` instead of the promised graceful degradation.

## What to build

Make `pack`'s `value_signals` path consume a per-entity centrality **list** the same way `expand` does — reduce it to a scalar (the **peak**, `max`) before applying the weight — so the framework's own emitted `kg_node_centrality` column drives the pack value-signal as documented. Scalar signals (recency, numeric role priority) keep working unchanged; genuinely-malformed values (strings, non-numeric or non-finite list elements) still fail loudly; an absent key still contributes nothing.

## Acceptance criteria

- `Packer({"budget": …, "value_signals": {"kg_node_centrality": w}}).process(q, cands)` where a candidate carries `kg_node_centrality = [0.5, 0.9]` runs without error and applies `w · peak` (= `w · 0.9`) as that candidate's signal bonus, promoting it exactly as a scalar `0.9` would.
- Semantics agree with `expand._peak_centrality`: an empty list contributes `0.0`; a non-numeric / non-finite list element raises loudly; a `bool` is rejected.
- Scalar value-signals and the absent-key graceful-degradation path are unchanged.
- A non-vacuous regression test that FAILS on the old scalar-only code and passes on the fix.
- Feature tests green; ruff + mypy --strict clean. No regression to `expand`, the e2e graph-columns test, or the byte-identical-when-unwired export.
