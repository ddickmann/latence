# R5 · wired_but_dead · 3 — pack's centrality leg cannot consume the exported `kg_node_centrality` list

**Severity:** major
**Status:** RESOLVED (duplicate) — already fixed on this branch in commit `e984dda` under ticket `issues/audit/R5-correctness_algorithms-2.md`.
**Area:** `packages/latence-retrieval/src/latence_retrieval/pack.py` (`_signal_bonus` / `_signal_scalar`), against the emitted column in `packages/latence-core/src/latence_core/stages/export.py:147-148,163` and ADR-0051 §27.

## What was wrong

The framework emits `kg_node_centrality` as a **per-entity `list<double>`** (one centrality float per entity of a chunk; `export._context_columns_for_row`, Parquet type `pa.list_(pa.float64())`). `export.py` and ADR-0051 §27 explicitly name `pack`'s centrality value-signal — "the Knapsack packer's centrality inputs" — as a consumer of that column.

But `pack`'s `value_signals` path (`Packer._signal_bonus`) required each present signal to be a scalar `int`/`float`. Handed the emitted list it took the `not isinstance(value, (int, float))` branch and raised:

```
ValueError: pack value_signal 'kg_node_centrality' must be numeric where present, got [0.5, 0.9] for candidate 'c1'.
```

So the documented graph-augmented-pack wiring crashed on the framework's own exported corpus. `expand` consumed the identical column fine via `_peak_centrality`, so the ADR-0051 R4 claim that `pack` reads graph features off the exported corpus was false — pack's centrality leg was wired-but-dead.

## Failure scenario

A store copies the exported `kg_node_centrality` column (e.g. `[0.5, 0.9]`) into `candidate.metadata` and wires pack per the docs:

```python
Packer({"budget": 100, "value_signals": {"kg_node_centrality": 1.0}}).process(q, [c1, ...])
```

Every centrality-bearing candidate raised `ValueError` instead of the promised graceful signal application, crashing the entire pack stage.

## What was built (in `e984dda`)

`pack`'s value-signal path now reduces a per-entity centrality **list** to its **peak** (`max`) before applying the weight (`_signal_scalar` → list/tuple branch), identically to `expand._peak_centrality`, so both ADR-0051 query-time consumers read the one emitted `kg_node_centrality` column the same way. Scalar signals keep working unchanged; an empty list contributes `0.0`; non-numeric / non-finite elements and `bool` fail loudly.

## Acceptance criteria (all met by `e984dda`)

- `Packer({"budget": …, "value_signals": {"kg_node_centrality": w}}).process(q, cands)` with a candidate carrying `kg_node_centrality = [0.5, 0.9]` runs without error and applies `w · peak` (= `w · 0.9`). ✅
- Semantics agree with `expand._peak_centrality`: empty list → `0.0`; non-numeric / non-finite element raises; `bool` rejected. ✅
- Scalar value-signals and absent-key graceful degradation unchanged. ✅
- Non-vacuous regression tests that FAIL on the old scalar-only code and pass on the fix, in `packages/latence-retrieval/tests/test_pack.py`:
  - `test_pack_consumes_the_frameworks_per_entity_centrality_list_by_its_peak`
  - `test_pack_centrality_list_peak_matches_the_equivalent_scalar_signal`
  plus empty-list, non-numeric-element, and non-finite-element cases. ✅
- Feature tests green; ruff + mypy --strict clean; no regression to `expand`, the e2e graph-columns test, or the byte-identical-when-unwired export. ✅

## Disposition

No further code change. The R5 audit surfaced this defect twice under two slugs (`correctness_algorithms-2` and `wired_but_dead-3`); the fix landed under the former. This ticket is filed for traceability of the `wired_but_dead` slug and closed as a resolved duplicate.
