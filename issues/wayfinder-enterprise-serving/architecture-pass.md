# Architecture pass — 2026-08-13 (improve-codebase-architecture, autonomous mode)

Hard gate (Dennis): **all outputs byte-identical afterwards** — benchmark cells, campaign
artifacts, exports, rankings. Oracles: ticket-01 parity harness (21,215-query 0-diff
baseline), ticket-02 serving parity summaries, ticket-03 incremental-equivalence comparator,
the committed `s5-retrieval/results/*.json`, and the full 5,139-test suite.

Three read-only explorations (core / retrieval+serving / campaign+benchmark) produced 18
candidates and a rejected list. Selection below is mechanical: deletion-test gate, ADR
respect, disjoint-file waves, one oracle per candidate.

## Selected, in waves (files disjoint within a wave)

| Wave | ID | Deepening | Oracle |
|---|---|---|---|
| 1 | A | Graph-source read index: file-identity-keyed caching (the ADR-0056-amendment pattern) behind `NetworkxGraphSource`/`DuckDBGraphSource`/`DuckDBGraphDocumentSource`, `cache=False` = verbatim old path; + explicit `_INDEX_CACHE` lifetime policy (LRU + superseded-signature eviction). Closes FINDINGS F1; fixes the tiered-search × per-call-reconnect hazard | ticket-01 parity.py extended (cached vs uncached), package tests |
| 1 | B | KG export publication rides one atomic-writers set like the corpus export (no torn `graph-*.parquet`, no half-published TTL/KV set) | ticket-03 comparator (sha256s those files) + test_graph_export |
| 1 | C | Benchmark/wizard: fix the `or default` int-flag defect (`--retrieval-seed-top-k 0` silently swallowed); result-cell schema module (`cell.py`, five undeclared shapes → one); `LegSpec` table replacing ~8 membership-test sites | re-render REPORT.md/metrics.json from committed cells + forced re-run of one cell, `git diff --exit-code`; wizard round-trip byte tests |
| 1 | D | `blocking` module: public home for the blocking key + closure (`_tokens`/`_record_reach` privates un-leaked, pseudo-mention hack removed, duplicate `MentionView` renamed) | full suite (delta/disambiguation tests) + ticket-03 comparator |
| 2 | E | `RunStore`: the run layout + checkpoint decode module (12 hand-built paths in 6 packages, 6 `# noqa: SLF001` reach-throughs → one owned interface; ADR-0010's missing owner) | full suite + ticket-03 comparator |
| 2 | F | KG-column contract: `KgColumns` single parser (4 duplicate parsers, prose lockstep invariant → code); `CandidateDecorator` seam on `GraphRescueRetrieval`, `decorate=None` default (byte-identical), decoration opt-in — records the shipped-vs-bench divergence (bm25-only candidates lose KG payload) honestly | serving parity reference-lane before/after; package suites |
| 3 | G | `ReuseOverlay`: ADR-0061 §5 fold rule 6 sites → 1 module + direct unit tests for `_merge_overlay_doc_ordered` | ticket-03 comparator + the three e2e delta suites |
| 3 | H | `TorchRuntime` owner for process-global torch state + the missing embedder-st composite (closes F7 structurally) | roster-host suite single-process (the F7 configuration) + perf tests |
| 4 | I | `StageInputs` + `input_plan` on `CapabilityDescriptor` (450-line dispatch switch, 2 frozensets, 10 validators → data; deletes 3 dead `_require_*`) | full suite (error messages pinned) + ticket-03 comparator |
| 5 | J | `GraphRescueRetrieval.retrieve()/select()` split → delete `benchmark/s5/rescue.py`'s duplicate pipeline (~600 lockstep lines). **Blocked until the pod sweep completes**; requires a NEW A/B oracle (RescuePipeline vs shipped, full unions + gates + selections, `==` per query, full gold sets) | the new A/B harness, modelled on ticket-01 parity.py |
| 5 | K | S5 becomes a real campaign stage (`run(plan)` API + StageSpec + wizard `s5:` block) and the `retrieval:` block gets its consumer (`rescue_config()` pinned to `_CONFIG_KEYS`) | dual-entry cell diff; wizard non-interactive byte-diff (committed campaign.yaml regenerated in-commit) |

Waves 3–5 sequential where they share `runner.py` (E → G → I). One candidate = one agent = own commits = own oracle run. A candidate that cannot keep its oracle green is abandoned, main stays green.

## Rejected with load-bearing reasons (pass memory — do not re-propose)

- `contracts.py` split (ADR-0006: one schema, dual serialization; cross-record guarantees)
- Bounded-staleness / incremental BM25 df / embedding cache (ADR-0061 §2/§3 rejected with measurement; escape hatch = a measured ADR amendment)
- Unifying in-run full-union vs committed-store affected-set resolution (ADR-0061 §2 deliberate split; ADR-0018 bound)
- BM25 tokenizer duplication core↔retrieval (documented import-independence, ADR-0048/0049)
- Engine/stage pairs (ported algorithms + provenance plumbing; real seam, second providers exist)
- The gold-resolution stack, stage runner/state machine, `Bm25Backend`, `generate_pipeline_yaml`, s5-vs-library tuning divergence (each deep or deliberately divergent; see exploration reports)
- Parity-oracle code paths (`_scan_candidates`, `cache=False`, `_exhaustive_hits`, pure-Python kernels): **untouchable** — they are the mechanism by which byte-identity is provable
- `latence-sink-qdrant` reshape (index-time; follows the KG-column module, doesn't lead it)
- `IdentityProcessor`, `fuser.py`, `registry.py`, `orchestrator.retrieve`, `quality_report_builder.py`, `StageExecution` (each already deep / already the target shape)

Janitorial riders (attach to the wave touching the file): dead exports `pagemap.PAGINATED_DATASETS`, `Export.files`; `run_matrix.py:250` no-op; ADR index cross-ref for the 0049/0056 solver-lineage confusion.
