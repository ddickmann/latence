# Wave 3-G — `ReuseOverlay`: ADR-0061 §5's fold rule, written once

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 3-G). One refactor commit
(`d0cda6c`): pure re-homing of a total function of (stage, carrier, pipeline) — behavior
byte-identical, proven by running the gates, not by assertion.

## The deepening (commit `d0cda6c`)

- **New module** `packages/latence-core/src/latence_core/reuse_overlay.py` — the single home of
  the W17/ADR-0043 + ADR-0061 cross-run reuse fold:
  - `merge_overlay_doc_ordered` — the document-interleaved arrival-order merge (moved verbatim
    from `LocalRunner._merge_overlay_doc_ordered`), now directly unit-testable;
  - `natural_record_order` — commit 4249620's emission-order key, applied at `ReuseOverlay`
    construction (the read-side fix for the lexicographically-id-sorted reuse store);
  - `stage_produces` / `is_doc_level_stage` / `folds_at` — the fold-target decision, one place;
  - `overlay_carriers` — the mixed-carrier (`Record` sentinel → both mention carriers)
    expansion, previously inlined at three sites;
  - `ReuseOverlay.fold_into(expected, sources, pipeline, trailer=…)` — the fold parameterized
    by the caller's enumeration + a NAMED trailer policy (`OverlayTrailer`); plus
    `count_for_stage` (the counting twin), `records_for_carrier` (the Export face) and
    `has_carrier` (the wiring-check face). Pure module: no I/O, no Storage.
- `runner.py`: the six hand-synchronised re-derivations become one call each; each 15–20-line
  docstring re-derivation replaced by a pointer to the module. Public surface, method
  signatures and the two test seams (`_stream_parent_inputs_of_type` subclass override,
  `_stream_all_records_of_type` monkeypatch) unchanged. `_records_of_type` is the one new
  private (the per-Stage carrier-narrowed stream every fold site hands to `fold_into`).
  Deleted from `runner.py`: `_ProvenanceCarrier`/`_TP`, `_DIGIT_RUNS`/`_natural_record_order`,
  `_overlay_of_type`, `_is_doc_level_stage`, `_stage_produces`, `_merge_overlay_doc_ordered`,
  and the two inline mixed-carrier expansions (`_gathered_count`, the Export branch).

## 6→1 consolidation — site → call, with each site's preserved trailer policy

| former site (runner.py) | enumeration | now | trailer policy (REAL per-site behavior, preserved exactly) |
|---|---|---|---|
| `_stream_inputs_of_type` | `depends_on` parents | `fold_into` | `APPEND_IF_CARRIER_UNSEEN` — accumulator union semantics; a corpus-level parent producing the carrier counts as *seen* (its checkpoint already carries the union), so nothing is appended there |
| `_stream_parent_inputs_of_type` | `depends_on` parents | `fold_into` | `NEVER` — a doc-level Stage's input is its parents' records, never records no parent produced |
| `_stream_all_records_of_type` | whole pipeline, topological order | `fold_into` | `ALWAYS` — corpus-wide contract: an unclaimed overlay IS the corpus's only records of that carrier |
| `_gathered_count` | `depends_on` parents | `count_for_stage` | counting twin: once per carrier at the first doc-level producer; corpus-level parents top up nothing |
| Export dispatch branch | nearest-producer walk | `records_for_carrier` + `merge_overlay_doc_ordered` | *produces* is tautological under the walk — only the level gate remains; the mixed sentinel delivers both mention overlays |
| `_scan_input_carriers` | `depends_on` parents | `has_carrier` | exact-type lookup (a mixed `Record` declaration is not a concrete overlay carrier) — as before |

## Direct unit tests (new: `test_reuse_overlay.py`, 20 tests)

The merge previously had NO direct tests (only the heavy e2e delta suites). Now pinned at unit
level: the rebuild arrival-order property (leading/middle/trailing reused documents,
within-document order — the property the ADR-0061-named order-dependences reduce to: resolver
first-seen display spelling, hyperedge γ-tie/`merged_sources` order, profiler last-document
fields, export row order); the natural-key interleave regression (id-sorted store → numeric
emission order, including a >10-chunk document through `fold_into` — 4249620's shape, still
covered at e2e by `test_e2e_delta_enrichment_incremental` and now also pinned directly); fold
at the FIRST doc-level producer only; corpus-level suppression; all three trailer policies;
`count_for_stage` once-per-carrier / mixed / corpus-level-zero; the Export and wiring-check
faces (mixed-sentinel expansion, exact-type `has_carrier`).

## Gates (all run)

- **e2e delta oracles**: `test_e2e_delta_incremental` + `test_e2e_delta_enrichment_incremental`
  + `test_e2e_delta` + `test_delta_chaos` — green (29 tests). Plus the delta pack
  (contract/state/provider/observability/cli/type-vocabulary), the context-enrichment +
  accumulator streaming packs, `test_capability_descriptor`, `test_run_store`, `test_runner`,
  `test_e2e_s3`, and `latence-disambig-embedding`'s e2e — all green.
- **Wave 2-E real-pipeline byte-diff protocol (batch leg)**: 6 docs from
  `benchmark/sota-corpus/ticket03-proof/input/markdown` through the s3 zero-LLM profile (CPU,
  dense embedder ON), before = pristine `fc58406` worktree, after = `d0cda6c` worktree (each
  leg's full `packages/*/src` set pinned on `PYTHONPATH`, same store path both legs):
  **30/31 files byte-identical** — all 14 checkpoints, all 10 export artifacts (corpus
  jsonl+parquet with embeddings, graph nodes/edges/hyperedges parquet, hyperedge KV, graphml,
  ttl, bm25 postings+stats), manifest, quarantine/screening/type-vocabulary sidecars; `wal.log`
  identical after dropping `ts`; quality-report json+md identical after dropping the duration
  fields — exactly the Wave 2-E divergence set (wal ts + report durations), nothing else.
- **Incremental delta leg** (the code actually touched): base build of 4 docs then a 2-doc add
  via `latence delta` (noembed profile; run ids `w3g-base`/`w3g-add`), same A/B worktrees:
  **73/77 files byte-identical** including every checkpoint, every export, and the whole
  committed corpus store (`_latence/corpus/**`: doc-level reuse records, schema, WAL delta
  bodies); the 4 normalized files are the two run `wal.log`s (ts) and the two corpus
  `version.json`s (`created_ts`); quality reports duration-only diffs. Churn witness: the add
  run reports `documents_reused: 4, documents_extracted: 2, corpus_record_count: 787,
  to_version: 1` — the overlay fold genuinely engaged on the compared path.
- **Full latence-core suite** at the merged head (with Wave 3-H's `17822c7` in the tree),
  single process, `PYTHONHASHSEED=0`: ran to `[100%]` with **2,323 passed / 7 skipped / 0
  failed** counted from the progress markers (zero `F`/`E`, zero `FAILED`/`ERROR` lines) and
  **pytest exit 0** — a clean exit, unlike F9's SIGSEGV; only the final count line did not
  flush after the warnings summary (the same teardown quirk family, now without the crash).
- `ruff check` + `mypy --strict` clean on `reuse_overlay.py`, `runner.py`,
  `test_reuse_overlay.py`.
