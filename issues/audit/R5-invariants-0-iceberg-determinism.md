# R5-invariants-0 — The opt-in Iceberg Export injects a uuid4 filename into the export deliverable, breaking the determinism invariant

- **Severity:** major
- **Dimension:** invariants (DETERMINISM — "no hidden `now()`/random in pipeline or export paths")
- **Branch:** feat/release-fleet-integration
- **File:** `packages/latence-core/src/latence_core/iceberg.py` (`IcebergCorpusTable._publish_data_file`)
- **Consumed by:** `packages/latence-core/src/latence_core/stacks/validate.py` — `_export_artifacts`
  (walks `export/` recursively) and `_check_determinism` (a plain `primary == second`)
- **Status:** fixed

(Filed under a disambiguated name: `R5-invariants-0.md` is already taken by an unrelated
`feat/retrieval-tooling` ticket, per the existing `R1-invariants-0-gliner-device.md` precedent.)

## What is wrong

`_publish_data_file` names the immutable Iceberg data file
`<out_dir>/iceberg-data/<basename>-<uuid.uuid4().hex>.parquet`. That is a hidden random value on
the **export deliverable path** — the one place the framework promises there is none.

The randomness was there to satisfy Iceberg's immutability requirement (the catalog must never
point at a path whose bytes later change, which rules out the deliberately-rewritten
`records.parquet` itself). But a **content-addressed** name satisfies that requirement *more*
strongly: a path named after the sha256 of its own bytes can only ever hold one byte sequence.
The randomness bought nothing the hash does not buy.

## Failure scenario

Wire the documented optional `iceberg` block on the Export stage (a REST catalog URI) and run
`latence stack validate <stack.yaml>`:

1. Each Export run copies `records.parquet` to `<out_dir>/iceberg-data/records-<uuid4>.parquet`.
2. `_export_artifacts` walks `export/` **recursively** and keys artifacts by run-relative path, so
   the primary run yields `{'records.parquet', 'iceberg-data/records-<uuidA>.parquet'}` and the
   fresh determinism run yields `{'records.parquet', 'iceberg-data/records-<uuidB>.parquet'}`.
3. `_check_determinism` is a plain `passed = primary == second`, so it **FAILS** and reports
   `export diverged — primary_only=['iceberg-data/records-<uuidA>.parquet'],
   second_only=['iceberg-data/records-<uuidB>.parquet']` on a pipeline that is in fact perfectly
   deterministic — a false red that reads as a pipeline-nondeterminism bug.

Secondary consequences of the same defect:

- the export directory is not reproducible, defeating any adopter-side byte-diff or idempotency
  check of the deliverable;
- re-runs into a stable `out_dir` accumulate unbounded full-corpus copies (a new uuid every run),
  where content addressing collapses identical content onto one file.

No test covered it: `test_export_iceberg.py` only asserted that some returned URI contains
`/iceberg-data/`.

## What to build

Derive the data file's name from the sha256 of the bytes actually published, computed **streamed**
— folded into the copy that already streams through the `Storage` seam — so ADR-0033's O(chunk)
memory bound survives (never hash a whole-corpus buffer).

Two consequences to handle honestly:

- If the content-addressed target already exists, the copy is **skipped**: the bytes are by
  construction identical, and a file the catalog may already reference must never be rewritten.
- Identical content in the same `out_dir` now maps to ONE path, so `add_files` must be called with
  `check_duplicate_files=False` to keep the documented `append` semantics (a repeated run adds a
  snapshot; a repeated FULL export duplicates rows — exactly the pre-existing behaviour, which the
  uuid previously produced via a second byte-identical copy).

## Acceptance criteria

1. Two independent Export runs over identical records into two fresh `out_dir`s produce
   **byte-identical export trees** when compared by run-relative key — the real
   `latence_core.stacks.validate._export_artifacts` of both runs compares equal, *including* the
   `iceberg-data/` entry, so `_check_determinism` passes.
2. The data-file name is content-derived and stable: identical corpus bytes ⇒ identical file name;
   different corpus bytes ⇒ different file name.
3. Memory stays O(chunk): the digest is computed over the streaming copy (ADR-0033).
4. Iceberg immutability is preserved: the catalog never references a path whose bytes can change,
   and an already-published identical data file is not rewritten under the catalog.
5. The documented `append` / `overwrite` delta semantics are unchanged — a repeated run in `append`
   mode still accumulates a snapshot and still scans back both runs' rows.
6. OFF-BY-DEFAULT stays byte-identical and `pyiceberg` is still never imported when the block is
   absent.
7. A regression test fails on the old code and passes on the fix; `make test` stays green at or
   above 2985 passed / 28 skipped.
