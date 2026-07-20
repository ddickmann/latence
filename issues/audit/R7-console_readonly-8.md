# R7 / console_readonly / 8 — the quarantine + screening sidecar path is unbounded in the run's size

**Severity:** major
**Where:** `packages/latence-console/src/latence_console/artifacts.py`
(`RunArtifacts.quarantined`, `RunArtifacts.findings`, `RunArtifacts.flag_checkpoints`),
reached from `server.py::ConsoleApp._quarantine_view` → `browse.quarantine_entries` /
`browse.flag_entries`.
**Invariant at stake:** ADR-0033 (memory-bounded reading), ADR-0034 (no silent
resource-exhaustion failure), and the module's own documented promise
*"Bounded in the run's size (audit R4)"*.

## What is wrong

The audit-R4 memory-bounding fix wired two things: `_read_jsonl` streaming, and the
`MAX_SCAN_RECORDS`-capped `_scan()` — but the cap was applied to the **checkpoints** path
only. The **sidecar** path (`quarantine/*.jsonl`, `screening/*.jsonl`) was left uncapped:

* `quarantined()` builds a full Python list of every `QuarantineRecord` in the run;
* `findings()` builds a full Python list of every `ScreeningFinding` in the run, `PASS`
  verdicts included;
* `flag_checkpoints()` then folds that whole list into a dict.

R6 hardened this very page (`_bounded_lines`) without noticing the sidecar path, and the
module docstring still claims the reader is bounded in the run's size. It is not.

This is not a memory/latency tradeoff. `flag_checkpoints()` needs only the `FLAG`-verdict
subset; the `PASS` findings — one per document per screening checkpoint, i.e. the
overwhelming majority — are materialised, held, and then discarded unread.

## Failure scenario

Measured with `tracemalloc`: a run whose `screening/content_screen.jsonl` holds 200,000
routine `PASS` `ScreeningFinding`s (57.4 MB on disk — one finding per document, entirely
ordinary for a 200k-document corpus, and small next to the 142 MB checkpoint the R4 issue
itself cites) makes `GET /run/q/quarantine` peak at **289.5 MB** of traced Python
allocation to render a 3.9 KB page that contains none of those findings.

The cost is linear and uncapped: a 2M-document corpus puts ~2.9 GB behind a single browse
request and the console is OOM-killed — the exact resource-exhaustion failure R4 was filed
to close, on the one page R6 hardened. For contrast, the chunk inspector over the same tree
is budgeted at 24 MB by the existing regression tests.

## Acceptance criteria

1. Rendering the quarantine view over a run with a large screening sidecar costs memory
   bounded by `MAX_SCAN_RECORDS`, **not** by the number of records on disk.
2. `flag_checkpoints()` never retains a `PASS` finding: it streams the sidecars and keeps
   only the `FLAG`-verdict `(document_id, category) -> checkpoint` entries.
3. `quarantined()` and `findings()` are capped at `MAX_SCAN_RECORDS` like the checkpoint
   scan.
4. Truncating a sidecar is **surfaced**, never silent: `truncated()` / `partial()` go true
   and the page hedges its counts and absence claims, exactly as the capped checkpoint scan
   already does (ADR-0034, ADR-0017/0021 — a capped read may never be rendered as an
   affirmative "the run holds none").
5. Behaviour on runs that fit under the cap is unchanged, byte-for-byte, including the
   plain (unhedged) wording of a complete read.
6. A non-vacuous regression test asserts the memory budget directly with `tracemalloc` and
   fails on the pre-fix reader.
