# R4 — the #188 browse views materialise a whole run in memory and render every row into one page

- **Severity:** major
- **Dimension:** console read-only / scale + resource exhaustion
- **Branch:** feat/release-fleet-integration
- **Tickets:** #188 (the debugging views), ADR-0010/0017 (run state is files), ADR-0033 (memory-bounded), ADR-0042/0044 (redaction floor)
- **Status:** fixed

## What is wrong

`packages/latence-console/src/latence_console/artifacts.py` (`_read_jsonl`, `_scan`) and
`packages/latence-console/src/latence_console/render.py` (`render_chunks`, `render_graph`,
`render_quarantine`).

Three compounding faults:

1. **Whole-file materialisation.** `_read_jsonl` called `Storage.read_text(uri)` — the entire
   JSONL checkpoint decoded into one `str` — and then `.splitlines()`, a second full copy as a
   list of `str`. A 142 MB `checkpoints/redact.jsonl` therefore cost ~300 MB before a single
   record was parsed.
2. **Whole-run materialisation, even for a single-record view.** `_scan()` read *every* checkpoint
   file and built dicts of every `ChunkRecord`, `DocumentRecord`, `EntityMention`, `GraphNode` and
   `GraphEdge` in the run. `chunk()`, `document()`, `mention()` and `node()` — the four
   *single-record* lookups behind the chunk inspector and the entity page — all went through it.
   Opening one 6 KB chunk page cost the whole run.
3. **Unpaginated rendering.** `render_chunks` / `render_graph` / `render_quarantine` emitted one
   `<tr>` (or one edge `<div class="card">`) per record, with no cap and no pager, into a single
   HTML document.

Nothing is cached between requests *by design* (`RunIndex.artifacts()` builds a fresh
`RunArtifacts` per request, so the console keeps no state — ADR-0010/0017), so every page load
repeated the whole cost.

## Failure scenario

Probed against a real store: a run scaled to 50,000 chunk records (a 142 MB
`checkpoints/redact.jsonl` — modest for a real corpus).

- `ConsoleApp.respond("GET", "/run/big/chunks")` returned a single **9.5 MB HTML page** in 4.6 s
  with **1.35 GB peak RSS**.
- `ConsoleApp.respond("GET", "/run/big/chunk?id=…#chunk-7")` — a 6 KB page showing **one** chunk —
  cost 3.8 s and **1.4 GB peak RSS**, because `RunArtifacts.chunk()` went through `_scan()`.

A 500k-chunk run (~1.4 GB of checkpoints) extrapolates to >13 GB and an OOM. There was no row cap,
no pagination, and no scale caveat in `packages/latence-console/README.md` or
`docs/getting-started/run-console.md`.

The green gate cannot see this: every fixture run under `packages/latence-console/tests/` has 1–5
chunks, so peak memory and page size are indistinguishable from the fixed code.

## What to build

1. **Stream JSONL.** `_read_jsonl` reads through `Storage.open(uri, "rb")` line by line and decodes
   per line — never the whole file as one `str`. The same torn-line / bad-encoding tolerance holds.
2. **Bounded single-record lookups.** `chunk()`, `document()`, `mention()` and `node()` stream the
   checkpoints **most-downstream file first** and retain only the matching record, so the chunk
   inspector and the entity page are O(1) in run size. They reuse the full scan only when a list
   view on the same request already built it (the graph view resolves many mentions).
3. **A bounded scan for the list views.** `_scan()` caps how many records of each kind it retains
   (`MAX_SCAN_RECORDS`) and reports `truncated()`. To make truncation **redaction-safe**, the scan
   now walks checkpoints in *reverse* dependency order with downstream-wins semantics
   (`first file to supply a key wins`, later lines within the same file still win), so every record
   retained is the most-downstream version — a truncated scan can never surface a pre-Redaction
   chunk (ADR-0042/0044). Identical output to the old last-wins order when nothing is truncated.
4. **Pagination.** `render_chunks`, `render_graph` and `render_quarantine` render at most
   `PAGE_ROWS` rows per page with a `?page=` pager, and state honestly when the scan was truncated.
5. **Docs.** A scale caveat in `packages/latence-console/README.md` and
   `docs/getting-started/run-console.md`.

## Acceptance criteria

- [x] `RunArtifacts.chunk()` on a large run never materialises the run: it reads no more than the
      one checkpoint file that supplies the record, and holds no per-record maps.
- [x] `_read_jsonl` never calls `Storage.read_text`; a run whose checkpoint is far larger than the
      allowed peak is still readable.
- [x] `/run/<id>/chunks`, `/run/<id>/graph` and `/run/<id>/quarantine` emit a bounded page
      (`PAGE_ROWS` rows) with a working `?page=` pager; page 1 links to page 2 and back.
- [x] A truncated scan is stated on the page, never silently presented as complete.
- [x] The most-downstream (redacted) version of a chunk is still the one shown, both in the list
      and in the inspector — the reverse-order scan is byte-identical to the old order on the
      existing fixtures.
- [x] Non-vacuous regression test (`tests/test_console_scale.py`) builds a run far larger than the
      memory/page budget and asserts the budget; it fails on the pre-fix code and passes on the fix.
- [x] `make test` stays at or above 2985 passed / 28 skipped; ruff + mypy --strict clean.
