# latence-console

The read-only run console for the [Latence framework](https://github.com/ddickmann/latence-framework).

One command turns the files a pipeline run already left on Storage into a page a **non-engineer**
can read:

```bash
pip install latence-console
latence-console ./corpus      # or: latence-console s3://bucket/prefix
```

It serves `http://127.0.0.1:8787/`:

- **The run list** — every run on that Storage location with its status (Passed / Needs attention /
  Failed / Unfinished), pipeline, document count, and start/finish timestamps.
- **The Quality Report view** — for a selected run: headline verdicts in plain language
  ("Every one of the 82 pieces of output can be traced back to the exact page it came from"),
  then the per-Stage table, the screening reasons, the knowledge-graph statistics, the
  completeness numbers, and — for an incremental run — what changed since the previous version.

And, from every run page, the debugging loop:

- **Held back & flagged** (`/run/<id>/quarantine`) — every record Screening removed from the
  pipeline, with the checkpoint, detector, category, reason and the bytes withheld; and every
  record it kept but *flagged*, with the risk marker that rode into the corpus so a RAG consumer
  can exclude it.
- **The chunk inspector** (`/run/<id>/chunks`) — answers "why is this chunk wrong". Each chunk
  resolves back through its **Provenance** to its source document, the source *page*, and the
  character offsets it occupies there — resolved through the record's own `offset_map`/`page_map`
  with the framework's own resolvers, so the console cannot disagree with the pipeline about where
  a chunk came from. Its Classification and its Redaction disposition are stated alongside,
  including the case that matters most: a chunk whose redaction policy ran *no detectors* says the
  control was OFF, rather than letting an empty PII list read as "clean".
- **The knowledge-graph browser** (`/run/<id>/graph`) — entities and edges, navigable. **Every edge
  is shown with its Evidence**: the source mentions it rests on, resolved to their surface text,
  file, page and character offsets, with the confidence. A mention the run no longer holds is
  shown as unresolved, never quietly dropped.

Only the corpus's **PII-handled** text is ever rendered: a chunk's `masked_content` when Redaction
produced one, and never the parent document's raw markdown.

## What it deliberately is not

- **Not a writer.** It reaches only `exists` / `ls` / `read_text` / `open(uri, "rb")` / `modified`
  on the Storage seam. `POST`, `PUT`, `PATCH` and `DELETE` are refused with `405` before a path is
  even parsed.
- **Not stateful.** `ConsoleApp` holds nothing between requests — no session, no cache, no index it
  builds and persists. Every page is re-derived from the files at request time, which is why
  "no new state" is structural here rather than a promise (ADR-0010, ADR-0017).
- **Not networked.** Pages are single self-contained HTML documents with inlined CSS: no CDN, no
  font host, no script, no beacon. The server binds the loopback interface only and refuses any
  other bind address. It works with the network cable pulled out.
- **Not a JSON dump.** The Quality Report's nineteen schema versions of counts are translated into
  sentences with a verdict attached. The raw report is still on Storage for anyone who wants it.

## On a large run

A real run's checkpoints are hundreds of megabytes, and the console re-reads them on every request
(it caches nothing, by design). So it is **bounded**, not exhaustive:

- Checkpoints are **streamed** line by line — never decoded into memory whole — and a single-record
  view (the chunk inspector, an entity page) is a targeted lookup that costs one file pass, not the
  whole run.
- A **list** view reads at most `MAX_SCAN_RECORDS` (2000) records of each kind. Past that the page
  says so, in a *Partial view* banner — it never presents a prefix as if it were the whole run.
  Every record it does list is the most-downstream (post-Redaction) version, so a truncated list can
  never show text that a Redaction Stage had already rewritten.
- Every list view **paginates** at `PAGE_ROWS` (200) rows per page (`?page=`), so page size does not
  grow with the run.

For an exhaustive audit of a large run, read the corpus and checkpoint files directly — they are the
source of truth (ADR-0017), and the console is a viewer over them.

## Scope

The tracer slice (#186) built the run list and the Quality Report view; #188 deepened it with the
quarantine browser, the chunk inspector and the knowledge-graph browser. All of it reads material
the pipeline already wrote — Provenance, Evidence, offset maps, screening sidecars — so the deep
views add **no** new persistent state and no new dependency.

Licensed under Apache-2.0.
