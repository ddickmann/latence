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

## What it deliberately is not

- **Not a writer.** It reaches only `exists` / `ls` / `read_text` / `modified` on the Storage seam.
  `POST`, `PUT`, `PATCH` and `DELETE` are refused with `405` before a path is even parsed.
- **Not stateful.** `ConsoleApp` holds nothing between requests — no session, no cache, no index it
  builds and persists. Every page is re-derived from the files at request time, which is why
  "no new state" is structural here rather than a promise (ADR-0010, ADR-0017).
- **Not networked.** Pages are single self-contained HTML documents with inlined CSS: no CDN, no
  font host, no script, no beacon. The server binds the loopback interface only and refuses any
  other bind address. It works with the network cable pulled out.
- **Not a JSON dump.** The Quality Report's nineteen schema versions of counts are translated into
  sentences with a verdict attached. The raw report is still on Storage for anyone who wants it.

## Scope

This is the tracer slice (#186): run list + Quality Report view. The deep inspectors — the
quarantine browser, the chunk inspector, the knowledge-graph browser — are the follow-up slice
(#188).

Licensed under Apache-2.0.
