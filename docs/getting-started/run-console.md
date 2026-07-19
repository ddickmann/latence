# Read the results: the run console

A run leaves everything a stakeholder needs behind as files — the run manifest, the
[Quality Report](../adr/0008-quality-report-as-first-class-artifact.md), the corpus and the
graph ([ADR-0010](../adr/0010-local-runner-state-is-files-on-storage.md),
[ADR-0017](../adr/0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md)). The **run
console** is the face over those files: one command, a localhost page, and a quality verdict
a non-engineer can act on.

## One command

```bash
pip install latence-console

latence-console ./corpus            # a folder a run wrote to
latence-console s3://bucket/prefix  # or any fsspec Storage location
```

It prints the URL and serves it on this machine only:

```
Latence run console — read-only, this machine only.
  open http://127.0.0.1:8787/
  Ctrl-C to stop.
```

Point it at the same location you passed as `--out` / `storage_uri`. After the
[one-command demo](index.md#the-one-command-path):

```bash
uv run latence-demo                 # writes to /tmp/latence-demo
latence-console /tmp/latence-demo
```

## What you see

**The run list.** Every run on that Storage location, newest first, with the pipeline it ran,
how many documents it read, when it started and finished, and a status:

| Status | What it means |
| --- | --- |
| **Passed** | The run finished and every quality check held. |
| **Needs attention** | The run finished, but a check did not hold — read the checks before using the output. |
| **Failed** | A pipeline step raised. The output is incomplete. |
| **Unfinished** | A manifest exists but no Quality Report: still running, or it stopped early. |

**The Quality Report view.** Select a run and the report is rendered as headline verdicts
first — plain sentences with the numbers in them:

> *Every one of the 82 pieces of output can be traced back to the exact page it came from.*
>
> *The knowledge graph holds 6 things connected by 33 relationships, 100.0% of which quote the
> sentence they came from.*
>
> *3 files were held back as unsafe and 1 item was flagged for review.*

Beneath the verdicts sits the drill-down: the per-Stage table (what ran, records in → out,
time, outcome — including a step that was **skipped** because the machine could not run it),
why each file was held back or flagged, the knowledge-graph statistics, the corpus and
extraction counts, the contract-completeness numbers, and — for an incremental run
([ADR-0018](../adr/0018-incremental-corpus-deltas.md)) — what changed since the previous
version.

The worst verdict is listed first, and a check the run never made is never claimed: a pipeline
with no Redaction Stage gets **no** reassuring "no personal data found" line.

## What it deliberately does not do

- **It never writes.** It reaches only `exists` / `ls` / `read_text` / `modified` on the
  Storage seam. `POST`, `PUT`, `PATCH` and `DELETE` are refused with `405`. Browsing every
  page leaves the storage tree byte-identical.
- **It holds no state.** There is no database, no cache, no index it builds and persists.
  Every page is re-derived from the files at request time, so a run that lands while the
  console is open shows up on the next reload.
- **It never phones home.** Pages are single self-contained HTML documents with inlined CSS —
  no CDN, no font host, no script, no analytics. It works with the network cable pulled out,
  and it binds the loopback interface only (it refuses any other bind address).
- **It answers only this machine.** A request is refused with `403` unless its `Host` names
  loopback (`localhost`, `127.0.0.1`, `[::1]`) on the port the console was started on, and any
  `Origin` it carries is a loopback origin too. Binding to loopback alone would not stop DNS
  rebinding — a page in another tab re-resolving its own domain to `127.0.0.1` reaches the
  socket from your own browser, and only the `Host` header gives it away.
- **It is not a JSON dump.** The machine-readable `quality-report.json` is still on Storage
  for anyone who wants it; the console is the human half.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--port` | `8787` | Localhost port to serve on. |
| `--host` | `127.0.0.1` | Loopback address to bind. A non-loopback address is refused. |

## Scope

This is the run list plus the Quality Report view. The deep inspectors — the quarantine
browser, the chunk inspector, and the knowledge-graph browser — are the follow-up slice.
