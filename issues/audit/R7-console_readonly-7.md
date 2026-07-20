# R7 / console_readonly / 7 — the console renders an unreadable checkpoint as an empty run

**Severity:** major
**Where:** `packages/latence-console/src/latence_console/artifacts.py` — `RunArtifacts._read_jsonl`
(the two `except (OSError, ValueError): return` arms), surfaced by
`packages/latence-console/src/latence_console/render.py` `_empty` / `_count` in
`render_chunks` / `render_graph` / `render_entity` / `render_quarantine`.
**Invariants violated:** ADR-0034 (no silent errors), ADR-0017/0021 (the files are the source of
truth; the page must not state the opposite of what they say).

## What is wrong

`_read_jsonl` catches `OSError` / `ValueError` twice — once around the `Storage.open` call and once
around the streaming loop — and in both arms simply `return`s an exhausted iterator with **no
signal anywhere**. An unreadable, or partially-readable, checkpoint is therefore
indistinguishable from an empty one, and the browse views turn that into an *affirmative* claim
about the run:

* `/run/<id>/chunks` → "This run checkpointed no chunks."
* `/run/<id>/graph` → "Entities (0)", "Relations (0)", "This run assembled no graph nodes."
* `/run/<id>/quarantine` → "No record in this run's corpus carries a risk marker."

The R5/R6 fixes already taught these views to *qualify* an absence claim when the read was
incomplete (`_count`, `_empty(partial=…)`, the `Partial view` notice), but that qualification is
driven solely by `artifacts.truncated()` — i.e. only by the `MAX_SCAN_RECORDS` cap. An IO failure
never sets it, so the honest "partial read" machinery is bypassed by exactly the case that most
needs it.

The mid-file arm is worse: an `OSError` raised part-way through a checkpoint yields a silent
**prefix** of the file, and the resulting partial list is rendered as a complete one.

## Failure scenario

Reproduced against the real DemoBuilder corpus: copy the demo run, `chmod 000` every file under
`_latence/runs/demo/checkpoints/`, then browse. All three views return HTTP 200 with no error and
no `Partial view` notice, while the run on disk holds 4 chunks, 6 entities, 33 edges and a
`prompt_injection` risk marker.

The triggers are ordinary ops, not exotica:

* differential ACLs on the checkpoints tree — it holds UNREDACTED `content`, so restricting it
  while the Quality Report stays readable is a sensible posture;
* an S3 bucket policy, or a transient 403/5xx, on the checkpoints prefix;
* a file removed by a concurrent ADR-0018 Purge between the `ls` and the `open`.

## What to build

1. `RunArtifacts` remembers every URI whose read failed (both arms, including a failure *during*
   streaming, which leaves a prefix), and exposes it: `unreadable()` and a `partial()` predicate
   that is true when the reader either hit the cap **or** could not read something.
2. The four list views are driven by `partial()`, not `truncated()`, so every capped-scan
   qualification the R5/R6 fixes introduced (`_count`'s "read so far", `_empty(partial=…)`) also
   fires on an IO failure.
3. The `Partial view` notice states the *right reason*: an unreadable-file read gets its own
   wording naming the files that could not be read, rather than claiming the run is too large.
4. No exception escapes and no traceback is rendered — the console stays legible over whatever is
   on disk; it simply stops asserting emptiness it cannot support.

## Acceptance criteria

* **AC1** With every checkpoint file unreadable (`Storage.open` raising `OSError`), `/chunks` and
  `/graph` render the `Partial view` notice and do **not** render
  "This run checkpointed no chunks." / "This run assembled no graph nodes." /
  "This run assembled no graph edges."; the headings read "0 read so far", not "(0)".
  (`/entity` is reachable only when the node itself was read, so it is covered by AC3's
  mid-file case, where the surviving prefix resolves the node but not its edges.)
* **AC2** With the screening sidecar unreadable, `/quarantine` does **not** render
  "No record in this run's corpus carries a risk marker."; it renders the `Partial view` notice.
* **AC3** A failure raised **mid-file** (after some lines were yielded) is recorded too — the
  surviving prefix is never presented as the whole file.
* **AC4** `RunArtifacts.unreadable()` names the URIs that failed; `partial()` is true; the
  pre-existing `truncated()` still means only "the `MAX_SCAN_RECORDS` cap dropped records".
* **AC5** A healthy run is byte-identical to before: no notice, bare counts, the same affirmative
  empty-state text on a genuinely empty run.
* **AC6** No regression: `make test` stays at or above 3320 passed / 29 skipped.
