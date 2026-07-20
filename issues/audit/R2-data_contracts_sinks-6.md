# R2-data_contracts_sinks-6 — `verify-absent` must not prove erasure by looking at nothing

**Severity:** major
**Stream:** feat/release-fleet-integration (#183, Qdrant sink)
**Sites:**
- `packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py:340-341` (`verify_absent`)
- `packages/latence-sink-qdrant/src/latence_sink_qdrant/cli.py:127-135` (the `verify-absent` verb)
- `packages/latence-sink-qdrant/README.md:78-81` (runbook step 4 — "prove it")

## What is wrong

`QdrantSink.verify_absent` short-circuits to `[]` — the affirmative "absent" answer — when the
target collection does not exist. The CLI reads that empty list as proof and prints
`verified absent: N record id(s) are not in the collection` with exit code 0.

An empty list from `verify_absent` is the *evidence* the documented GDPR Retraction/Purge runbook
ends with. But the current code cannot distinguish

* "I asked the store and the records are gone", from
* "I looked at a collection that does not exist and therefore saw nothing".

Both produce the same exit-0 green.

## Failure scenario

1. Load an Export containing record `c1` into collection `corpus`.
2. Run the README's step-4 audit command against a **mistyped** collection:

   ```bash
   latence-sink-qdrant verify-absent c1 --collection corpuss --url $QDRANT_URL
   ```

3. Output: `verified absent: 1 record id(s) are not in the collection`, exit code **0** — while
   `c1` is still present, retrievable and searchable in `corpus`.

The identical false green occurs with a correct collection name against the wrong `--url`
(a staging cluster, a stale env var), i.e. every operator typo in this command turns into
affirmative erasure evidence.

`test_verify_absent_on_a_collection_that_does_not_exist` currently asserts this behaviour as
intended, so the green suite cannot see the defect.

## What to build

`verify_absent` must refuse to answer when there is nothing to inspect: a missing collection is a
**typed `SinkError`**, not an affirmative absence. This is deliberately distinct from `load`, whose
create-if-absent semantics on a missing collection are correct — an audit verb has no business
creating or assuming anything.

- New typed error `SinkVerificationError(SinkError)` in `errors.py`, exported from the package.
- `verify_absent` raises it when `_collection_exists()` is false, with a message naming the
  collection and telling the operator to check `--collection` / `--url`, and carrying no store
  text, URL or api key (ADR-0034).
- The CLI already maps `SinkError` to exit code 2 with `error: …` on stderr — so the typo path
  becomes a loud non-zero, never a green.
- README runbook step 4 and the CLI/module docstrings state the guarantee.

## Acceptance criteria

1. `QdrantSink(cfg, client=fake).verify_absent(["c1"])` against a non-existent collection raises
   `SinkVerificationError` (a `SinkError`), and does **not** return `[]`.
2. The raised message names the collection and mentions checking the collection/URL; it contains
   no api key and no raw store/driver text.
3. `cli.main(["verify-absent", "c1", "--collection", "<typo>"])` returns exit code **2** and prints
   nothing resembling `verified absent` on stdout, while `c1` is still live in the real collection.
4. `load` is unchanged: loading into a non-existent collection still creates it, and loading a
   zero-record Export against a non-existent collection still returns a `LoadReport` without
   raising.
5. `verify_absent` against an existing collection is unchanged (empty list = genuinely absent,
   non-empty = the still-present subset).
6. `make test` stays green at or above 2985 passed / 28 skipped.
