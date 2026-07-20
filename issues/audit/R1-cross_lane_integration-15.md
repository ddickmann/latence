# R1 — the Qdrant sink and the Qdrant retrieval backend disagree on the payload key holding the chunk text

- **Severity:** major
- **Dimension:** cross-lane integration / ADR-0034 (no silent errors)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #183 (qdrant sink), #173 (QdrantBackend), ADR-0017, ADR-0034, ADR-0048
- **Status:** fixed

## What is wrong

Two halves of the same product meet at one Qdrant collection and neither names the other's key.

The **write** side: `JsonlParquetExport` calls the text column `content`
(`packages/latence-core/src/latence_core/stages/export.py:879` — `pa.field("content", pa.string())`),
and the sink carries **every column except `embedding`** into the payload verbatim
(`packages/latence-sink-qdrant/src/latence_sink_qdrant/export_reader.py:_to_row`). So every point the
sink loads holds the chunk text under the payload key `content`.

The **read** side: `QdrantBackend` — the shipped consumer of exactly that collection — defaulted
`text_key` to `"text"` (`packages/latence-retrieval/src/latence_retrieval/qdrant.py:134`) and then
popped it with a **silent empty-string fallback**
(`packages/latence-retrieval/src/latence_retrieval/qdrant.py:279` —
`text = payload.pop(self._text_key, "")`).

Nothing anywhere documented the required override: `grep -rn text_key packages/latence-sink-qdrant/
docs/` returned nothing.

## Failure scenario

Proven end to end. A real Export (`records.parquet` written by `JsonlParquetExport`) is loaded by the
real `QdrantSink` into the shipped `FakeQdrantClient` — `LoadReport(... upserted=2 ...)`, stored
payload keys `['_latence_digest', '_latence_load_id', 'classification', 'content', 'media_type',
'provenance', 'record_id', 'schema_version']`. There is no `text` key. A **default-configured**
`QdrantBackend` over that same collection then pops a key that is not there and returns every
`Candidate` with `text == ""`:

```
AssertionError: assert {''} == {'alpha text', 'beta text'}
```

No error, no warning. Downstream, the rerank / pack / bm25-rescore processors all score on
`Candidate.text`: the reranker scores nothing, the Packer packs zero tokens, and the run reports a
**green retrieval** over a collection that carries the text perfectly well.

## What to build

1. Make the two shipped halves agree by default: `QdrantBackend`'s `text_key` defaults to
   `DEFAULT_TEXT_KEY = "content"` — the Export's own column name, i.e. the key the sink actually
   writes. A foreign collection is the `text_key` override, which keeps working unchanged.
2. Kill the silent fallback (ADR-0034): a **non-empty** payload that does not carry `text_key` raises
   a typed `BackendSearchError` naming the key it looked for and the keys the point does have, plus
   the config fix. Payload *keys* are schema, so naming them leaks no content; payload **values**
   never appear in the message. A point with no payload at all is unchanged (empty candidate) — the
   store returned nothing to map.
3. Document the handoff on both sides (sink README, retrieval README, retrieval tutorial) so the key
   is named where either half is read.

## Acceptance criteria

- A collection loaded by `QdrantSink` and queried by a **default-configured** `QdrantBackend`
  returns `Candidate.text` equal to the chunk text the pipeline exported.
- A `text_key` the payload does not carry raises `BackendSearchError` (never `text == ""`); the
  message names the missing key and the keys present, and contains no payload value.
- An explicit `text_key` override still reads a foreign collection's key.
- The cross-package handoff is covered by a test that loads a **real** Export through the **real**
  sink and reads the **verbatim stored payload** back through the backend
  (`packages/latence-sink-qdrant/tests/test_retrieval_handoff.py`), so neither package can drift
  from the other again.
- No regression: `make test` green at or above 2985 passed / 28 skipped.
