# R4-invariants-0 — The Qdrant sink's Export reader leaks chunk content (PII) into an error message

**Severity:** major
**Invariant:** ADR-0034 (no content/PII in anything an operator reads back), ADR-0044/0042 (the
universal redaction floor exists precisely to keep this text out of the places humans copy-paste)
**Site:** `packages/latence-sink-qdrant/src/latence_sink_qdrant/export_reader.py:150` (`_to_row`)

## What is wrong

```python
msg = f"The Export at {uri!r} has a record with an empty {RECORD_ID_COLUMN!r}: {raw!r:.200}"
```

`raw` is the *entire* raw Parquet row — every exported column, including `content`, which is the
chunk text itself. That message becomes a `SinkExportError` and lands in operator logs / stderr /
the ticket a human files about the failed load.

The `:.200` truncation does not save it. The Export schema order is
`record_id, schema_version, provenance, classification, content, ...`, so with an *empty*
`record_id` (the only way to reach this branch) the repr reaches `content` at roughly char ~155 —
well inside the 200-char window — and emits it verbatim.

This is the exact discipline the same stream enforces on the sibling retrieval backend, where the
test asserts `"delta" not in str(excinfo.value)  # keys are schema; VALUES are content — never
leaked` (`packages/latence-retrieval/tests/test_qdrant_backend.py:351`).

## Failure scenario

Write a `records.parquet` with one row `record_id=""`,
`content="Patient Jane Doe SSN 123-45-6789 card 4111111111111111"`, then run
`latence-sink-qdrant load` at that directory (the sink accepts any Export-shaped Parquet, including
hand-built or third-party ones). The raised error reads:

```
The Export at '.../records.parquet' has a record with an empty 'record_id':
{'record_id': '', 'schema_version': 1, 'provenance': '{"offset": 0, ...}',
 'classification': '{"level": "internal"}', 'content': 'Patient Jane Doe SSN 123-45-6789 card
```

Redacted-corpus content and PII, verbatim, on the one path where a human will paste the log
somewhere else.

## Acceptance criteria

1. The empty-`record_id` error names the offending row by **position only** (row index within the
   Export) and, at most, the *names* of the columns present — never any column **value**.
2. No other error string in `export_reader.py` interpolates a raw row or a `content` value.
   (`record_id` itself is content-addressed identity, not content, and stays.)
3. A regression test writes an Export whose `content` is a distinctive PII-shaped sentinel and a
   blank `record_id`, and asserts the sentinel does **not** appear in `str(excinfo.value)` while the
   error is still actionable (says which row).
4. `make test` stays green at or above 2985 passed / 28 skipped.
