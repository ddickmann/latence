# R6-console_readonly-12 — `MAX_LINE_BYTES` is a phantom guard

**Severity:** major
**Area:** `packages/latence-console/src/latence_console/artifacts.py` (`_read_jsonl`, constant at :85-87)
**Status:** FIXED

## What is wrong

`_read_jsonl` iterated the raw handle directly:

```python
for raw in handle:
    if len(raw) > MAX_LINE_BYTES:
        continue
```

File iteration reads a whole line into one `bytes` object *before* `len(raw)` can be tested, so the
guard rejects a line the process has already allocated in full. The reader therefore accumulates
exactly the memory its own docstring says it refuses to accumulate:

> "How many bytes of a single JSONL line the reader will accumulate before abandoning it … reading
> it into memory is the very failure this module bounds."

## Failure scenario (measured)

A run whose `checkpoints/chunk.jsonl` opens with a single 64 MB line, then a real chunk record:
`GET /run/torn/chunk?id=<doc>#chunk-3` peaked at **129.9 MB** of traced allocation against a 24 MB
budget. This is not hypothetical — `checkpoints/source.jsonl` carries the raw ingested bytes inline,
one document per line (measured max line 4,194,780 bytes for a 2 MB `dump.txt`), so a run over a
100 MB PDF puts a ~200 MB single line in front of *every* browse request.

## What to build

Assemble lines from the raw stream instead of asking the stream for one line: read in fixed blocks,
split on `\n`, and abandon a line's accumulated bytes the moment they would exceed `MAX_LINE_BYTES`,
then skip the remainder of that line without buffering it.

## Acceptance criteria

- [x] Peak allocation while reading past an over-long line is bounded by
      `MAX_LINE_BYTES + READ_BLOCK_BYTES`, independent of the line's real size.
- [x] Abandoning a line skips exactly that line: records before and after it still parse and render.
- [x] A final line with no trailing newline is still yielded; blank/torn lines still skipped.
- [x] The `contains` pre-filter, targeted lookups and list views are unchanged in behaviour
      (off-by-default / byte-identical outputs for every run without an over-long line).
- [x] Regression test fails on the old reader and passes on the new one:
      `packages/latence-console/tests/test_console_scale.py::test_overlong_line_is_never_accumulated`
      (pre-fix: 129.9 MB peak vs 24 MB budget) plus
      `::test_records_around_an_overlong_line_still_render`.
- [x] Suite stays at or above 3320 passed / 29 skipped.
