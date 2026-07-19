# R2-console_readonly-12 — Chunk inspector reports chunk-LOCAL offsets as document offsets when a chunk has no `offset_map`

**Severity:** major
**Area:** `packages/latence-console/src/latence_console/browse.py` (`_redaction_disposition`)
**Invariant at risk:** ADR-0021/0031 — Provenance is true: reported offsets map to real original-document offsets.

## What is wrong

`_redaction_disposition` resolves each PII span's position in the source document through the
chunk's `offset_map`. When the chunk carries **no** `offset_map` it falls back to the raw
chunk-LOCAL offsets:

```python
source_start, source_end = span.char_start, span.char_end   # chunk-local!
```

and then feeds those chunk-local numbers into `_pages_for_span(page_map, ...)`, which is a
**document**-coordinate page map. Both the "In the source document" span and the "Page(s)"
column are therefore silently wrong — no error, no "—".

Core takes the correct degrade for exactly this case
(`packages/latence-core/src/latence_core/redaction_policy.py:289`):

```python
orig_start = plan.base_char_start + local_start
```

and `contracts.py:112` documents the rule: a chunk with no offset map "degrades to the
`char_start + local` shift". The console diverges from the framework's own documented rule.

## Failure scenario (observed, not hypothetical)

A `ChunkRecord` from a third-party Chunk Provider — or a pre-v11 checkpoint the console is
explicitly built to read — with `page_map` set, `offset_map=None`,
`provenance.char_start=91`, and a PII span at chunk-local `0–12` with `page_start=None`.

Probe against the real `browse.inspect_chunk`:

```
PII row  source_span = 0–12    pages = 1      <- WRONG
chunk    resolved_span = 91–120  resolved_pages = 2
```

Truth is `91–103` on page **2**. The *same rendered page* simultaneously reports the enclosing
chunk at the correct document offsets, so the inspector contradicts itself, and emits a false
Provenance claim with no error surfaced.

No existing test exercises a chunk that has `pii_spans` but no `offset_map`
(`tests/reports.py::_sensitive_chunk` omits `offset_map` but carries zero spans), which is why
the suite stayed green.

## What to build

In `_redaction_disposition`, when the chunk has no `offset_map`, shift each span by the chunk's
document anchor `chunk.provenance.char_start`, matching core's documented degrade. When the
chunk has **no** anchor either (`provenance.char_start is None`), there is no way to place the
span in the document: report the source span and pages as unknown (`—`) rather than passing
chunk-local numbers off as document coordinates.

The `chunk_span` column keeps showing the raw chunk-local offsets — that is its job.

## Acceptance criteria

1. With `offset_map=None` and `provenance.char_start=91`, a PII span at chunk-local `0–12`
   renders `source_span = 91–103` and resolves to the page containing offset 91.
2. With `offset_map=None` **and** `provenance.char_start=None`, `source_span` and `pages`
   render `—` (never chunk-local numbers presented as document offsets).
3. `chunk_span` is unchanged in every case (still the chunk-local span).
4. Behaviour when `offset_map` IS present is byte-identical to before.
5. A regression test covers case 1 and 2; it fails on the pre-fix code.
6. `make test` stays green at or above 2985 passed / 28 skipped.
