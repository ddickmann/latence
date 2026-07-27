# R2 — ADR-0033 was flipped to "accepted" while every doc-level Stage's INPUT was still whole-corpus

- **Severity:** major
- **Dimension:** invariants and ADRs / ADR-0033 (streaming, memory-bounded runner)
- **Branch:** feat/arch-deepening
- **Site:** `docs/adr/0033-streaming-memory-bounded-runner.md` (header/status, "Honest memory
  boundary", "Remaining") vs
  `packages/latence-core/src/latence_core/runner.py` — `LocalRunner._run`
  (`incoming = self._gather_inputs(...)` for every non-accumulator Stage), `_gather_inputs`,
  `_as_type`, `_extract_input_chunks`; and
  `issues/backlog-post-v1/110-batch-docs-doc-level-streaming.md`
- **Invariants at stake:** ADR-0033 (O(batch), no whole-corpus materialisation), ADR-0034 (typed
  errors, never silent), determinism / byte-identical outputs, ADR-0016 thin core (no new deps)
- **Status:** FIXED (this commit)

## What was wrong

C6 streamed the **output** side of a doc-level Stage: `_dispatch_stage` hands the Provider's
iterator back uncollapsed and `_run` writes it record-by-record into the checkpoint, so a Stage's
records never exist as a list. On the strength of that, this branch flipped ADR-0033 from
*partially accepted* to **accepted**, rewrote "Honest memory boundary" to claim

> The per-doc-level-stage transient is `O(checkpoint batch)` + whatever the Provider itself holds.

and

> **Net:** IO transients (Export + checkpoint write), the retained live set, and the
> per-doc-level-stage transient are all bounded

leaving only "Peak RSS at 100k / 1M documents" under **NOT bounded today** — and retired issue
#110 in prose as "still NOT BUILT, and **no longer needed for memory**".

The **input** side was untouched. `LocalRunner._run` still called `_gather_inputs` for every
non-accumulator Stage, concatenating all parents' records into one `list[StageOutput]` that lived
in `ctx.incoming` for the Stage's entire execution. `_as_type` then built a **second** list of the
same size, and under `extract_on: masked` `_extract_input_chunks` built a **third** — of fresh
`model_copy` objects. An Entity Stage over a 1M-chunk corpus therefore still held ~1M ChunkRecords
(up to 3×) and OOMed exactly as before C6, while the ADR recorded that transient as bounded.

The ADR's own preserved warning — *"do not claim a bound the code does not hold"* — was violated by
the ADR's own rewrite, and the issue that tracked the missing bound was retired on that basis.

## Failure scenario (reproduced)

Source → Parse → Chunk → Entity over a plaintext corpus, instrumenting the input the Runner hands
each Stage:

| corpus | `len(ctx.incoming)` at `parse` | at `chunk` | at `entities` |
|---|---|---|---|
| 4 documents | 4 | 4 | 4 |
| 16 documents | 16 | 16 | 16 |

(The audit's own probe, on a corpus that chunks 12× per document, measured the same shape at
`content_screening` and `entity_extraction`: 48 records at 4 documents, 192 at 16 — the record
count per Stage scales with the corpus, whatever the chunking ratio.)

Exactly linear in corpus size, at every doc-level Stage — not `O(checkpoint batch)`. And the whole
input was read, decoded and held *before* the Provider saw its first record: with a probe
extractor counting how many chunk-checkpoint records had been read at each pull, a 6-chunk corpus
gave `[6, 6, 6, 6, 6, 6]` instead of `[1, 2, 3, 4, 5, 6]`.

Pre-fix output of the new regression tests:

```
E  AssertionError: Stage 'parse' received a MATERIALISED input list: 4 records at 4 documents,
   16 records at 16 documents — the input transient is corpus-linear, not O(checkpoint batch).
   ADR-0033's 'Honest memory boundary' claims this transient is bounded.
E  assert (4 is None)

E  AssertionError: the Entity Stage's input was not pulled lazily: reads-per-pull
   [6, 6, 6, 6, 6]… over 6 chunks. Streaming means the k-th pull has read exactly k records;
   a materialised input reads the whole checkpoint before the Provider sees anything.
E  assert [6, 6, 6, 6, 6, 6] == [1, 2, 3, 4, 5, 6]
```

## What was built

**The code half — stream the input, for the Capabilities whose Protocol allows it.**
`StageContext.incoming` is now an `Iterable[StageOutput]`, not a `list` — a *smaller* interface (a
consumer may iterate it at most once), not another layer. For
`runner._STREAMED_INPUT_KINDS` — Parse, Chunk, Schema Induction, Entity Extraction, Fused,
Redaction, i.e. every doc-level Capability whose Provider consumes an `Iterable` exactly once —
`_run` hands the Stage `_stream_doc_inputs(...)`: a lazy stream over the parents' outputs, read
from the live dict if retained and otherwise back from their checkpoints, in the same
`depends_on` order and yielding byte-identical records. Narrowing (`_as_type_stream`) and W18
masking (`_extract_input_chunks_streamed`) are lazy too, so the second and third copies are gone
as well. Both sides of those Stages are now `O(checkpoint batch)`.

`records_in` stays exact without a buffer: `_streamed_input_count` counts a released parent's
checkpoint **lines** (one record per line by construction of `_write_checkpoint`) rather than
decoding and re-validating the corpus a second time, so the span / lineage / `StageMetrics` column
is unchanged and costs no memory.

The four remaining doc-level Capabilities are left materialised **on purpose, and are now
documented as unbounded**: Screening returns a `ScreeningOutcome` holding passed + quarantined (and
Content Screening's page-map re-homing needs the input set to know what was dropped), Relation
Extraction and Context Enrichment partition one stream into two carriers, and Export writes a file.
Streaming the Runner's half would move no memory because the Provider materialises the same
records anyway — that is a Provider-Protocol or a scheduling bound (#110), not a dispatch one.

One check was **removed rather than ported**: `_require_redaction_inputs` could never fire (the
`_as_type` above it already rejected any non-chunk record), so its better W16/ADR-0042 wiring
advice never reached an operator. It is now the streamed narrowing's own message for the redaction
branch — one check instead of two, and reachable for the first time.

**The doc half — make ADR-0033 true again.** Status back to *partially accepted*, with the header,
"Honest memory boundary", "Built", "Remaining" and "Rejected" sections stating the bound
per-Capability: output bounded everywhere (C6), input bounded for the six streamed Capabilities
(R2), input still `O(corpus-slice)` for the other four. Issue #110 is un-retired and rescoped to
exactly those four, with the R2 lesson recorded in its DoD ("an ADR status flip is not a memory
fix").

## Acceptance criteria

- [x] No doc-level Stage in `_STREAMED_INPUT_KINDS` receives a materialised input list, at any
      corpus size.
- [x] The stream is genuinely lazy: pulling the k-th input record has read exactly k records back
      from the parent's checkpoint.
- [x] `records_in` is unchanged and exact, computed without decoding the corpus twice.
- [x] A mis-wired doc-level Stage still fails with the same typed `ContractError` (ADR-0034) — no
      silently-empty Stage.
- [x] ADR-0033 states only bounds the code holds, per-Capability, and names the four that are
      still unbounded; issue #110 is restored and rescoped to them.
- [x] Regression tests that FAIL pre-fix:
      `packages/latence-core/tests/test_doc_level_input_streaming.py::test_doc_level_stage_input_is_never_materialised_whole_corpus`
      and
      `packages/latence-core/tests/test_doc_level_input_streaming.py::test_entity_stage_pulls_its_input_one_record_at_a_time`
- [x] `make test` green (3839 passed, 29 skipped — baseline 3836 + the 3 new); `make lint` and
      `make type` (mypy --strict) green.
