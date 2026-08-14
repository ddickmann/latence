# Wave 4-I — the Stage input plan becomes descriptor data (`input_plan` + `StageInputs`)

Architecture pass 2026-08-13 (see `../architecture-pass.md`, row 4-I). One refactor commit
(`00e3a93`) on top of Wave 2-E's `run_store` and Wave 3-G's `reuse_overlay` re-homings, plus a
docs/comment follow-up (`b67ed68`, ADR-0033's symbol pointers + FINDINGS F11 — no code): the shape
and the wiring rules move onto the Capability, the resolution moves into one object, behavior is
byte-identical — proven by running the gates, not by assertion.

## The friction, as found on the current tree

`capability_descriptor.py` exists so that "Adding a Capability = adding one of these" is literally
true. The INPUT half had never moved in, so **"what does Graph Assembly receive?" was a four-place
question**:

1. `_run`'s four-way branch, which picked BOTH the `incoming` shape and the `records_in` counting
   rule (accumulator / per-carrier / streamed / gathered);
2. two hand-maintained frozensets (`_STREAMED_INPUT_KINDS`, `_PER_CARRIER_INPUT_KINDS`) that the
   descriptor's fields did **not** derive — while `_ACCUMULATOR_KINDS` DID (the precedent);
3. `_dispatch_stage` re-deciding the shape a second time
   (`gathered = [] if capability in _STREAMED or _PER_CARRIER else list(incoming)`), to keep a
   branch from double-reading a one-shot generator;
4. the `_require_*` family — a table (required carriers, forbidden carriers, message) expressed as
   ~10 methods over ~356 lines, **three of them dead**.

Plus the seam leak: `StageDispatch.dispatch_stage(...)` took the Runner's entire mutable
`outputs: dict[str, list[StageOutput]]` because the dispatch re-opened per-carrier streams itself.

## The deepening (commit `00e3a93`)

- **`CapabilityDescriptor.input_plan`** — a required field, so the exhaustiveness gate covers it:
  `StageInputPlan(shape, requires, exclusive, scan_short_circuit)`. `StageInputShape` names the
  executor's four arms; `CarrierRequirement(carrier, message, subclass_ok)` and
  `ExclusiveCarrier(carrier, message)` each carry the **exact** typed error they raise, so every
  pinned wiring message has exactly one copy and lives beside the Capability it constrains.
  `STREAMED_INPUT_KINDS` / `PER_CARRIER_INPUT_KINDS` are now DERIVED (like `ACCUMULATOR_KINDS`),
  and `index_descriptors` refuses at import a `shape is ACCUMULATOR` that disagrees with
  `accumulates` — the executor shapes memory off one and delivers input off the other; they name
  one fact and can no longer drift.
- **New module `stage_inputs.py`** — `StageInputs`: `gathered()` / `stream()` / `stream_of()` /
  `corpus_records_of()` / `count()` / `validate()` / `carriers_present()` / `materialised`, over a
  three-method `StageRecordSource` (the local `RunRecords`: live retained map first, checkpoint
  fallback, reads routed through the engine's public `stage_records` / `stage_record_count` so the
  subclass test seam still sees every read). No I/O of its own, no Provider, and it re-decides
  nothing the descriptor declares — including the ADR-0061 fold, which stays `ReuseOverlay`'s rule
  (this module supplies each site's enumeration + trailer policy, exactly as the six hand-written
  gathers did).
- **The seam narrows.** `StageContext` drops **both** `incoming` and `outputs` for one
  `inputs: StageInputs`; `dispatch_stage` loses two parameters. The Airflow adapter builds its
  context through the engine's new public `LocalRunner.stage_inputs(...)`, so it no longer needs a
  second copy of which Capabilities stream and which gather.
- **The dispatch is provider-call-per-kind.** Each branch is now: check the Provider against the
  Capability's declared Protocol → `inputs.validate()` → ask for the shape → call the Provider.
  `validate()` is called from EVERY branch at the exact position the old `_require_*` call sat,
  and is a no-op that reads nothing for a Capability with no rule — which is what lets the dispatch
  stop knowing which Capabilities validate. The second shape decision is not merely deleted but
  unnecessary: every accessor re-reads through the source, so `gathered()` is idempotent.
- **`latence stack check` gains a `wiring` family.** A Stage whose parents cannot supply a required
  carrier is decidable offline from the same two declarations (`input_plan` + the Stage carrier),
  and it raises the SAME message the Runner would after a full extraction. Only the "required"
  half is checked, and only for a Stage that has parents — the exclusive rules are per-record facts
  a linter would have to guess at, and an empty input is legitimately empty at both ends.

## The input_plan table (per Capability, as declared)

| Capability | shape | required carriers (in order) | exclusive | scan short-circuit |
|---|---|---|---|---|
| `source` | gathered | — | — | — |
| `intake_screening` | gathered | — | — | — |
| `parse` | streamed | — | — | — |
| `chunk` | streamed | — | — | — |
| `content_screening` | gathered | — | — | — |
| `schema_induction` | streamed | — | — | — |
| `entity_extraction` | streamed | — | — | — |
| `relation_extraction` | gathered | EntityMention, ChunkRecord | — | — |
| `fused_entity_relation` | streamed | — | — | — |
| `redaction` | streamed | — | — | — |
| `profiling` | accumulator | DocumentRecord | — | — |
| `type_consolidation` | accumulator | EntityMention | — | — |
| `disambiguation` | accumulator | EntityMention | — | — |
| `graph_assembly` | accumulator | — | DisambiguationRecord | — |
| `graph_completion` | accumulator | — | GraphRecord | — |
| `context_enrichment` | per_carrier | ChunkRecord, GraphRecord | — | **yes** |
| `embedding` | gathered | — | — | — |
| `export` | gathered | — | — | — |

The four shapes partition the kinds exactly as the executor's `if/elif/elif/else` did, and the two
derived frozensets equal the old literals (pinned in `test_stage_inputs.py`).

`scan_short_circuit` is **declared, not derived** from "has no exclusive rule" (which is the
soundness condition). It changes how much of the input is READ, and only Context Enrichment
(dogfood-3) ever had it: the accumulator checks still pay the full pass. Turning that into an early
exit is a measurable change to make deliberately — logged as a follow-up, not taken here.

## Deleted

- the two frozensets `_STREAMED_INPUT_KINDS`, `_PER_CARRIER_INPUT_KINDS` (now derived);
- the three **dead** validators (zero call sites, verified repo-wide): `_require_profiling_inputs`,
  `_require_disambiguation_inputs`, `_require_graph_inputs`;
- the live validator family, now data: `_require_relation_inputs`,
  `_require_context_enrichment_inputs_streamed`, `_require_profiling_inputs_streamed`,
  `_require_disambiguation_inputs_streamed`, `_require_type_consolidation_inputs_streamed`,
  `_require_graph_inputs_streamed`, `_require_graph_completion_inputs_streamed`, and
  `_scan_input_carriers`;
- the eight input-resolution privates re-homed into `StageInputs`: `_records_of_type`,
  `_stream_inputs_of_type`, `_stream_parent_inputs_of_type`, `_stream_all_records_of_type`,
  `_gathered_count`, `_gather_inputs`, `_stream_doc_inputs`, `_streamed_input_count`
  (`_stage_record_count` became the public `stage_record_count`, the counting twin of the existing
  public `stage_records`).

`runner.py` **2741 → 2029 lines** at `00e3a93` (2033 after the comment follow-up); the new
`stage_inputs.py` is 368. Public surface otherwise unchanged; `_stage_records`,
`_as_type` / `_as_type_stream` / `_as_records`, the Export overlay fold and `_stream_checkpoint`
(the read-counting test seam) are untouched.

Two test seams MOVED with the code they instrumented (they were Wave 3-G's, on methods that no
longer exist): `_MaterialisingRunner` now overrides `stage_inputs` and materialises `stream_of`;
`test_e2e_embedding_disambiguation`'s chunk-drop probe patches `StageInputs.corpus_records_of`. The
two `_InputSpyRunner`s read `inputs.materialised` — the transient itself — instead of inferring it
from a parameter's runtime type.

## Byte-identity gate (run, both legs, A/B worktrees)

Before = pristine `d1631b6` worktree, after = `00e3a93` worktree; each leg's full `packages/*/src`
set pinned on `PYTHONPATH` (verified: `latence_core` resolved from the leg's own tree), same store
path, same run ids, `PYTHONHASHSEED=0`.

- **Batch leg** — 6 docs from `benchmark/sota-corpus/ticket03-proof/input/markdown` through the
  ticket-03 s3 zero-LLM profile (gliner2 extract + redact on CPU, dense `granite-embedding-311m`
  export ON): **30/33 files byte-identical** — all 14 checkpoints, all 10 export artifacts (corpus
  jsonl+parquet with embeddings, graph nodes/edges/hyperedges parquet, hyperedge KV, graphml, ttl,
  bm25 postings+stats), manifest, quarantine/screening/type-vocabulary sidecars. The 3 remaining
  are exactly the established divergence set: `wal.log` identical after dropping `ts`,
  `quality-report.json` + `.md` identical after dropping the duration fields. Every metrics row's
  `records_in`/`records_out` is identical, including the accumulator counts (`profiling 441 in`,
  `enrich 506 in`) — the counting rules survived the move intact.
- **Incremental delta leg** (the code actually touched): base build of 4 docs then a 2-doc add via
  `latence delta` (noembed profile; run ids `w4i-base`/`w4i-add`): **73/81 files byte-identical**,
  including every checkpoint, every export, and the whole committed corpus store
  (`_latence/corpus/**`: doc-level reuse records, schema, WAL delta bodies). The 8 normalized are
  the two run `wal.log`s (ts), the two `quality-report` pairs (durations) and the two corpus
  `version.json`s (`created_ts`). Churn witness: the add run reports
  `2 extracted, 4 reused` — the overlay fold genuinely engaged on the compared path.

## Gates

- **Full latence-core suite** at the committed revision, single process, `PYTHONHASHSEED=0`:
  ran to `[100%]` with **2,353 passed / 7 skipped / 0 failed** counted from the progress markers
  (zero `F`/`E`, zero `FAILED`/`ERROR` lines) and **pytest exit 0**. Same shape as Wave 3-G's tail
  (2,323 + 7 there; +30 is exactly the new `test_stage_inputs.py`), and again only the final count
  line did not flush after the warnings summary — the F9/F10 teardown-quirk family, without the
  crash.
- Targeted packs: `latence-runner-airflow` (the second Runner, whose `StageContext` construction
  changed), `latence-disambig-embedding` e2e (the corpus-wide chunk gather), `latence-demo`,
  `latence-schema-inducer`, `latence-parser-document`, `latence-pii-presidio`, `latence-embedder-st`
  — all green.
- New `test_stage_inputs.py` (30 tests): the derived kind sets vs the historical literals, the
  shape partition, the import-time shape/accumulates gate, the shapes and their laziness, the
  per-shape counting, **all nine wiring messages pinned byte-for-byte**, requirement order, the
  exclusive rule, the declared short-circuit (and that no exclusive plan may declare it), the
  all-reused-parent overlay top-up, and the three trailer policies — every one of them driven
  against a FAKE record source, which the old form made impossible.
- `ruff check packages/` clean; `mypy --strict` clean over `latence-core` + `latence-runner-airflow`
  (105 source files).
