# Task: what the retrieval removal left behind

Type: task
Mode: HITL
Status: open
Blocked by:

## Question

*Remove the retrieval tooling* took out the packages, the core modules and the retrieval docs.
The rewrite of the ten stale pages on 2026-11-10 turned up things that removal missed, plus
places where the code and its own help text disagree. Each needs a fix or a decision.

**Left over from the removal**

1. **The release gates still demand removed capabilities.** `RELEASE-GATES.md` Gates 2 and 3,
   and its closing section, require SPLADE proven as a live signal against a real Qdrant and a
   graph leg proven to retrieve. `test_public_release_gate.py` asserts that "splade" and
   "graph leg" appear there. Neither can ever be met now. **Owner's decision:** what the release
   gates are for the re-scoped product. The test changes with the gates, as one decision.
2. **The benchmark campaign still reuses pre-removal exports.**
   `latence_benchmark/campaign/pipeline_stage.py::verify_reused_export` accepts an export by
   finding `graph-hyperedges.kv`, a file only the earlier pipeline wrote. The reuse path exists
   to feed the removed retrieval stage. Decide whether the reuse path goes, or is re-keyed on a
   file the current pipeline writes.
3. **The labelling tool still has a hyperedge mode.** `latence-label calibrate` serves a
   judgment UI over `graph-hyperedges.parquet`, which no pipeline writes any more.
4. **Stale text in kept code:**
   - `latence contracts --capability` help says "e.g. 'tokenizer'", a removed seam;
   - the diagram in `stacks/phase_boundary.py`'s docstring still shows an `enrich` stage.
5. **`ROADMAP.md` is entirely pre-removal** (16 lines mention retrieval, hyperedges, SPLADE or
   Qdrant). It belongs to the docs-and-site rewrite on the map.
6. **`docs/anatomy/02` and `03`** still carry the dated "written before the removal" warning.

**Found on the way, not caused by the removal**

7. **A purge keeps the document's file name.** `latence purge` says "NOTHING is retained".
   `CorpusStore._reconcile_manifest` rewrites only the fingerprint and the record count of each
   Corpus Version manifest and keeps the delta lineage verbatim, and `DeltaEntry` carries
   `file_name`. So after a purge the document's id, content hash and file name are still in
   `_latence/corpus/wal/v*/version.json`. A file name can be personal data. **Owner's
   decision:** scrub those fields on purge, or correct the claim. Either way the help text and
   the behaviour have to agree, with a test.
8. `latence run` on a shipped `stacks/*.yaml` exits 0 after writing into a literal
   `{STORAGE_URI}` directory with no documents (reported by the page rewrite; not reproduced
   again here).
9. `manifest.json` redacts every key containing "token", which also hides the chunk budget
   (`max_tokens`, `overlap_tokens`, `min_tokens`) (reported by the page rewrite; not reproduced
   again here).
10. `RELEASE-GATES.md` links `issues/e2e-extended/RUNBOOK.md`, which does not exist. The same
    dangling link is on `main`.
11. The deploy artifacts' own comments disagree with what they do: the compose file and the
    Helm notes say a run writes a knowledge graph, but the shipped served-OCR stack has no graph
    stage; the Helm CronJob is described as incremental but passes a fixed run id (reported by
    the page rewrite; not reproduced again here).

## Done when

- Items 1 and 7 are decided by the owner and implemented, each with a test that was red first.
- Items 2–4 are fixed or removed, with the tests that covered them adjusted to the kept
  behaviour and nothing weakened.
- Items 8–11 are each reproduced, then fixed or recorded as intended.
- Items 5 and 6 are closed by the docs-and-site rewrite or here.
