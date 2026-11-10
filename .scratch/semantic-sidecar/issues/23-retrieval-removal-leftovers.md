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

1. *(Done 2026-11-10; owner chose "replace with the new bars".)* **The release gates still
   demanded removed capabilities.** `RELEASE-GATES.md` Gates 2 and 3,
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

7. *(Done 2026-11-10; owner chose "scrub them on purge".)* A purge now drops every live
   listing of the document from every Corpus Version, and with it the file name and content
   hash. **One deviation from the owner's choice:** the content-addressed id stays, as a
   DELETED tombstone. `DeltaRunner._prior_deletions` reads it to keep a Source that still serves
   the file from re-ingesting the document, so removing it would let an erased document come
   back. The help text says so. Original finding: **a purge kept the document's file name.** `latence purge` says "NOTHING is retained".
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

**Statements the page rewrite left in place without checking them against code**

12. `docs/deployment.md`: the helm and docker commands (checked against templates and values,
    not executed); the "~24 GB GPU" sizing for the default LLM; the `huggingface-cli` steps; the
    2.7× OCR figure (cited to the serving guide, not re-measured); the Airflow "tested, not yet
    operated" paragraph beyond `build_dag` and the CI job existing.
13. `docs/tutorials/05-production.md` §6: `env placement`, `tune` and `--tuned` are described
    from `--help` text only.
14. `docs/tutorials/04-bringing-your-own-provider.md` §8: the `AdapterBase` helper names are
    verified; the descriptions of what they do are not line-checked.
15. `docs/anatomy/index.md`: the "why this one" model rationales, the one-line corpus
    descriptions and the Gemma sentence.
16. `docs/BENCHMARKS-E2E.md`: the literature figures (Hits@1 0.26–0.31).
17. `docs/guides/benchmark-campaign.md`: stage behaviour is described from reading the code; no
    stage was run.

## Done when

- Items 1 and 7 are decided by the owner and implemented, each with a test that was red first.
- Items 2–4 are fixed or removed, with the tests that covered them adjusted to the kept
  behaviour and nothing weakened.
- Items 8–11 are each reproduced, then fixed or recorded as intended.
- Items 5 and 6 are closed by the docs-and-site rewrite or here.
- Items 12–17 are each verified or reworded to what can be verified.
