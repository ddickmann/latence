# Task: inventory the carve-out: what is kept, dropped and upgraded, and the coupling

Type: task
Mode: AFK
Status: resolved
Blocked by: 

## Question

Produce the exact keep / drop / upgrade list for every package and every `latence-core`
module group, with the import sites where kept code reaches dropped code.

The list must decide, per ADR-0068:

- what "retrieval" removes: hyperedges, Context Enrichment, `graph_features`, the BM25 and
  context-header export columns, the retrieval capability protocols and descriptors, the
  stacks and tests;
- what stays only as a resolution signal (the embedders).

*Remove the retrieval tooling* builds from this list.

## Answer

Resolved 2026-11-08, AFK, from a read-only sweep of the tree at `d96b30d`. Line counts are
source lines in `src/`.

**Packages**

| Package | LOC | Verdict |
|---|---|---|
| latence-retrieval | 10,790 | **drop** |
| latence-sink-qdrant | 2,644 | **drop** |
| latence-splade / latence-muvera | 766 / 291 | **drop** |
| latence-embedder-st / -endpoint | 342 / 206 | **keep, as a resolution signal only.** A sub-Provider of `disambiguation.glinker`, `disambiguation.embedding` and type consolidation's semantic rung |
| latence-extract-gliner2, latence-gliner25, latence-ner-gliner, latence-ner-endpoint, latence-relation-gliner, latence-relation-llm | 1,063 / 2,046 / 379 / 329 / 516 / 519 | **upgrade.** The set is re-selected in *the text-path upgrade*; packages that lose the bake-off are retired there |
| latence-pii-gliner / -gliner2 / -presidio | 406 / 1,096 / 422 | keep the PII path; the GLiNER ones are upgraded with the text path |
| latence-schema-inducer, latence-label-profiler | 2,006 / 1,727 | keep (upgrade the inducer with the text path) |
| latence-disambig-glinker, latence-disambig-embedding | 3,046 / 599 | keep. Re-decided in *the upgrade of the resolution engine*. glinker is the runtime hotspot (73% of the dogfood run) |
| latence-linkpred-ultra | 2,806 (1,194 vendored) | keep (owner) |
| parsers ×8 | 196–957 each | keep. The OCR set may be trimmed with the text path |
| latence-benchmark | 11,010 | **split:** keep parse, redaction, redfm and `labeling/`; drop `campaign/retrieval_*` and the S5 retrieval matrix |
| latence-console, latence-demo | 3,471 / 1,420 | keep; strip hyperedge rendering (`render.py`, `viewer.py`) |
| latence-runner-airflow, latence-source-sharepoint | 410 / 2,036 | keep as optional. Revisited in *runtime, storage and deployment* |

**`latence-core` (54,734 LOC)**

- **Drop:**
  - `hyperedge.py` (971), `hyperedge_store.py` (261), `sentences.py` (174, used only by hyperedges);
  - `bm25.py` (341), `signals.py` (260), `splade_index.py` (246);
  - `stages/context_enrichment.py` (501);
  - `redaction_survival.py` (56), once the export and SPLADE paths are gone;
  - `graph_features.py` (234) and its stage (142). Decision: dropped. Its PageRank and
    community features had retrieval expansion as their only consumer. If ontology work later
    wants graph analytics, it comes back as a new Capability with its own consumer.
- **Trim:**
  - `cli.py`: the `retrieval` Typer group, :73–81 and :1815–1930;
  - `setup/recipe.py` and `setup/wizard.py`: 37 bm25, embedder-as-signal and sparse references;
  - `stages/export.py`: the BM25 accumulator (:42), redaction survival (:62), and about 205
    context-header and embedder references (:102–450). The corpus export stays, as a record of
    processed text;
  - `stages/graph_assembly.py`: the opt-in hyperedge path (:50), plus the chunk and mention
    feed into it at `runner.py:1542`;
  - `stages/graph_export.py`: the hyperedge KV sidecar (:70);
  - `conformance/cases.py`: `SparseHashingEmbedder` (:1854);
  - `capability.py:551–1046`: the ContextEnricher, SparseEmbedder, ComposingSparseEmbedder,
    MultiVectorEmbedder, FdeConverter and Tokenizer protocols;
  - `capability_descriptor.py:636–679` and `pipeline.py:99`: the CONTEXT_ENRICHMENT kind;
  - `contracts.py`: ContextHeader (:1017), GraphHyperedge, HyperedgeRelation,
    HyperedgeSourceRef and `GraphScope.HYPEREDGE` (:2359–2515). This bumps `SCHEMA_VERSION`
    from 21;
  - `quality.py` and `quality_report_builder.py`: the HyperedgeQuality, token_cost and
    context_enrichment sections;
  - `envspec.py`, `compat.py`, `placement.py`, `providers/perf.py`: remove the entries for
    dropped Providers.
- **Keep:**
  - contracts, Capability registry, pipeline and stage execution, Runner, run_store, Storage,
    errors, tracing, observability;
  - the text path (chunking, offset/page maps, induced labels, blocking);
  - screening, entity, relation, redaction, header_refs, inline_refs, profiling;
  - disambiguation, type consolidation, graph, graph completion, graph export (nodes and
    edges);
  - delta and state, reuse overlay;
  - quality and gold set, bake-off, tune, conformance;
  - `embedding.py` (the `embedding.hashing` offline fallback);
  - `iceberg.py` (its UC / Polaris REST registration is useful later).
- **Tests:** 19 of the 190 core test files exercise dropped modules (`test_hyperedge*`,
  `test_bm25*`, `test_context_enrichment*`, `test_splade_index`, `test_signal_generators`,
  `test_retrieval_*`, and others). They go with their modules.
- **Stacks:** 6 of 11 use retrieval signals: context-enrichment, gpu-sota*, graph-features,
  retrieval-signals. The retrieval-only ones are deleted. Mixed ones keep their extraction and
  graph Stages and lose the retrieval keys.
- **Docs:**
  - `docs/retrieval-tooling.md`, `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`,
    `docs/TUTORIAL-retrieval-tooling.md`, `docs/TUTORIAL-rfc-retrieval-proof.md`,
    `docs/embedders.md` (rewritten as a resolution-signal page);
  - the retrieval sections of `docs/agent-briefing.md`, `llms.txt`, README and `mkdocs.yml`.

**Coupling.** No kept package imports a dropped package. Core reaches retrieval code only
through the import sites listed under *Trim* and through optional config branches, so the
removal is one moderate slice: *Remove the retrieval tooling*.

**Surfaced for other tickets:**

- The repo records no MRR or Hits@k for ULTRA. The 82-RFC dogfood run took 13,738 records in
  and emitted 0 edges, and the report cannot tell a skip from a calibration refusal. This goes
  to *Record the baseline*.
- Provenance requires file fields; mentions require char and page spans and a chunk id;
  Evidence has no locator of its own; `MergeDecision` has no actor; no must-link or
  cannot-link exists; `TypeVocabulary` is flat; the TTL uses a private `lg:` vocabulary.
  This goes to *the destination and the domain model*, *the upgrade of the resolution engine*,
  and *ontology induction and the output standards*.
