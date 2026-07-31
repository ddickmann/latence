# 01 — Research: spec-vs-repo evidence audit

Type: research
Status: resolved
Blocked by: —

## Question

What does the repo already provide vs what [the spec](../spec.md) assumes? Verify concretely, with file/line evidence:

a. **RelationMention argument offsets** — do relation records (ADR-0023, ADR-0031) carry head/tail argument char spans usable for EU segmentation (spec 1.2 rule 2 needs "GLiREL argument offsets")?
b. **glinker link confidence** — is per-mention link confidence persisted through Disambiguation into exported records (ADR-0026/0046)? Spec 1.3/1.5/6.1 needs it per EU.
c. **gliner_relex fused provider** — output shapes; can it run per sentence-window? What relation confidence does it emit?
d. **Existing knapsack Packer** (`packages/latence-retrieval/src/latence_retrieval/pack.py`) — interface fit for the frozen-value refine phase (spec 3.3): can it be reused unchanged as the final packer?
e. **Graph parquet schema** (ADR-0027/0051, Export) — what of `graph-nodes/edges.parquet` + denormalised chunk columns is reusable for the sidecar store (spec 1.6)?
f. **Sentence splitting** — what exists in Chunk today; is pysbd or spaCy senter already a dependency anywhere; DE/EN abbreviation safety.
g. **Parked-branch overlap** — do `lane-F2-graph-leg-and-fusion`, `w17-incremental-extraction`, `fuzzy-token-guard`, or other unmerged branches contain work this effort would duplicate or should absorb?
h. **Benchmark harness** — where a MuSiQue/HotpotQA supporting-fact eval would land (`latence-benchmark` vs `benchmark/e2e`), and what the current chunk-topk baseline path is.

Deliver a gap table: spec requirement → exists / partial / missing → pointer.

## Answer

Audited on the working checkout (branch `t-wiz-2-lexicon-candidates`, which has `main` as ancestor; all cited files are identical in intent on `main`). All paths relative to repo root.

### Gap table

| Spec requirement | Status | Pointer |
|---|---|---|
| 1.2 r1 — abbreviation-safe sentence splitter (pysbd/spaCy senter, DE/EN) | **missing** | `packages/latence-core/src/latence_core/chunking.py:226` (`_SENTENCE_RE = r"[.!?](?:\s|$)"` — naive regex); pysbd nowhere in repo; spaCy only inside `latence-pii-presidio` |
| 1.2 r2 — relation argument (head/tail) char offsets | **partial** | Not on `RelationMention` (covering span only, `packages/latence-core/src/latence_core/contracts.py:1176-1180`); recoverable by joining `head_mention_id`/`tail_mention_id` → `EntityMention.provenance.char_start/end` (true original offsets, ADR-0031) |
| 1.3/1.5/6.1 — per-mention link confidence persisted | **missing** | Neural link score is dropped: `packages/latence-disambig-glinker/src/latence_disambig_glinker/linker.py:241-258` copies only `canonical_entity_id` from `LinkedEntity` (which has `score`, `backends.py:151-160`). `CanonicalEntity` has no per-mention confidence field (`contracts.py:1900-1955`) |
| 1.3 — hyperedge record (entity set m≥2, tau_text, token_cost, support, has_relations) | **missing** | Graph is strictly binary: `GraphEdge` = source/target node pair (`contracts.py` GraphEdge; `stages/graph_export.py:210-238`) |
| 1.4 — alias table persisted; entity `description` composed from relations | **missing** | Aliases exist only in glinker-internal `CanonicalEntityRecord` (`latence-disambig-glinker/.../models.py:44+`) and are dropped at the `CanonicalEntity` contract (no `aliases` field, `contracts.py:1900-1955`); no description composition anywhere |
| 1.4 — dedup/merge with `support` counter | **partial** | Edge folding exists with max-confidence, but no support count: `stages/graph_assembly.py:406-418` (folds duplicate relations, keeps `confidence = max`, no counter) |
| 1.5 — calibrated γ (isotonic, weighted combination) | **missing** | No calibration code anywhere; raw scores flow through unclamped except [0,1] clamps |
| 1.6 — sidecar store (incidence CSR, entity postings, A_E, eu_text KV, degrees) | **missing** | Nothing exists. Nearest reusable substrate: `graph-nodes/edges.parquet` (`stages/graph_export.py:141-238`) + DuckDB-over-parquet traversal (`latence-retrieval/src/latence_retrieval/multihop.py:173+`, ADR-0051) |
| 1.6 — entity_vecs / eu_vecs collections | **missing** | Vector sink stores chunk embeddings only (`latence-sink-qdrant`); no entity or EU embeddings |
| 1.7 — precomputed `token_cost` per item | **partial** | Consumer side exists: `Packer` `token_key` reads a precomputed count from candidate metadata (`latence-retrieval/src/latence_retrieval/pack.py:223-226,345-366`); nothing writes token counts at index time |
| Part 2 — entry + RWR confidence propagation | **missing** | Nearest analogues: `GraphRetriever` first-stage graph leg (`graph_retrieval.py:14-31`) and hop-decay `MultiHopExpander` (`multihop.py`) — bonus/traversal, not propagation |
| 3.3 — existing knapsack reusable as frozen-value final packer | **partial (yes, with caveats)** | `pack.py:200-434` — see (d) below |
| 3.4 — chain assembly/ordering | **missing** | No component/BFS-ordering code; packer output is score-desc order (`pack.py:342`) |
| 3.5 — UtilityModel protocol + selection-trace logging | **missing** | No such seam; `Packer` value = scaled `Candidate.score` + metadata signals (`pack.py:368-382`) |
| Eval — MuSiQue/HotpotQA supporting-fact recall@budget | **missing** | No MuSiQue/HotpotQA anywhere (repo-wide grep). Harness seams exist: `packages/latence-benchmark/src/latence_benchmark/datasets.py` (pinned acquisition + licence gate), `benchmark/rfc/run_benchmark.py` (recall@k engine seam) |

### a. RelationMention argument offsets

`RelationMention` (`packages/latence-core/src/latence_core/contracts.py:1151-1226`) carries `head_mention_id`/`tail_mention_id` + denormalised `head_text`/`tail_text`, but its Provenance span is the **covering span** only — "from the earlier mention's start to the later mention's end" (`contracts.py:1176-1180`, enforced by validator at 1211-1226). Per-argument offsets are NOT on the relation record. They ARE recoverable with one join: each endpoint `EntityMention` must carry `char_start`/`char_end` (validator `contracts.py:1131-1146`) in **true original-markdown coordinates** — ADR-0031 (`docs/adr/0031-...md`) fixed all four emitting providers to resolve model-local offsets through `OffsetIndex(chunk.offset_map)` (e.g. `latence-relation-gliner/src/latence_relation_gliner/provider.py:388-393`). At extraction time the argument spans exist transiently: the raw relex relation dicts carry `head`/`tail` sub-dicts with chunk-local `start`/`end` (`provider.py:367-371` comment documents the real contract), resolved via `by_span` (`provider.py:386-414, 462-473`) and then discarded. **Verdict: partial — spec 1.2 rule 2 is satisfiable via the mention-id join; no schema change strictly required, but the offsets are one indirection away, not "GLiREL argument offsets" on the relation.**

### b. glinker link confidence

Three distinct confidences exist; the one the spec needs is the one that is dropped:
- **NER mention confidence** — persisted (`EntityMention.confidence`, `contracts.py:1117`).
- **Merge-decision confidence** — persisted per merge, not per mention (`MergeDecision.confidence`, `contracts.py:1872-1898`; ADR-0026/0046 "no silent over-merge").
- **Neural link confidence** — `LinkedEntity.score` (`latence-disambig-glinker/src/latence_disambig_glinker/backends.py:151-160`) is produced by the backend but `_apply_linking_results` (`linker.py:241-258`) copies only `canonical_entity_id`; the score never reaches any record. Cluster "confidence" is the **mean of member NER scores**, not link scores (`linker.py:222`, fed from `provider.py:283` `score=mention.confidence`), and lands only as cluster-level `Evidence.confidence` (`latence-core/src/latence_core/stages/disambiguation.py:269`) → `GraphNode.confidence` → `graph-nodes.parquet` `confidence` column (`stages/graph_export.py:174`). `CanonicalEntity` (`contracts.py:1900-1955`) has no per-mention confidence, no aliases field. KB linking is binary (`kb_id` or `None`, `stages/disambiguation.py:272,284`).

**Verdict: missing.** Spec 1.5 (γ term w1) and 6.1 (weighted incidence `B[v,e] = link_conf · role_weight`) need per-mention link confidence; today it would have to be added to the glinker result-application path and threaded into an exported record. Relation-side confidence, by contrast, survives end-to-end: `RelationMention.confidence` → `NormalizedRelation.confidence` (`contracts.py:1985`) → `GraphEdge.confidence` → `edges.parquet`.

### c. Fused providers — shapes, windows, relation confidence

Two fused providers, both implement `FusedEntityRelationExtractor` returning `FusedExtraction {mentions: [EntityMention], relations: [RelationMention]}` (`latence-core/src/latence_core/capability.py:255-267`):
- **`relation.gliner_relex`** (`latence-relation-gliner/src/latence_relation_gliner/provider.py`): raw entity dicts carry chunk-local `start/end/text/label/score`; relation dicts carry `head`/`tail` sub-dicts (`start/end/text/type/entity_idx`) plus **relation-level `score`** (`provider.py:367-371`), emitted as `RelationMention.confidence` (`provider.py:446`). Model window bounded at `max_len` 768 tokens (`provider.py:112-116, 208`).
- **`fused_entity_relation.gliner2`** (`latence-extract-gliner2/src/latence_extract_gliner2/provider.py`): batched `batch_extract_entities`/`batch_extract_relations` with `include_spans=True, include_confidence=True` (`provider.py:22-26, 203-205`). gliner2 1.3.2 carries confidence **only on head/tail endpoints**, so relation confidence = min of endpoint confidences, fallback 1.0 (`_relation_confidence`, `provider.py:983-1000`).

**Per-sentence-window:** neither has a sentence/window parameter — both consume `Iterable[ChunkRecord]` (`extract_fused` at gliner-relex `provider.py:263`, gliner2 `provider.py:598`) and extract over the whole chunk `content`. Running per sentence window means constructing synthetic sentence-sized `ChunkRecord`s (feasible — `ChunkRecord` carries `content` + `offset_map` + provenance, `contracts.py:947-1012` — and both providers batch across chunks so many small texts are efficient), or slicing post-hoc using mention offsets. **Verdict: partial — batching supports small texts well, but the sentence-window loop of spec 1.2 does not exist.**

### d. Existing knapsack Packer as the frozen-value refine phase

`Packer` (`latence-retrieval/src/latence_retrieval/pack.py:200-434`) is a deterministic greedy QKP: value = max-ratio-scaled `Candidate.score` + metadata `value_signals` (`pack.py:368-382, 491-506`), cost = `token_key` metadata override else `TokenCounter` (`pack.py:345-366`), redundancy = `similarity_key` map / `cluster_key` / lexical Jaccard weighted by `redundancy_weight` (`pack.py:423-434`), greedy marginal-per-token with ties by ascending id (`pack.py:384-421`).

Fit as spec 3.3's final packer with FROZEN values: **workable** — put `v(e|A_final)` into `Candidate.score`, `token_cost` into `metadata[token_key]`, set `redundancy_weight: 0` (pure additive budgeted knapsack, `pack.py:213-215`). Caveats, honestly stated: (1) it is the **same greedy family**, not a DP/ILP — the spec's premise that the existing solver "repairs greedy's boundary mistakes (swap a large low-density item for two small ones)" is **false for this solver**: re-running greedy over frozen values approximately reproduces the greedy phase and performs no swap repair; (2) stop condition is `marginal <= 0` (`pack.py:407`) — no `theta_min`; (3) `_relevance_scaled` clamps negative scores to 0 and divides by the max (monotone, ordering-safe, but values are rescaled); (4) minimum cost 1 token (`pack.py:366`); (5) output order is score-desc (`pack.py:342`), not chain order — 3.4 must reorder. The `Candidate` contract is engine-agnostic (id/text/score/metadata) so EU→Candidate adaptation is trivial. **Verdict: partial — reusable unchanged as *a* final packer, but it adds no boundary-repair value beyond the greedy phase; a true swap/DP refine would be new code.**

### e. Graph parquet schema reusable for the sidecar

`stages/graph_export.py`: `*-nodes.parquet` columns (`:156-208`): `node_id, entity_id, type, canonical_name, kb_id, confidence, member_mention_ids (JSON), source_document_ids (JSON), properties (JSON), evidence (JSON)` + optional first-class `centrality` (float64) and `community` (string) when graph features ran (ADR-0051). `*-edges.parquet` (`:210-238`): `edge_id, relation_id, source_node_id, target_node_id, label, confidence, properties, evidence`. Denormalised chunk columns (`stages/export.py:285-325`, ADR-0039/0051): `context_header, kg_node_ids, neighbor_node_ids, kg_node_community, kg_node_centrality` — per-chunk entity node ids give a ready chunk↔entity incidence bootstrap. Also reusable: `DuckDbGraphSource` recursive-CTE traversal over the edge parquet (`latence-retrieval/src/latence_retrieval/multihop.py:173+`) and `GraphRetriever`/`GraphDocumentSource` (`graph_retrieval.py`) as the pattern for `aliases.duckdb`-style zero-ops sidecar queries.

**Reusable:** entities side of `entities.parquet` (id, name, type, kb_id, confidence, degree-via-centrality), relations source for hyperedge `relations[]`, chunk↔entity incidence seed. **Missing for spec 1.6:** `aliases[]`, `description`, `tau_text`/EU spans, `token_cost`, `support`, `has_relations`, incidence CSR, `A_E`, postings, degrees, eu_text KV — i.e., everything hyperedge-shaped is new.

### f. Sentence splitting

The only splitter is `_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")` (`latence-core/src/latence_core/chunking.py:226`) used by `_sentence_spans` (`:341-359`) / `chunk_document_by_sentence` (`:296`) for the sentence-window Chunker (`stages/chunk.py:117-181`). It is **not abbreviation-safe**: "Dr. Meier", "z. B.", "Inc. " all split. pysbd appears nowhere in the repo (grep over all `*.py`/`*.toml`). spaCy is a dependency **only** of `latence-pii-presidio` (its `en_core_web_sm` NLP engine, `latence-pii-presidio/src/latence_pii_presidio/provider.py:285-291`) — not importable from core without that optional package, and English-only as configured. **Verdict: spec 1.2 rule 1 needs a new dependency (pysbd MIT, or spaCy senter multi-lang) or a hand-rolled DE/EN abbreviation list; nothing existing qualifies.** Useful adjacent asset: sentence spans are offset-preserving (`_sentence_spans` tiles the text), matching the EU-span discipline.

### g. Parked-branch overlap

No duplication risk from the named branches — all their content is already integrated into `main` (branch tips are patch-id-different because integration was rebase/squash-style, but content-verified):
- `lane-F2-graph-leg-and-fusion`: `git log lane-F2-graph-leg-and-fusion ^main` is **empty** — fully merged (it delivered the `GraphRetriever` leg + fusion now on main).
- `w17-incremental-extraction`: 4 commits not ancestor-reachable, but ADR-0043 and the delta/incremental code are on main; `git diff main w17-…` shows the branch ~160k lines *behind* main — it is stale, not unmerged work.
- `fuzzy-token-guard`: its unique commit `8c47344` (distinguishing-token guard in `entity_resolver.py`) is content-present on main (`git grep "distinguishing" main -- packages/latence-disambig-glinker/src/latence_disambig_glinker/entity_resolver.py` → 5 hits).
- `wip/c1-sidecar-superseded` (`9840f32`): a **page-map** sidecar design, explicitly superseded; unrelated to the hypergraph sidecar despite the name.
- Sweep of all 60+ local branches (`git rev-list --count <b> ^main`): every branch with "unique" commits is a pre-integration pod/wave branch (`_podfinal`, `wizard-sota`, `sota-final`, …); none contains hyperedge, EU, propagation, or knapsack-refine work. Nothing to absorb.

### h. Benchmark harness landing spot

There is **no** MuSiQue/HotpotQA/supporting-fact code anywhere (repo-wide grep). Two existing harnesses split the concern:
- `packages/latence-benchmark` — extraction-quality harness (NER/RE/parse/redaction) with the right **dataset-acquisition seam**: `DatasetSpec` pinned to full commit SHAs + `licensing.require_usable` gate + `LocalDatasetSource`/`HuggingFaceDatasetSource` (`src/latence_benchmark/datasets.py:1-70`). MuSiQue (CC-BY-4.0) / HotpotQA (CC-BY-SA-4.0) dataset specs belong here.
- `benchmark/rfc/run_benchmark.py` — the e2e retrieval-comparison pattern: configs `dense/bm25/splade/fused`, `recall_at_k` over gold chunk ids (`:198-206`), a lazy real-engine seam (`build_real_engine`) so offline CI runs on faithful doubles, and the **current chunk-topk baseline path**: dense = query embedding → `QdrantBackend` top-k; fused = dense+BM25+SPLADE → RRF → KG injection + `MultiHopExpander` (2-hop, hub-filtered) → cross-encoder `Reranker` → `Packer` @16k (`:9-22`, `DEFAULT_TOP_K=10`, `DEFAULT_PACK_BUDGET=16_000` at `:63-64`). A supporting-fact recall@budget eval is a new sibling (the spec's `eval/supporting_facts.py`) that should reuse this engine-seam pattern and score against **supporting-fact chunk/EU ids** instead of gold chunk sets; the baseline curve to dominate is exactly `run_benchmark.py`'s fused/dense recall@k lists truncated by the same token budget.
